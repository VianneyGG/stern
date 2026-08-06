# video_encoding/watcher/escalate.py
"""Headless `claude -p` escalation for failure modes that need judgment:
repeat death after auto-resubmit, a stalled RUNNING job, or quota pressure
(spec §3 Escalator, §5 safety guardrails).

Cooldown + daily budget are the two automated brakes (spec §5, guardrails
4-5); the transcript-before-action guarantee (guardrail 6) is why this
writes stdout to disk via a live file handle (subprocess.Popen) instead of
subprocess.run's capture-then-write-at-exit -- a kill mid-call still leaves
whatever was written on disk.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

from video_encoding.watcher.detectors import Flag
from video_encoding.watcher.epoch_advance import exclude_node
from video_encoding.watcher.registry import JobRecord
from video_encoding.watcher.state import WatcherState

DEFAULT_ESCALATION_COOLDOWN_S = 3600
DEFAULT_ESCALATION_BUDGET_PER_DAY = 10
DEFAULT_MAX_TURNS = 15
DEFAULT_NAN_CONCURRENCY_TIMEOUT_S = 1800.0

_ALLOWED_TOOLS = (
    "Read,Grep,Glob,"
    "Edit(video_encoding/conf/**),"
    "Bash(video_encoding/scripts/slurm_wrapper.sh:*)"
)
# No Edit at all -- unlike the generic escalation above, a NaN diagnosis may
# only read state and resubmit via SLURM, never touch a live experiment
# config while other nodes in the same run are still training against it.
_NAN_ALLOWED_TOOLS = "Read,Grep,Glob,Bash(video_encoding/scripts/slurm_wrapper.sh:*)"


def _today_utc(now: float) -> str:
    return datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y-%m-%d")


def should_escalate(
    state: WatcherState,
    subject: str,
    *,
    now: float,
    cooldown_s: int = DEFAULT_ESCALATION_COOLDOWN_S,
    budget_per_day: int = DEFAULT_ESCALATION_BUDGET_PER_DAY,
) -> bool:
    today = _today_utc(now)
    count_today = state.escalation_count_today if state.escalation_day == today else 0
    if count_today >= budget_per_day:
        return False
    job_state = state.jobs.get(subject)
    if job_state is not None and job_state.last_escalation_at is not None:
        if now - job_state.last_escalation_at < cooldown_s:
            return False
    return True


def record_escalation(state: WatcherState, subject: str, *, now: float) -> None:
    today = _today_utc(now)
    if state.escalation_day != today:
        state.escalation_day = today
        state.escalation_count_today = 0
    state.escalation_count_today += 1
    job_state = state.job(subject)
    job_state.last_escalation_at = now
    job_state.status = "escalated"


def build_prompt(flag: Flag, *, registry_entry: JobRecord | None) -> str:
    lines = [
        "A SLURM training watcher flagged an incident that needs judgment.",
        f"Subject: {flag.subject}",
        f"Kind: {flag.kind}",
        f"Detail: {flag.detail}",
    ]
    if registry_entry is not None:
        lines += [
            f"Experiment: {registry_entry.experiment}",
            f"Run tag: {registry_entry.run_tag}",
            f"Submitted at: {registry_entry.submitted_at}",
            f"Nodelist: {registry_entry.nodelist}",
        ]
    lines += [
        "",
        "Diagnose using squeue/sacct (via video_encoding/scripts/slurm_wrapper.sh)",
        "and by reading video_encoding/conf/**, runs/state/, and runs/results/.",
        "If action is warranted, you may resubmit (sbatch), kill (scancel), or",
        "edit a Hydra config under video_encoding/conf/** -- nothing else.",
        "State your diagnosis and any action taken clearly before finishing.",
    ]
    return "\n".join(lines)


def run_escalation(
    flag: Flag,
    *,
    repo_root: Path,
    transcript_dir: Path,
    registry_entry: JobRecord | None,
    now_str: str,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> Path:
    transcript_dir.mkdir(parents=True, exist_ok=True)
    safe_subject = flag.subject.replace("/", "_").replace(":", "_")
    transcript_path = transcript_dir / f"{now_str}-{safe_subject}.md"

    prompt = build_prompt(flag, registry_entry=registry_entry)
    cmd = [
        "claude",
        "-p",
        "--max-turns",
        str(max_turns),
        "--allowed-tools",
        _ALLOWED_TOOLS,
        prompt,
    ]

    with transcript_path.open("w") as fh:
        try:
            proc = subprocess.Popen(
                cmd, stdout=fh, stderr=subprocess.STDOUT, text=True, cwd=repo_root
            )
            proc.wait()
        except (OSError, subprocess.SubprocessError) as exc:
            fh.write(f"ERROR: failed to launch claude -p: {exc}\n")

    return transcript_path


def build_nan_prompt(flag: Flag, *, run_id: str, node: str) -> str:
    return "\n".join(
        [
            "A DiLoCo mesh training node reported a NaN/Inf train_loss.",
            f"Run: {run_id}",
            f"Node: {node}",
            f"Detail: {flag.detail}",
            "",
            "The node has already been excluded from the current round's merge.",
            "Diagnose using squeue/sacct (via video_encoding/scripts/slurm_wrapper.sh)",
            "and by reading runs/state/ and runs/results/.",
            "If you judge the divergence recoverable, you may resubmit the node's",
            "worker via sbatch (video_encoding/scripts/run_mesh_worker.sh <experiment>"
            f" {run_id}), optionally with a halved inner-LR Hydra override, so it",
            "reloads from the run's last merged master checkpoint.",
            "This is the node's ONE automatic retry for this run -- if it NaNs",
            "again, do not resubmit; leave it excluded.",
            "State your diagnosis and any action taken clearly before finishing.",
        ]
    )


def remediate_nan_node(
    flag: Flag,
    *,
    run_id: str,
    node: str,
    round_num: int,
    state: WatcherState,
    repo_root: Path,
    transcript_dir: Path,
    now: float,
    now_str: str,
    dry_run: bool = False,
    concurrency_timeout_s: float = DEFAULT_NAN_CONCURRENCY_TIMEOUT_S,
) -> str:
    """React to a `detect_nan` Flag: exclude the node, then either spawn one
    detached diagnostic `claude -p` retry or drain the node for the rest of
    this run. Returns one of "retried" | "drained" | "capped" | "dry_run" |
    "spawn_failed" for the caller's action log.

    Never blocks: this runs inside watch_training.py's per-tick flock
    (spec: the whole main() holds LOCK_PATH), so a multi-minute `claude -p`
    call via `.wait()` would stall every subsequent 60s tick -- the exact
    stall this watcher exists to prevent. `subprocess.Popen` is spawned
    detached (`start_new_session=True`) and never waited on; the report is
    read back, if at all, on a later tick.
    """
    exclude_node(
        state, node, f"nan: train_loss NaN/Inf on round {round_num} of {run_id}"
    )

    safe_node = node.replace("/", "_")
    transcript_path = transcript_dir / f"nan-{run_id}-{safe_node}-{now_str}.md"

    if dry_run:
        # Deliberately still writes a report -- the point of dry-run on
        # first deploy is visibility into what WOULD have happened, not
        # silence, while `claude -p` itself stays uninvoked.
        transcript_dir.mkdir(parents=True, exist_ok=True)
        transcript_path.write_text(
            "DRY RUN -- no claude -p invocation, no resubmit.\n"
            f"Run: {run_id}\nNode: {node}\nRound: {round_num}\n"
            f"Detail: {flag.detail}\n"
        )
        return "dry_run"

    job_state = state.job(flag.subject)
    if job_state.auto_resubmit_count > 0:
        # Already used this node+run's one retry and it NaN'd again --
        # drain. exclude_node above already keeps it out of the round;
        # this just stops any further escalation for this subject.
        job_state.status = "resolved"
        return "drained"

    in_flight_since = state.nan_escalations_in_flight.get(run_id)
    if in_flight_since is not None and now - in_flight_since < concurrency_timeout_s:
        return "capped"

    transcript_dir.mkdir(parents=True, exist_ok=True)
    prompt = build_nan_prompt(flag, run_id=run_id, node=node)
    cmd = [
        "claude",
        "-p",
        "--max-turns",
        str(DEFAULT_MAX_TURNS),
        "--allowed-tools",
        _NAN_ALLOWED_TOOLS,
        prompt,
    ]

    fh = transcript_path.open("w")
    try:
        subprocess.Popen(
            cmd,
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=repo_root,
            start_new_session=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        fh.write(f"ERROR: failed to launch claude -p: {exc}\n")
        fh.close()
        return "spawn_failed"
    # Closing our copy of fh does not truncate the child's writes -- Popen
    # dup'd the fd for the child before returning (POSIX), and this process
    # exits right after main() returns, so holding it open would only leak
    # an fd across ticks.
    fh.close()

    job_state.auto_resubmit_count += 1
    state.nan_escalations_in_flight[run_id] = now
    return "retried"
