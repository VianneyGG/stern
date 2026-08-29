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

import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from video_encoding.watcher.detectors import Flag
from video_encoding.watcher.epoch_advance import exclude_node
from video_encoding.watcher.notify import notify
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


def _claude_bin() -> str:
    """Absolute path to claude, for subprocesses launched from the watcher.

    Same PATH gap as mesh_tick.py::_uv_bin -- `uv run` from cron only
    prepends the venv's bin dir to PATH, not ~/.local/bin, so a bare
    "claude" resolves fine interactively but raises FileNotFoundError on
    every cron tick, silently disabling every escalation path (quota,
    stalled, repeat-dead, NaN).
    """
    return shutil.which("claude") or str(Path.home() / ".local" / "bin" / "claude")


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
        _claude_bin(),
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


def _resubmit_nan_node(
    *, run_id: str, node: str, round_num: int, repo_root: Path
) -> str | None:
    """Reload this run's MeshRunConfig from mesh_run_configs.json and
    resubmit `node` with a halved inner-LR override, reloading from the
    run's last merged master checkpoint (mesh_round.py's cold-start
    contract already handles a fresh worker picking up master.pt)."""
    from video_encoding.watcher.mesh_tick import (
        load_run_configs,
        resubmit_with_override,
    )

    configs_path = repo_root / "runs" / "state" / "mesh_run_configs.json"
    matching = [c for c in load_run_configs(configs_path) if c.run_id == run_id]
    if not matching:
        return None
    run_config = matching[0]
    registry_path = repo_root / "runs" / "state" / "jobs_registry.json"
    submitted_at = datetime.now(timezone.utc).isoformat()
    return resubmit_with_override(
        run_config,
        node,
        repo_root,
        registry_path,
        submitted_at,
        start_round=round_num + 1,
        override="train.lr_mult=0.5",
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
    """React to a `detect_nan` Flag: exclude the node, then either resubmit
    it once with a scripted halved-LR retry (no `claude -p`; NaN's cause is
    divergence, not something needing LLM judgment) or drain the node for
    the rest of this run. Returns one of "retried_scripted" | "drained" |
    "capped" | "dry_run" | "spawn_failed" for the caller's action log.
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

    job_id = _resubmit_nan_node(
        run_id=run_id, node=node, round_num=round_num, repo_root=repo_root
    )
    if job_id is None:
        return "spawn_failed"

    transcript_dir.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(
        "Scripted NaN remediation (no claude -p) -- exclude, halve inner LR, "
        f"resubmit.\nRun: {run_id}\nNode: {node}\nRound: {round_num}\n"
        f"Detail: {flag.detail}\nResubmitted as job {job_id}.\n"
    )
    notify(
        f"nan:{run_id}:{node}",
        f"{node} NaN'd on round {round_num} of {run_id}; excluded and resubmitted "
        f"(one retry, halved inner LR) as job {job_id}",
    )
    job_state.auto_resubmit_count += 1
    state.nan_escalations_in_flight[run_id] = now
    return "retried_scripted"
