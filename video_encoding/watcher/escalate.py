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
from video_encoding.watcher.registry import JobRecord
from video_encoding.watcher.state import WatcherState

DEFAULT_ESCALATION_COOLDOWN_S = 3600
DEFAULT_ESCALATION_BUDGET_PER_DAY = 10
DEFAULT_MAX_TURNS = 15

_ALLOWED_TOOLS = (
    "Read,Grep,Glob,"
    "Edit(video_encoding/conf/**),"
    "Bash(video_encoding/scripts/slurm_wrapper.sh:*)"
)


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
        "and by reading video_encoding/conf/**, .claude/state/, and .claude/results/.",
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
