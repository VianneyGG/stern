"""Watcher state: per-job tracking + the adaptive backoff timer.

Persisted to .claude/state/watcher_state.json. The watcher has no long-lived
process (it's cron-driven, spec §2) -- this file *is* its memory between
invocations: loaded fresh each tick, mutated in memory by tick.run_full_check,
written back atomically by the caller.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

FAST_INTERVAL_S = 60.0
SLOW_INTERVAL_S = 1800.0
BACKOFF_MULTIPLIER = 1.5


@dataclass
class JobState:
    slurm_job_id: str
    auto_resubmit_count: int = 0
    last_escalation_at: float | None = None
    status: str = "healthy"  # healthy|stalled|dead|escalated|resolved


@dataclass
class WatcherState:
    next_full_check_at: float = 0.0
    current_interval_s: float = FAST_INTERVAL_S
    jobs: dict[str, JobState] = field(default_factory=dict)
    escalation_day: str | None = None
    escalation_count_today: int = 0

    def job(self, slurm_job_id: str) -> JobState:
        return self.jobs.setdefault(slurm_job_id, JobState(slurm_job_id=slurm_job_id))


def load_state(path: Path) -> WatcherState:
    if not path.exists():
        # next_full_check_at=0.0 -- any real `now` is >= 0, so the very first
        # tick after deployment always runs a full check immediately.
        return WatcherState()
    payload = json.loads(path.read_text())
    jobs = {k: JobState(**v) for k, v in payload.get("jobs", {}).items()}
    return WatcherState(
        next_full_check_at=payload.get("next_full_check_at", 0.0),
        current_interval_s=payload.get("current_interval_s", FAST_INTERVAL_S),
        jobs=jobs,
        escalation_day=payload.get("escalation_day"),
        escalation_count_today=payload.get("escalation_count_today", 0),
    )


def save_state(path: Path, state: WatcherState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "next_full_check_at": state.next_full_check_at,
        "current_interval_s": state.current_interval_s,
        "jobs": {k: asdict(v) for k, v in state.jobs.items()},
        "escalation_day": state.escalation_day,
        "escalation_count_today": state.escalation_count_today,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def advance_backoff(state: WatcherState, *, now: float, action_taken: bool) -> None:
    """Mutate state's interval/next-check-time per spec §4 step 5."""
    if action_taken:
        state.current_interval_s = FAST_INTERVAL_S
    else:
        state.current_interval_s = min(
            state.current_interval_s * BACKOFF_MULTIPLIER, SLOW_INTERVAL_S
        )
    state.next_full_check_at = now + state.current_interval_s
