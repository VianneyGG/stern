"""Pure, read-only detectors over squeue/registry/mtime state.

No I/O, no subprocess calls -- every one is a plain pytest against fixture
data (spec §6). `now` is always injected as an epoch-seconds float so tests
never depend on wall-clock time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from video_encoding.watcher.registry import JobRecord
from video_encoding.watcher.state import JobState

DEFAULT_STALL_THRESHOLD_S = 2700
DEFAULT_QUOTA_THRESHOLD_PCT = 90


@dataclass(frozen=True)
class Flag:
    subject: str  # slurm_job_id for dead/stalled; "quota:<location>" for quota
    kind: str  # "dead" | "dead_after_resubmit" | "stalled" | "quota"
    detail: str


def _parse_iso(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def detect_dead(
    registry: list[JobRecord],
    squeue: dict[str, dict],
    result_mtimes: dict[str, float],
    job_states: dict[str, JobState],
    *,
    now: float,
) -> list[Flag]:
    flags: list[Flag] = []
    for record in registry:
        jid = record.slurm_job_id
        if jid in squeue:
            continue  # still known to squeue -- not dead
        if jid in result_mtimes:
            continue  # finished with a result -- done, not dead
        state = job_states.get(jid)
        if state is not None and state.status in ("resolved", "escalated"):
            continue  # already handled this tick or a prior one
        if state is not None and state.auto_resubmit_count > 0:
            flags.append(
                Flag(
                    jid,
                    "dead_after_resubmit",
                    f"job {jid} dead again after {state.auto_resubmit_count} auto-resubmit(s)",
                )
            )
        else:
            flags.append(
                Flag(
                    jid,
                    "dead",
                    f"job {jid} not in squeue, no result JSON ever appeared",
                )
            )
    return flags


def detect_stalled(
    squeue: dict[str, dict],
    result_mtimes: dict[str, float],
    registry: list[JobRecord],
    *,
    stall_threshold_s: int = DEFAULT_STALL_THRESHOLD_S,
    now: float,
) -> list[Flag]:
    registry_by_id = {r.slurm_job_id: r for r in registry}
    flags: list[Flag] = []
    for jid, info in squeue.items():
        if info["state"] != "RUNNING":
            continue
        mtime = result_mtimes.get(jid)
        if mtime is None:
            record = registry_by_id.get(jid)
            mtime = _parse_iso(record.submitted_at) if record else None
        if mtime is None:
            continue  # no reference point at all -- nothing to compare
        age_s = now - mtime
        if age_s > stall_threshold_s:
            flags.append(
                Flag(
                    jid,
                    "stalled",
                    f"job {jid} RUNNING but no progress in {int(age_s)}s",
                )
            )
    return flags


def detect_quota(
    disk_usage_pct: dict[str, float],
    *,
    threshold_pct: float = DEFAULT_QUOTA_THRESHOLD_PCT,
) -> list[Flag]:
    flags: list[Flag] = []
    for location, pct in disk_usage_pct.items():
        if pct >= threshold_pct:
            flags.append(
                Flag(
                    f"quota:{location}",
                    "quota",
                    f"{location} at {pct:.1f}% (threshold {threshold_pct}%)",
                )
            )
    return flags
