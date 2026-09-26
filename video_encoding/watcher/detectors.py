"""Pure, read-only detectors over squeue/registry/mtime state.

No I/O, no subprocess calls -- every one is a plain pytest against fixture
data (spec §6). `now` is always injected as an epoch-seconds float so tests
never depend on wall-clock time.
"""

from __future__ import annotations

import json
import math
from collections.abc import Collection
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from video_encoding.watcher.registry import JobRecord
from video_encoding.watcher.state import JobState, WatcherState

DEFAULT_STALL_THRESHOLD_S = 2700
DEFAULT_QUOTA_THRESHOLD_PCT = 90


@dataclass(frozen=True)
class Flag:
    subject: (
        str  # slurm_job_id for dead/stalled; "quota:<location>"; "nan:<run_id>:<node>"
    )
    kind: str  # "dead" | "dead_after_resubmit" | "stalled" | "quota" | "nan" | "nan_after_retry"
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


DEFAULT_FOREIGN_GPU_PROBE_INTERVAL_S = 300.0


def detect_foreign_gpu_process(
    node: str,
    slurm_job_id: str,
    *,
    gpu_pids: Collection[int] | None,
    job_pid_tree: Collection[int] | None,
) -> Flag | None:
    """Pure: takes pre-fetched probe results (caller does the ssh I/O via
    node_health.query_gpu_process_pids/query_job_pid_tree, throttled like
    _recover_pool's RECOVERY_INTERVAL_S -- no probing every node every tick).
    None input (probe failed) yields no flag, matching truly_idle's
    fail-closed-on-unknown stance inverted for this read-only check: an
    unreachable probe should not manufacture a false positive."""
    if gpu_pids is None or job_pid_tree is None:
        return None
    foreign = set(gpu_pids) - set(job_pid_tree)
    if not foreign:
        return None
    return Flag(
        subject=f"foreign_gpu:{slurm_job_id}:{node}",
        kind="foreign_gpu_process",
        detail=f"{node}: foreign GPU PID(s) {sorted(foreign)} outside job {slurm_job_id}'s process tree",
    )


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


_DRAINED_STATE_PREFIXES = ("drain", "down")


def detect_drained_nodes(
    node_states: dict[str, str], roster: Collection[str]
) -> list[Flag]:
    """Flag roster nodes SLURM itself has drained/downed (sinfo state),
    distinct from this repo's software-level exclude_node. A node absent
    from node_states (sinfo failed, or the node isn't in the partition)
    yields no flag -- fail-closed-on-unknown, matching
    detect_foreign_gpu_process's stance."""
    flags = []
    for node in roster:
        state = node_states.get(node)
        if state is None:
            continue
        if state.lower().startswith(_DRAINED_STATE_PREFIXES):
            flags.append(
                Flag(
                    subject=f"drain:{node}",
                    kind="node_drained",
                    detail=f"{node}: sinfo state={state}",
                )
            )
    return flags


def _last_json_line(path: Path) -> dict | None:
    last: str | None = None
    try:
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    last = line
    except OSError:
        return None
    if last is None:
        return None
    try:
        return json.loads(last)
    except json.JSONDecodeError:
        return None


def detect_nan(
    run_id: str,
    mesh_shared: Path,
    state: WatcherState,
) -> list[Flag]:
    """Flag mesh nodes whose latest per-round metric is NaN/Inf train_loss.

    Reads mesh_shared/metrics/<run_id>/<node>.jsonl (appended once per round
    by training/mesh_round.py::_append_round_metrics) -- only the LAST line
    per node is live; a node that NaN'd once and then recovered on a later
    round must not keep getting flagged.

    A relaunch of an existing run_id reuses the same metrics directory (unlike
    the checkpoint dir, nothing renames or clears it), so until a worker
    appends its first round the last line is still the *previous* attempt's
    -- typically a NaN, since a failed attempt is what prompted the relaunch.
    On 2026-08-21 that window excluded 8 healthy nodes and shrank one arm's
    roster from 16 to exactly quorum. Records older than the current round's
    start therefore cannot describe this round and are skipped.
    """
    flags: list[Flag] = []
    run_state = state.runs.get(run_id)
    if run_state is not None and run_state.status == "DONE":
        # A DONE run's workers are gone; its last metrics line is frozen at
        # finish time and would otherwise re-flag (and re-escalate) forever.
        return flags
    metrics_dir = mesh_shared / "metrics" / run_id
    if not metrics_dir.is_dir():
        return flags
    for node_file in sorted(metrics_dir.glob("*.jsonl")):
        node = node_file.stem
        subject = f"nan:{run_id}:{node}"
        job_state = state.jobs.get(subject)
        if job_state is not None and job_state.status == "resolved":
            continue  # already drained -- permanently done with this node
        record = _last_json_line(node_file)
        if record is None:
            continue
        if run_state is not None and run_state.round_started_at:
            ts = record.get("ts")
            if isinstance(ts, (int, float)) and ts < run_state.round_started_at:
                continue  # left over from a previous attempt at this run_id
        loss = record.get("train_loss")
        if not isinstance(loss, (int, float)) or not (
            math.isnan(loss) or math.isinf(loss)
        ):
            continue
        kind = (
            "nan_after_retry"
            if job_state is not None and job_state.auto_resubmit_count > 0
            else "nan"
        )
        # Detection-only annotation (design §2): a coarse proxy for "a
        # MERGING->SYNCING transition just preceded this NaN" -- a bad peer
        # may have been averaged in before exclusion, not necessarily this
        # node's own divergence. Reconstructing exact round history from the
        # metrics JSONL is out of scope; this changes the diagnosis in the
        # detail string, not the action.
        poisoned_merge = (
            run_state is not None
            and run_state.status in ("SYNCING", "RUNNING")
            and run_state.round > 0
        )
        detail = (
            f"{node}: train_loss={loss} round={record.get('round')} run_id={run_id}"
        )
        if poisoned_merge:
            detail += (
                " (a MERGING->SYNCING transition just preceded this -- a bad "
                "peer may have been averaged in before exclusion, not "
                "necessarily this node's own divergence)"
            )
        flags.append(Flag(subject, kind, detail))
    return flags
