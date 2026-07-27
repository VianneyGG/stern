"""Orchestrates one full-check tick (spec §4): load inputs, run the 3
detectors, remediate-or-escalate each flag, advance the backoff timer.

Impure by nature (calls remediate/escalate, which shell out) -- the
detectors it calls stay pure; this module is the one place their outputs
turn into subprocess calls. `state` is mutated in place and NOT persisted
here -- the caller (scripts/watch_training.py) owns save_state, so this
function stays trivially testable against an in-memory WatcherState.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from video_encoding.watcher.detectors import detect_dead, detect_quota, detect_stalled
from video_encoding.watcher.escalate import (
    run_escalation,
    should_escalate,
    record_escalation,
)
from video_encoding.watcher.quota import collect_disk_usage_pct
from video_encoding.watcher.registry import load_registry
from video_encoding.watcher.remediate import remediate_dead_job
from video_encoding.watcher.results import result_json_mtimes
from video_encoding.watcher.squeue import load_squeue_snapshot
from video_encoding.watcher.state import WatcherState, advance_backoff


@dataclass
class TickResult:
    num_flags: int
    actions: list[str]
    action_taken: bool


def run_full_check(
    *,
    repo_root: Path,
    state_dir: Path,
    results_root: Path,
    now: float,
    state: WatcherState,
    mesh_run_ids: frozenset[str] = frozenset(),
) -> TickResult:
    registry = load_registry(state_dir / "jobs_registry.json")
    squeue = load_squeue_snapshot(state_dir / "squeue.json")
    mtimes = result_json_mtimes(registry, results_root)
    registry_by_id = {r.slurm_job_id: r for r in registry}

    # Mesh worker jobs are registered here too (mesh_tick._submit_worker_job
    # shares this registry) but must never reach remediate_dead_job below --
    # it hardcodes train.sbatch (the single-stage script), which crashes
    # immediately against a mesh run_id/experiment. mesh_tick.py's own
    # missed_rounds/_replenish_pool machinery already owns recovery for
    # these nodes; skip them here entirely rather than double-handle.
    def _is_mesh_job(subject: str) -> bool:
        record = registry_by_id.get(subject)
        return record is not None and record.run_tag in mesh_run_ids

    dead_flags = [
        f
        for f in detect_dead(registry, squeue, mtimes, state.jobs, now=now)
        if not _is_mesh_job(f.subject)
    ]
    stalled_flags = [
        f
        for f in detect_stalled(squeue, mtimes, registry, now=now)
        if not _is_mesh_job(f.subject)
    ]
    running_nodes = sorted(
        {
            info["nodelist"]
            for info in squeue.values()
            if info["state"] == "RUNNING" and info["nodelist"]
        }
    )
    disk_usage = collect_disk_usage_pct(running_nodes)
    quota_flags = detect_quota(disk_usage)

    actions: list[str] = []
    transcript_dir = state_dir / "escalations"
    now_str = datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y%m%d-%H%M%S")

    def _try_escalate(flag) -> None:
        if not should_escalate(state, flag.subject, now=now):
            return
        record = registry_by_id.get(flag.subject)
        transcript_path = run_escalation(
            flag,
            repo_root=repo_root,
            transcript_dir=transcript_dir,
            registry_entry=record,
            now_str=now_str,
        )
        record_escalation(state, flag.subject, now=now)
        actions.append(f"escalate {flag.subject} ({flag.kind}) -> {transcript_path}")

    for flag in dead_flags:
        job_state = state.job(flag.subject)
        record = registry_by_id.get(flag.subject)
        if job_state.auto_resubmit_count == 0 and record is not None:
            new_job_id = remediate_dead_job(record, repo_root=repo_root)
            if new_job_id:
                new_state = state.job(new_job_id)
                new_state.auto_resubmit_count = job_state.auto_resubmit_count + 1
                new_state.status = "healthy"
                job_state.status = "resolved"
                actions.append(f"resubmit {flag.subject} -> {new_job_id}")
                continue
            # resubmit attempt itself failed -- fall through to escalate
        _try_escalate(flag)

    for flag in [*stalled_flags, *quota_flags]:
        _try_escalate(flag)

    action_taken = len(actions) > 0
    advance_backoff(state, now=now, action_taken=action_taken)

    return TickResult(
        num_flags=len(dead_flags) + len(stalled_flags) + len(quota_flags),
        actions=actions,
        action_taken=action_taken,
    )
