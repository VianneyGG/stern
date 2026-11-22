"""Watcher state: per-job tracking + the adaptive backoff timer.

Persisted to runs/state/watcher_state.json. The watcher has no long-lived
process (it is cron-driven), so this file is its memory between invocations:
loaded fresh each tick, mutated in memory by tick.run_full_check, and written
back atomically by the caller.
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
class RunState:
    """Per-run state for the decentralized mesh training round machine.

    round/status track outer-optimizer progress; missed_rounds counts
    consecutive rounds a node failed to check in (keyed by node name),
    reset to 0 by a successful round, and read by epoch_advance's
    crash-loop exclusion.
    """

    run_id: str
    round: int = 0
    status: str = (
        "ASSIGNING"  # ASSIGNING|RUNNING|AWAITING_CHECKPOINTS|MERGING|SYNCING|DONE
    )
    missed_rounds: dict[str, int] = field(default_factory=dict)
    # node -> times epoch_advance.readmit_healthy_nodes has lifted this run's
    # crash-loop exclusion for it. Caps the second chance so a node that looks
    # healthy but keeps dying cannot flap in and out of the pool every round.
    # Defaulted, so a state file written before this field existed still loads.
    readmitted: dict[str, int] = field(default_factory=dict)
    # Epoch seconds of the last mesh_tick._recover_pool probe. That pass ssh's
    # every excluded roster node, so it is throttled rather than run on every
    # one-minute cron tick. Defaulted, so older state files still load.
    last_recovery_at: float = 0.0
    target_pool_size: int = 0  # 0 = not yet observed; set once by
    # mesh_tick.advance_one_run to len(run_config.nodes) the first tick
    # this run is seen, so pool_deficit (epoch_advance.py) has a stable
    # baseline even after replenishment grows run_config.nodes later.
    timeout_forced: bool = False  # set by _do_awaiting_checkpoints each time
    # it decides AWAITING_CHECKPOINTS -> MERGING (both branches, so it never
    # carries a stale value from a prior round); read by _do_merging to pass
    # run_outer_step.py's --timeout-forced, recorded into the round metrics
    # JSONL so a partial-quorum merge is distinguishable from a full one.
    round_started_at: float = 0.0  # set to `now` every time status becomes
    # RUNNING for a fresh round (_do_assigning, _do_syncing); _round_elapsed_s
    # falls back to this when zero checkpoints have arrived yet, so the
    # AWAITING_CHECKPOINTS timeout can still fire if the whole pool dies
    # before checking in even once -- pending-checkpoint mtimes alone can't
    # cover that case.
    quorum_relaxed: bool = False  # set by _do_awaiting_checkpoints each time
    # it decides AWAITING_CHECKPOINTS -> MERGING (every branch, so it never
    # carries a stale value); read by _do_merging to pass run_outer_step.py's
    # --quorum-relaxed and record it in the round metrics JSONL.
    any_round_quorum_relaxed: bool = False  # OR'd, never cleared. A run that
    # relaxed quorum even once must not be reported as a clean DONE.
    # All four set once, in _do_awaiting_checkpoints, every branch that
    # transitions to MERGING. _do_merging reads them once for this round's
    # metrics record, same lifecycle as timeout_forced/quorum_relaxed.
    # Last terminal note _relaunch_chunk_if_finished returned for this run.
    # A non-None return short-circuits the state machine for the tick, so a
    # condition that cannot clear itself would re-report the same note every
    # 60s and block the merge/timeout handlers. Repeating the same note means
    # nothing changed: report it once, then fall through.
    # Defaulted, so a state file written before this field existed loads.
    last_stall_note: str = ""
    round_misses_charged: int = 0
    round_exclusions_suspended: int = 0
    round_effective_quorum_k: int | None = None


@dataclass
class WatcherState:
    next_full_check_at: float = 0.0
    current_interval_s: float = FAST_INTERVAL_S
    jobs: dict[str, JobState] = field(default_factory=dict)
    escalation_day: str | None = None
    escalation_count_today: int = 0
    runs: dict[str, RunState] = field(default_factory=dict)
    excluded_nodes: dict[str, str] = field(default_factory=dict)  # node -> reason
    # run_id -> epoch time a NaN-remediation claude -p was spawned, so the
    # per-run concurrency cap (at most one in flight) survives across the
    # separate cron-spawned watch_training.py process each tick runs in --
    # nothing in-memory carries over otherwise. Cleared implicitly once
    # `now - value` exceeds the caller's concurrency_timeout_s, not by an
    # explicit pop (see escalate.remediate_nan_node).
    nan_escalations_in_flight: dict[str, float] = field(default_factory=dict)
    # Epoch seconds of the last cron tick, stamped unconditionally at the
    # top of watch_training.main(), even on a fast-path tick that returns
    # before run_full_check. mesh_continue_allowed.sh reads it to detect a
    # dead or wedged watcher within ~10 min, instead of a multi-hour
    # futile-requeue cycle.
    last_tick_at: float = 0.0

    def job(self, slurm_job_id: str) -> JobState:
        return self.jobs.setdefault(slurm_job_id, JobState(slurm_job_id=slurm_job_id))

    def run(self, run_id: str) -> RunState:
        return self.runs.setdefault(run_id, RunState(run_id=run_id))


def load_state(path: Path) -> WatcherState:
    if not path.exists():
        # next_full_check_at=0.0 -- any real `now` is >= 0, so the very first
        # tick after deployment always runs a full check immediately.
        return WatcherState()
    payload = json.loads(path.read_text())
    jobs = {k: JobState(**v) for k, v in payload.get("jobs", {}).items()}
    runs = {k: RunState(**v) for k, v in payload.get("runs", {}).items()}
    return WatcherState(
        next_full_check_at=payload.get("next_full_check_at", 0.0),
        current_interval_s=payload.get("current_interval_s", FAST_INTERVAL_S),
        jobs=jobs,
        escalation_day=payload.get("escalation_day"),
        escalation_count_today=payload.get("escalation_count_today", 0),
        runs=runs,
        excluded_nodes=payload.get("excluded_nodes", {}),
        nan_escalations_in_flight=payload.get("nan_escalations_in_flight", {}),
        last_tick_at=payload.get("last_tick_at", 0.0),
    )


def save_state(path: Path, state: WatcherState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "next_full_check_at": state.next_full_check_at,
        "current_interval_s": state.current_interval_s,
        "jobs": {k: asdict(v) for k, v in state.jobs.items()},
        "escalation_day": state.escalation_day,
        "escalation_count_today": state.escalation_count_today,
        "runs": {k: asdict(v) for k, v in state.runs.items()},
        "excluded_nodes": state.excluded_nodes,
        "nan_escalations_in_flight": state.nan_escalations_in_flight,
        "last_tick_at": state.last_tick_at,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


_ROUND_DURATION_WINDOW = 5  # rolling-average window, in completed rounds
_CHUNK_HISTORY_WINDOW = 5  # how many past chunks to report
# A chunk relaunch submits every worker in the pool (16+ nodes) within one
# tick, so their submitted_at timestamps land in a tight burst; a genuinely
# new chunk (the *next* relaunch, once the old workers have died and the
# watcher noticed) is tens of minutes to hours later. This only has to sit
# between those two scales.
_CHUNK_WAVE_GAP_S = 300.0


def _chunk_wave_starts(registry, run_id: str) -> list[float]:
    """Epoch start time of each submission wave (chunk) seen for `run_id`.

    A chunk is one pool-wide relaunch, not one slurm job -- the registry has
    one JobRecord per node, so this clusters those by submission burst
    rather than reporting per-job-id data (a pool's chunk identity is the
    run_id/wave, e.g. "pool-a", not any individual node's slurm job id).
    """
    timestamps = sorted(
        ts
        for ts in (_submitted_at_epoch(r) for r in registry if r.run_tag == run_id)
        if ts is not None
    )
    starts: list[float] = []
    for ts in timestamps:
        if not starts or ts - starts[-1] > _CHUNK_WAVE_GAP_S:
            starts.append(ts)
    return starts


def _chunk_durations(registry, run_id: str) -> list[float]:
    """Past chunks' durations: gap between one wave's start and the next's.

    The final (still-running or most-recently-finished) wave has no next
    start yet, so it contributes no entry -- only *completed* chunks have a
    known duration.
    """
    starts = _chunk_wave_starts(registry, run_id)
    durations = [b - a for a, b in zip(starts, starts[1:])]
    return durations[-_CHUNK_HISTORY_WINDOW:]


def _hours(seconds: float | None) -> float | None:
    return round(seconds / 3600.0, 3) if seconds is not None else None


def _round_pace(
    results_dir: Path, run_id: str, now: float
) -> tuple[float | None, list[float]]:
    """(avg_round_duration_s, recent per-round durations) from `<run_id>_rounds.jsonl`.

    Durations are gaps between consecutive merge timestamps (`ts`), so a
    round skipped by a crashed merge just widens that one gap rather than
    producing a bogus zero. None/[] when fewer than two rounds have merged
    yet -- there is no duration to report.
    """
    path = results_dir / f"{run_id}_rounds.jsonl"
    if not path.exists():
        return None, []
    timestamps: list[float] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            timestamps.append(json.loads(line)["ts"])
    if len(timestamps) < 2:
        return None, []
    durations = [b - a for a, b in zip(timestamps, timestamps[1:])]
    recent = durations[-_ROUND_DURATION_WINDOW:]
    return sum(recent) / len(recent), recent


def write_status_summary(
    path: Path,
    state: WatcherState,
    run_configs: list,
    *,
    state_dir: Path,
) -> None:
    """Human-readable X/N round + status snapshot, one section per pool (run).

    Written only by callers that already know something changed this tick
    (a mesh_tick action fired) -- this is not a second poller, just a cheap
    projection of state.runs taken at the same moment save_state is. Omits
    per-node identity (slurm job id -> node -> health) on purpose: with
    16-32 workers/run and dozens of runs ever tracked, that list was most of
    this file's size, and it's the wrong granularity anyway -- a pool's
    "job" is the run_id/chunk wave, not any one node's slurm job id;
    squeue.json/jobs_registry.json already carry the per-node detail.
    """
    import time as _time

    from stern.watcher.registry import load_registry

    registry = load_registry(state_dir / "jobs_registry.json")
    results_dir = state_dir.parent / "results" / "mesh"
    now = _time.time()

    def _run_entry(run_config) -> dict:
        run = state.run(run_config.run_id)
        avg_round_s, recent_round_s = _round_pace(results_dir, run_config.run_id, now)
        rounds_remaining = max(run_config.rounds - run.round, 0)
        eta_s = avg_round_s * rounds_remaining if avg_round_s is not None else None
        return {
            "status": run.status,
            "round": run.round,
            "rounds": run_config.rounds,
            "quorum_relaxed": run.any_round_quorum_relaxed,
            "missed_rounds": run.missed_rounds,
            "readmitted": run.readmitted,
            "round_misses_charged": run.round_misses_charged,
            "round_exclusions_suspended": run.round_exclusions_suspended,
            "round_effective_quorum_k": run.round_effective_quorum_k,
            "avg_round_duration_h": _hours(avg_round_s),
            "recent_round_durations_h": [_hours(d) for d in recent_round_s],
            "eta_h": _hours(eta_s),
            "eta_ts": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime(now + eta_s))
            if eta_s is not None
            else None,
            "past_chunk_durations_h": [
                _hours(d) for d in _chunk_durations(registry, run_config.run_id)
            ],
        }

    payload = {
        "updated_at": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime(now)),
        "excluded_nodes": state.excluded_nodes,
        "runs": {
            run_config.run_id: _run_entry(run_config) for run_config in run_configs
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def _submitted_at_epoch(record) -> float | None:
    from datetime import datetime

    try:
        return datetime.fromisoformat(record.submitted_at).timestamp()
    except (ValueError, TypeError):
        return None


def clear_run_state(state: WatcherState, run_id: str) -> None:
    """Wipe run_id's crash-loop exclusions when it reaches DONE.

    Only removes excluded_nodes entries whose reason names run_id
    (record_missed_round's exact format: "...consecutive rounds of
    {run_id}") -- exclude_node's own contract is that a node is "never
    auto-reversed... a human clears excluded_nodes by hand", so an
    exclusion from a different run, or a manually-flagged permanent
    hardware reason, must survive this call untouched.

    Deliberately does NOT pop run_id from state.runs: that dict is the only
    record watch_training.py has of every mesh run_id ever seen, and its
    union with the live mesh_run_configs.json is what exempts a finished
    run's worker jobs from remediate_dead_job (see watch_training.py's
    mesh_run_ids comment) -- removing it here would reopen that exact bug
    for any run that goes through DONE.
    """
    suffix = f"of {run_id}"
    state.excluded_nodes = {
        node: reason
        for node, reason in state.excluded_nodes.items()
        if not reason.endswith(suffix)
    }


def advance_backoff(state: WatcherState, *, now: float, action_taken: bool) -> None:
    """Set the next check time: reset the interval on action, else back off."""
    if action_taken:
        state.current_interval_s = FAST_INTERVAL_S
    else:
        state.current_interval_s = min(
            state.current_interval_s * BACKOFF_MULTIPLIER, SLOW_INTERVAL_S
        )
    state.next_full_check_at = now + state.current_interval_s
