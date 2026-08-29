"""Watcher state: per-job tracking + the adaptive backoff timer.

Persisted to runs/state/watcher_state.json. The watcher has no long-lived
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
class RunState:
    """Per-run state for the decentralized mesh training round machine.

    round/status track outer-optimizer progress; missed_rounds counts
    consecutive rounds a node failed to check in (keyed by node name),
    reset to 0 by a successful round, and read by epoch_advance's
    crash-loop exclusion (spec: docs/superpowers/plans mesh training).
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
    # --quorum-relaxed, recorded into the round metrics JSONL (design §5.2).
    any_round_quorum_relaxed: bool = False  # OR'd, never cleared -- a run
    # that relaxed quorum even once must not be reported as a clean DONE
    # (design §5.2).
    # All four set once, in _do_awaiting_checkpoints, every branch that
    # transitions to MERGING -- read once by _do_merging for this round's
    # metrics record, same lifecycle as timeout_forced/quorum_relaxed
    # (design §8).
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
    # top of watch_training.main() -- even on a fast-path tick that returns
    # before run_full_check. Read by mesh_continue_allowed.sh's fifth stop
    # signal (design §1.2/§3b) to detect a dead/wedged watcher in ~10 min
    # instead of a multi-hour futile-requeue cycle.
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
    """Mutate state's interval/next-check-time per spec §4 step 5."""
    if action_taken:
        state.current_interval_s = FAST_INTERVAL_S
    else:
        state.current_interval_s = min(
            state.current_interval_s * BACKOFF_MULTIPLIER, SLOW_INTERVAL_S
        )
    state.next_full_check_at = now + state.current_interval_s
