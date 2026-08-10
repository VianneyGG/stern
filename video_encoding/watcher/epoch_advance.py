"""Mesh round state machine helpers: pending-checkpoint discovery, node
exclusion, and crash-loop detection via missed-round counting.

Pure/I-O-light like detectors.py -- the only filesystem access is the
directory listing in pending_nodes (mirroring detectors' read of squeue /
registry snapshots as plain data, no subprocess calls). The subprocess
invocation of run_outer_step.py belongs in a future caller, not here.
"""

from __future__ import annotations

from pathlib import Path

from video_encoding.mesh.pending_files import parse_pending_filename
from video_encoding.watcher.state import RunState, WatcherState

MISSED_ROUND_THRESHOLD = 2


def pending_nodes(
    pending_dir: Path, run_id: str, *, round_num: int | None = None
) -> set[str]:
    """Nodes that have dropped a checkpoint for run_id this round.

    Parses `[RUN_ID]_[ROUND]_[NODE].pt` filenames (mesh/pending_files.py).
    Files are overwritten each round (peak disk bounded by node count, not
    round count), so a directory listing usually reflects only the current
    round's check-ins -- except right after a timeout-forced merge clears
    pending/ (mesh_tick._do_merging), where a straggler still finishing the
    PRIOR round can write into it before this round's real checkpoints
    land. `round_num` (pass the run's current `round`) excludes any such
    leftover from the count -- mtime alone can't: a straggler necessarily
    writes in real wall-clock time AFTER the new round already started, so
    no mtime-relative-to-round-start filter can tell it apart from a
    genuine current-round arrival. Only the round the file actually belongs
    to (carried in its own name) can.
    """
    nodes: set[str] = set()
    for entry in pending_dir.glob(f"{run_id}_*.pt"):
        parsed = parse_pending_filename(entry.stem, run_id)
        if parsed is None:
            continue
        file_round, node = parsed
        if round_num is not None and file_round != round_num:
            continue
        nodes.add(node)
    return nodes


def exclude_node(state: WatcherState, node: str, reason: str) -> None:
    """Permanently exclude a node from future round assignment.

    Never auto-reversed: quota exhaustion and repeated crashes must not
    self-heal silently. A human clears excluded_nodes by hand.
    """
    state.excluded_nodes[node] = reason


def record_missed_round(
    state: WatcherState,
    run_id: str,
    node: str,
    *,
    threshold: int = MISSED_ROUND_THRESHOLD,
) -> bool:
    """Record that `node` missed a round of `run_id`; exclude on crash-loop.

    One miss is tolerated as transient (mirrors tick.py's resubmit-once-then
    -escalate tolerance). `threshold` consecutive misses excludes the node.
    Returns True iff this call caused an exclusion.
    """
    run = state.run(run_id)
    count = run.missed_rounds.get(node, 0) + 1
    run.missed_rounds[node] = count
    if count >= threshold:
        exclude_node(
            state, node, f"crash-loop: missed {count} consecutive rounds of {run_id}"
        )
        return True
    return False


def eligible_nodes(state: WatcherState, candidates: list[str]) -> list[str]:
    """Candidates minus any node currently in excluded_nodes."""
    return [node for node in candidates if node not in state.excluded_nodes]


def pool_deficit(run: RunState, roster: list[str], state: WatcherState) -> int:
    """max(0, run.target_pool_size - len(eligible_nodes(state, roster))).
    Never negative -- a roster that has already grown past target_pool_size
    (e.g. after a previous replenishment already ran) reports 0, not a
    negative want."""
    available = len(eligible_nodes(state, roster))
    return max(0, run.target_pool_size - available)


def candidate_replacements(
    tier_roster: list[str], in_use: set[str], excluded: set[str]
) -> list[str]:
    """tier_roster's hosts, in tier_roster's own order, minus in_use minus
    excluded. Deterministic: same inputs always produce the same order."""
    return [node for node in tier_roster if node not in in_use and node not in excluded]


def select_idle_nodes(candidates: list[str], count: int, idle: set[str]) -> list[str]:
    """First `count` of `candidates` that are also present in `idle`, in
    candidate order. Returns fewer than `count` (never raises) when `idle`
    has too few matches -- callers decide how to handle a partial result."""
    result = []
    for node in candidates:
        if node in idle:
            result.append(node)
            if len(result) == count:
                break
    return result
