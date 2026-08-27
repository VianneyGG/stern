"""Locality-first shard assignment and replica-aware failure reassignment.

`compute_shard_counts` (Task 2) says *how many* shards each node trains this
round; `plan_placement` (Task 3) says *where* shards physically live. Those
two are computed independently -- a compute-weighted split and a
capacity-weighted storage layout are uncorrelated in general -- so naively
handing out shards without regard to placement would send nearly every
shard over the network for no reason. `assign_locality_first` closes that
gap: each node trains shards it already stores first, and only reaches into
the network-fetch path (Task 7) to fill out the remainder.

`reassign_failed` handles a node dying mid-round. Its in-flight shards need
a new trainer; routing them to a survivor that already holds a replica costs
zero network. A shard no survivor holds is dropped, not parked on an
arbitrary survivor -- there is no HTTP fetch path left to rescue it.

`assign_static` supersedes `assign_locality_first` for launching a round;
the latter survives only for its own tests.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path


def assign_locality_first(
    shard_counts: dict[str, int],
    placement: dict[str, list[str]],
    all_shards: list[str],
) -> dict[str, list[str]]:
    """Decide which specific shards each node trains this round.

    Two-phase, deterministic (nodes and the remainder pool are both walked
    in a fixed order so the same inputs always produce the same output):

    1. Each node takes shards from its own `placement` holdings first, up
       to its `shard_counts` budget, skipping any shard another node has
       already claimed (this is how an overlapping holding -- the same
       shard held by two nodes that both want it -- resolves: first node
       in the fixed order wins, no shard is ever double-assigned).
    2. Any node still short of its budget after phase 1 fills the rest from
       the pool of shards nobody claimed in phase 1, walked in
       `all_shards` order.

    Requires `sum(shard_counts.values()) == len(all_shards)` -- the two
    invariants this function promises ("every shard assigned exactly
    once" and "every count honoured exactly") are only simultaneously
    satisfiable when the totals match by construction.
    """
    if len(set(all_shards)) != len(all_shards):
        raise ValueError("assign_locality_first: all_shards contains duplicates")

    total_shards = len(all_shards)
    total_counts = sum(shard_counts.values())
    if total_counts != total_shards:
        raise ValueError(
            f"assign_locality_first: sum(shard_counts.values())={total_counts} "
            f"!= len(all_shards)={total_shards}; every shard must be assigned "
            "exactly once and every count honoured exactly"
        )

    all_shards_set = set(all_shards)
    nodes = sorted(shard_counts)
    assigned: dict[str, list[str]] = {h: [] for h in nodes}
    taken: set[str] = set()

    # Phase 1: locality -- each node claims its own holdings first.
    for h in nodes:
        need = shard_counts[h]
        for shard in placement.get(h, []):
            if len(assigned[h]) >= need:
                break
            if shard in taken or shard not in all_shards_set:
                continue
            assigned[h].append(shard)
            taken.add(shard)

    # Phase 2: fill remaining budget from the shards nobody claimed.
    remainder = deque(shard for shard in all_shards if shard not in taken)
    for h in nodes:
        need = shard_counts[h]
        while len(assigned[h]) < need:
            # total_counts == total_shards guarantees the pool never runs
            # out before every node's budget is filled.
            shard = remainder.popleft()
            assigned[h].append(shard)
            taken.add(shard)

    return assigned


def reassign_failed(
    failed_node: str,
    assignments: dict[str, list[str]],
    placement: dict[str, list[str]],
) -> tuple[dict[str, list[str]], list[str]]:
    """Redistribute a dead node's shards among survivors that hold them.

    Each orphaned shard goes to a surviving *replica holder* (per
    `placement`), so the reassignment costs zero network -- this is what
    storing multiple copies bought. Among several qualifying holders the
    least-loaded wins (ties broken by hostname for determinism); load is
    tracked as shards are handed out so one holder doesn't get every shard
    dumped on it.

    A shard **no survivor holds is dropped** and returned in the second
    element, not parked on the least-loaded survivor. That fallback is the
    2026-08-20 fault in its second location: the named node cannot open a
    shard it does not have, and with the P2P fetch removed
    (`mesh_shard_dataset.resolve_shard`) it now raises rather than silently
    skipping. Losing coverage of the shard is the honest outcome; the
    caller logs the drop so a degraded round is visible.

    `placement` must be real, verified holdings -- `verified_holdings.json`
    as written by `validate_seeding.py`, never a placement *plan*.
    """
    if failed_node not in assignments:
        raise ValueError(
            f"reassign_failed: {failed_node!r} is not a key in assignments"
        )

    result = {h: list(shards) for h, shards in assignments.items() if h != failed_node}
    if not result:
        raise ValueError(
            "reassign_failed: no surviving nodes left to reassign shards to"
        )

    load = {h: len(shards) for h, shards in result.items()}
    dropped: list[str] = []

    for shard in assignments[failed_node]:
        holders = [h for h in result if shard in placement.get(h, [])]
        if not holders:
            dropped.append(shard)
            continue
        target = min(holders, key=lambda h: (load[h], h))
        result[target].append(shard)
        load[target] += 1

    return result, dropped


def read_my_shards(assignments_path: Path, hostname: str) -> list[str]:
    """Read this node's shard list out of a round's assignments file.

    `assignments_path` is `~/stern_mesh/metadata/shard_assignments_[RUN_ID].json`,
    written by whatever assigns the round (`assign_locality_first` /
    `reassign_failed` above) with schema `{hostname: [shard, ...]}`. Called by
    `run_mesh_worker.sh` before the trainer starts.

    Raises `KeyError` naming the host, not a generic one, if `hostname` has no
    entry, OR if `hostname` maps to an empty list: training on an
    empty/default shard list would silently produce a meaningless checkpoint
    that still gets merged into the master, so an unassigned OR
    zero-shard host must fail loud instead of falling back to `[]`.
    """
    payload = json.loads(Path(assignments_path).read_text())
    if hostname not in payload:
        raise KeyError(
            f"read_my_shards: host {hostname!r} has no entry in "
            f"{assignments_path} (assigned hosts: {sorted(payload)})"
        )
    shards = payload[hostname]
    if not shards:
        raise KeyError(
            f"read_my_shards: host {hostname!r} has an empty shard list in "
            f"{assignments_path} -- training on zero shards would silently "
            "produce a meaningless checkpoint"
        )
    return shards


def assign_static(
    holdings: dict[str, list[str]], expected: set[str]
) -> dict[str, list[str]]:
    """Partition `expected` across the nodes that physically hold each shard.

    This replaces the locality-first/remainder-fill scheme, whose phase 2
    handed a node shards nobody local held and left the worker to fetch them
    over HTTP. When the shard map went stale on 2026-08-20 that path became
    the only path, nothing was serving, and every affected worker trained on
    zero samples while reporting it as a warning.

    Here a node can only ever be assigned a shard it already has, so
    attribution cannot disagree with reality. The result is disjoint: each
    shard goes to exactly one of its holders, chosen least-loaded-first so
    that replicated pools spread evenly rather than piling onto whichever
    node sorts first. Ties break on node name, so the output is
    deterministic.

    Note this does NOT require the pool to be deduplicated to 1x. Holdings
    may overlap freely; disjointness is a property of the partition, not of
    the disk.

    Raises ValueError if any expected shard has no holder -- an unsatisfiable
    assignment must fail here, loudly, rather than at 3am inside a worker.
    """
    assignment: dict[str, list[str]] = {node: [] for node in sorted(holdings)}
    holders = {
        shard: sorted(node for node, held in holdings.items() if shard in set(held))
        for shard in expected
    }

    orphans = sorted(shard for shard, nodes in holders.items() if not nodes)
    if orphans:
        raise ValueError(
            f"assign_static: {len(orphans)} expected shard(s) held by no node: "
            f"{', '.join(orphans[:10])}" + (" ..." if len(orphans) > 10 else "")
        )

    # Seed pass: every node that holds anything gets one shard before any
    # node gets a second. Without this, a node whose small holding is fully
    # replicated on earlier-sorting nodes loses every load tie while they
    # are all still at load 0 and ends up with nothing -- which
    # `mesh_round` reports as train_loss=nan / 0.0 samp/s and then skips
    # the push for, silently dropping merge quorum. Scarcest node first, and
    # each takes its most-constrained shard, so seeding a starved node never
    # strands another one.
    unassigned = set(expected)
    scarcity = {node: len(set(held) & expected) for node, held in holdings.items()}
    for node in sorted(holdings, key=lambda n: (scarcity[n], n)):
        candidates = [s for s in unassigned if node in holders[s]]
        if not candidates:
            continue
        shard = min(candidates, key=lambda s: (len(holders[s]), s))
        assignment[node].append(shard)
        unassigned.discard(shard)

    for shard in sorted(unassigned):
        owner = min(holders[shard], key=lambda n: (len(assignment[n]), n))
        assignment[owner].append(shard)

    return assignment
