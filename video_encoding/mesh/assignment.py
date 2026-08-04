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
a new trainer; routing them to a survivor that already holds a replica (the
second copy R=2 placement bought in Task 3) costs zero network. Only when
no survivor holds a replica does a shard fall back to the least-loaded
survivor, which will fetch it over HTTP.
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
) -> dict[str, list[str]]:
    """Redistribute a dead node's shards among the survivors.

    Each orphaned shard prefers a surviving replica holder (per
    `placement`) so the reassignment costs zero network -- this is what
    storing R=2 copies (Task 3) bought. Among multiple qualifying replica
    holders, or when none holds a replica, the least-loaded survivor wins
    (ties broken by hostname for determinism); load is tracked as shards
    are handed out so a single replica holder doesn't get every one of the
    dead node's shards dumped on it.
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

    for shard in assignments[failed_node]:
        replica_holders = [h for h in result if shard in placement.get(h, [])]
        candidates = replica_holders if replica_holders else list(result)
        target = min(candidates, key=lambda h: (load[h], h))
        result[target].append(shard)
        load[target] += 1

    return result


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
