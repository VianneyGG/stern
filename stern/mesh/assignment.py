"""Shard assignment for a training round, and reassignment after a node dies.

`scoring.compute_shard_counts` says how many shards each node trains this
round, and `placement.plan_placement` says where shards are stored. The two
are computed independently, so handing out shards without regard to placement
would send most shards over the network for no reason.

`assign_static` gives each shard to one of the nodes that already hold it, so
a node is only ever assigned a shard on its own disk.

`reassign_failed` handles a node dying mid-round. Its in-flight shards go to a
surviving replica holder, which costs no network. A shard no survivor holds is
dropped, since nothing can fetch it.
"""

from __future__ import annotations

import json
from pathlib import Path


def reassign_failed(
    failed_node: str,
    assignments: dict[str, list[str]],
    placement: dict[str, list[str]],
) -> tuple[dict[str, list[str]], list[str]]:
    """Redistribute a dead node's shards among survivors that hold them.

    Each orphaned shard goes to a surviving replica holder (per `placement`),
    so the reassignment costs no network. Among several holders the least
    loaded wins, ties broken by hostname. Load is updated as shards are handed
    out, so one holder does not take them all.

    A shard **no survivor holds is dropped** and returned in the second
    element. It is not parked on a survivor that lacks it, because that node
    cannot read the shard (`mesh_shard_dataset.resolve_shard` raises on a
    missing shard). The caller logs the drop so a degraded round is visible.

    `placement` must be verified holdings (`verified_holdings.json` from
    `validate_seeding.py`), never a placement plan.
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
    written by the round's assigner (`assign_static` or `reassign_failed`) with
    schema `{hostname: [shard, ...]}`. Called by `cluster/run_mesh_worker.sh`
    before the trainer starts.

    Raises `KeyError` naming the host if it has no entry or an empty list.
    Training on zero shards would produce a meaningless checkpoint that still
    gets merged into the master, so fail loudly instead of returning `[]`.
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

    A node is only ever assigned a shard it already has, so attribution cannot
    disagree with the disk. An older scheme let a node fetch leftover shards
    over HTTP. Once the shard map went stale, that fetch path was the only one,
    and workers trained on zero samples while logging a warning.

    Each shard goes to exactly one holder, least loaded first, so replicated
    pools spread evenly. Ties break on node name, so the output is
    deterministic. Holdings may overlap; disjointness is a property of the
    partition, not of the disk.

    Raises ValueError if any expected shard has no holder, so an unsatisfiable
    assignment fails here rather than inside a worker.
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

    # Seed pass: every node that holds anything gets one shard before any node
    # gets a second. Otherwise a node whose holdings are all replicated on
    # earlier-sorted nodes loses every load tie and trains nothing, and its
    # round reports train_loss=nan and skips the push, silently dropping quorum.
    # Scarcest node first, each taking its most constrained shard.
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
