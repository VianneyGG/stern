"""Capacity-aware shard placement with replication for the P2P training mesh.

`plan_placement` decides where each shard is stored before training starts.
`seed_nodes.py` copies shards to match the plan. `assignment.py` later decides
which node trains which shard, using the compute split from
`scoring.compute_shard_counts`.

Each node's storage share is `min(compute_share, capacity_share)`: capacity
bounds compute, never the reverse. A node that cannot hold its
compute-proportional share gets less work instead of overflowing its disk.
Free disk varies widely across nodes (90 GB to 905 GB in one measured pool),
so a purely compute-proportional split would overflow some of them.

Feasibility is `sum(capacity_share) >= 1`: the total headroom must hold
`replication` copies of the dataset. Gating on `sum(min(...)) < 1` would reject
healthy heterogeneous pools, so it is not the test. Shares are then
water-filled to sum to exactly 1 without exceeding any node's capacity share.

Shards go to the most under-target distinct hosts. Each node's slot count is
floored at its physical capacity. Flooring can leave a near-tight pool one slot
short, and then `plan_placement` raises `InsufficientCapacityError` rather than
overflow any disk.
"""

from __future__ import annotations

from stern.mesh.scoring import NodeInfo, compute_effective_weights

_EPS = 1e-9


class InsufficientCapacityError(ValueError):
    """Raised when the mesh cannot hold `replication` copies of the dataset,
    or when there are fewer nodes than the replication factor requires."""


def plan_placement(
    nodes: list[NodeInfo],
    shards: list[str],
    *,
    dataset_gb: float,
    replication: int = 2,
    min_free_gb: float = 50.0,
) -> dict[str, list[str]]:
    """Return `hostname -> [shard, ...]` with `replication` distinct holders per shard."""
    if replication < 1:
        raise ValueError(f"replication must be >= 1, got {replication}")
    if dataset_gb <= 0:
        raise ValueError(f"dataset_gb must be > 0, got {dataset_gb}")
    if len(nodes) < replication:
        raise InsufficientCapacityError(
            f"need at least replication={replication} distinct node(s) to "
            f"place replicas, got {len(nodes)}"
        )

    placement: dict[str, list[str]] = {node.hostname: [] for node in nodes}
    if not shards:
        return placement

    weights = compute_effective_weights(nodes)
    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError(
            "plan_placement: total effective weight is non-positive; "
            "cannot compute a proportional split"
        )

    compute_share = {h: w / total_weight for h, w in weights.items()}
    capacity_share = {
        node.hostname: max(0.0, node.free_gb - min_free_gb) / (replication * dataset_gb)
        for node in nodes
    }

    if sum(capacity_share.values()) < 1.0 - _EPS:
        raise InsufficientCapacityError(
            f"combined free capacity across {len(nodes)} node(s), after "
            f"{min_free_gb} GB headroom each, cannot hold replication="
            f"{replication} copies of a {dataset_gb} GB dataset"
        )

    share = _water_fill(compute_share, capacity_share)

    n_shards = len(shards)
    total_slots = n_shards * replication
    per_shard_gb = dataset_gb / n_shards

    max_slots = {
        node.hostname: int((max(0.0, node.free_gb - min_free_gb) / per_shard_gb) + _EPS)
        for node in nodes
    }
    target_slots = {h: share[h] * total_slots for h in share}

    assigned_count = {node.hostname: 0 for node in nodes}
    for shard in shards:
        chosen: list[str] = []
        for _ in range(replication):
            candidates = [
                h
                for h in assigned_count
                if h not in chosen and assigned_count[h] < max_slots[h]
            ]
            if not candidates:
                # Aggregate capacity passed the continuous check, but floor
                # rounding of `max_slots` can leave `sum(max_slots)` short of
                # `total_slots`. No host can take this slot without exceeding
                # its floored capacity, so raise rather than overflow one.
                raise InsufficientCapacityError(
                    f"plan_placement: per-node floor rounding of physical "
                    f"slot capacity left no eligible host to hold "
                    f"{shard!r} without exceeding its floored capacity "
                    f"(replication={replication}); sum(capacity_share) "
                    f"passed the continuous feasibility check but "
                    f"sum(max_slots)={sum(max_slots.values())} < "
                    f"total_slots={total_slots} after flooring. Reduce "
                    f"dataset_gb/replication or add capacity."
                )
            best = min(
                candidates,
                key=lambda h: (assigned_count[h] - target_slots[h], h),
            )
            chosen.append(best)
            assigned_count[best] += 1
            placement[best].append(shard)

    return placement


def _water_fill(
    compute_share: dict[str, float], capacity_share: dict[str, float]
) -> dict[str, float]:
    """Spread the shortfall from capacity-bound nodes over the unbound nodes.

    The spread is proportional to compute share, repeated until shares sum to
    1. No node ever exceeds its `capacity_share`. A single scalar rescale
    would push capacity-bound nodes over their cap.
    """
    share = {h: min(compute_share[h], capacity_share[h]) for h in compute_share}
    unbound = {h for h in share if capacity_share[h] > compute_share[h] + _EPS}

    while unbound:
        deficit = 1.0 - sum(share.values())
        if deficit <= _EPS:
            break
        unbound_compute_total = sum(compute_share[h] for h in unbound)
        newly_bound = set()
        for h in unbound:
            addition = deficit * (compute_share[h] / unbound_compute_total)
            candidate = share[h] + addition
            if candidate >= capacity_share[h] - _EPS:
                share[h] = capacity_share[h]
                newly_bound.add(h)
        if newly_bound:
            unbound -= newly_bound
            continue
        for h in unbound:
            addition = deficit * (compute_share[h] / unbound_compute_total)
            share[h] += addition
        break

    return share
