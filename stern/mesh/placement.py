"""Capacity-aware seeding plan with replication for the P2P training mesh.

`plan_placement` decides *where* each shard is physically stored before any
training happens: `mesh/seed.py` (Task 9) rsyncs shards to match this plan,
`mesh/assignment.py` (Task 4) later decides what each node *trains* this
round from `compute_shard_counts` (Task 2) combined with this placement.

Each node's storage share is `min(compute_share, capacity_share)` --
capacity bounds compute, never the other way around. A node that cannot
physically hold its proportional-to-compute share gets *less* work instead
of overflowing its disk. This matters because the fish cluster's free disk
ranges from 90 GB (node024) to 905 GB (node089) and drifts as other
students use `/Data` -- a purely compute-proportional split would silently
try to write more than some node's disk can hold.

**Feasibility check.** `compute_share_i = E_i / sum(E)` sums to exactly 1
over the node pool by construction, so `sum(min(compute_share_i,
capacity_share_i))` is *always* <= 1, and strictly < 1 the moment even one
node is capacity-bound (its capped share is below its compute share). That
makes "sum(min(...)) < 1" true in the overwhelmingly common case of a
heterogeneous-disk cluster -- exactly the case this module exists to
handle -- even when the *total* free capacity across the mesh is enormous
(e.g. one small node alongside several large ones). Gating
`InsufficientCapacityError` on that sum would raise on healthy, plainly
placeable pools. The condition that actually answers "can the mesh hold
`replication` copies of the dataset at all" is `sum(capacity_share_i) >= 1`
(total headroom-adjusted free space, in units of one full replicated
dataset) -- that is what is checked here.

**Renormalization.** Given feasibility, shares are brought up to sum to
exactly 1 by water-filling: capacity-bound nodes keep their capacity-capped
share (never pushed higher -- that would be the overflow this module exists
to prevent), and the shortfall is redistributed, proportional to compute
share, only among nodes not (yet) capacity-bound, iterating since a
redistribution can itself push a previously-fine node onto its own cap.
This always converges to `sum(w_i) == 1` without exceeding any node's
`capacity_share_i`, whenever `sum(capacity_share_i) >= 1`.

Shard-to-host assignment then greedily gives each shard's `replication`
slots to the most-under-target *distinct* hosts, with each node's integer
slot budget hard-capped at its true physical capacity (floored, so
floating-point error can only under- not over-allocate). The continuous
feasibility check above bounds *aggregate* capacity, not each node's
floored capacity individually -- per-node flooring can independently
shave up to ~1 slot off each node, so `sum(max_slots)` can still fall
short of `total_slots` in a near-tight pool even though
`sum(capacity_share_i) >= 1`. If discretization ever runs out of eligible
distinct hosts for a shard because of that flooring loss, it raises
`InsufficientCapacityError` rather than exceeding any node's floored
`max_slots` -- so this module never overflows a disk, full stop, even at
the cost of occasionally rejecting a continuous-feasible but
integer-infeasible input.
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
                # The continuous feasibility check (sum(capacity_share) >= 1)
                # guarantees enough *aggregate* capacity, but per-node floor
                # rounding of `max_slots` can independently shave up to
                # ~1 slot off each node's true physical capacity -- so
                # `sum(max_slots)` can still fall short of `total_slots` in a
                # near-tight pool. There is no host left that can take this
                # slot without exceeding its own floored physical capacity;
                # raise rather than silently overflow one (this is exactly
                # the "never an overflow" invariant this module exists to
                # hold, even when it costs rejecting a technically-tight
                # continuous-feasible input).
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
    """Redistribute the shortfall left by capacity-bound nodes among the
    not-(yet)-bound nodes, proportional to their compute share, until every
    node's share sums to exactly 1 -- without ever exceeding a node's own
    `capacity_share`. See module docstring for why this differs from a
    single-pass scalar renormalization.
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
