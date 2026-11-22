"""Effective compute weights and proportional shard counts for the mesh.

Each node in the P2P training mesh gets an effective weight `E_i`: a
cold-start prior from `SCORE_TABLE`, scaled by GPU count, replaced entirely by
measured throughput once available. Nodes are shared with other users' jobs,
so a contended GPU can be slower than the table says. The table is only a
starting point (arXiv 2607.02544 on fixed assumptions degrading on shared
infrastructure).

`compute_shard_counts` turns the weights into integer shard counts that sum to
`n_shards`. Flooring every node at one shard would silently flatten the split
to uniform once the weakest node's share rounds below that floor, so
`ProportionalityError` is raised instead. Rounding drift is spread across
nodes rather than dumped on one, since many nodes tie at a `.5` boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

SCORE_TABLE: dict[str, int] = {
    "fish": 100,
    "birds": 85,
    "bones": 70,
    "cars": 70,
    "rivers": 70,
    "countries": 30,
    "orchids": 15,
}


class ProportionalityError(ValueError):
    """Raised when a node pool cannot be split proportionally at this n_shards."""


@dataclass(frozen=True)
class NodeInfo:
    hostname: str
    group: str  # key into SCORE_TABLE
    subnet: str
    free_gb: float
    gpus: int = 1
    measured_samples_per_s: float | None = None


def effective_weight(node: NodeInfo) -> float:
    """`S_i x G_i`, or `measured_samples_per_s` when available (overrides the table)."""
    if node.measured_samples_per_s is not None:
        return float(node.measured_samples_per_s)
    return float(SCORE_TABLE[node.group]) * node.gpus


def compute_effective_weights(nodes: list[NodeInfo]) -> dict[str, float]:
    return {node.hostname: effective_weight(node) for node in nodes}


def _proportionality_error(
    min_shards_per_node: int, n_shards: int, pool_size: int
) -> ProportionalityError:
    return ProportionalityError(
        f"cannot split n_shards={n_shards} across {pool_size} nodes without "
        f"some node falling below min_shards_per_node={min_shards_per_node}; "
        "reduce the number of nodes in the pool or raise n_shards."
    )


def compute_shard_counts(
    weights: dict[str, float], n_shards: int, *, min_shards_per_node: int = 2
) -> dict[str, int]:
    """Split `n_shards` across nodes in proportion to `weights`, summing exactly.

    Each node gets `max(1, round(E_i / sum(E) * n_shards))`. Rounding drift is
    then settled one shard at a time, in descending-weight order, spread across
    nodes. Raises `ProportionalityError` if a node's unfloored share rounds below
    `min_shards_per_node`, or if the drift cannot be settled without pushing a
    node under that minimum.
    """
    if not weights:
        raise ValueError("compute_shard_counts: no nodes provided in `weights`")

    total = sum(weights.values())
    if total <= 0:
        raise ValueError(
            "compute_shard_counts: total effective weight is non-positive "
            f"(sum(weights.values())={total}); cannot compute a proportional split"
        )

    raw_rounded = {
        hostname: round(weight / total * n_shards)
        for hostname, weight in weights.items()
    }

    if min(raw_rounded.values()) < min_shards_per_node:
        raise _proportionality_error(min_shards_per_node, n_shards, len(weights))

    counts = {hostname: max(1, value) for hostname, value in raw_rounded.items()}

    # Settle in descending-weight order (ties broken by hostname) so the
    # nodes with the most slack absorb the drift first.
    order = sorted(weights, key=lambda hostname: (-weights[hostname], hostname))
    floor = max(1, min_shards_per_node)
    drift = n_shards - sum(counts.values())

    if drift > 0:
        i = 0
        while drift > 0:
            counts[order[i % len(order)]] += 1
            drift -= 1
            i += 1
    elif drift < 0:
        i = 0
        stalled = 0
        while drift < 0:
            hostname = order[i % len(order)]
            if counts[hostname] > floor:
                counts[hostname] -= 1
                drift += 1
                stalled = 0
            else:
                stalled += 1
                if stalled >= len(order):
                    # A full pass with no eligible node: the drift cannot be
                    # spread without going below the floor. Infeasible.
                    raise _proportionality_error(
                        min_shards_per_node, n_shards, len(weights)
                    )
            i += 1

    return counts
