"""Effective compute weights and proportional shard counts for the mesh.

Each node in the decentralized P2P training mesh gets an "effective weight"
`E_i`: a hand-assigned cold-start prior (`SCORE_TABLE`), scaled by GPU count,
overridden entirely by measured throughput once available. The fish-cluster
nodes are shared with other students' jobs (why `train.sbatch` uses
`--exclusive`), so a contended 3090 can be genuinely slower than an idle 2000
Ada — the static table is only ever a starting point (arXiv 2607.02544 on
fixed assumptions degrading on shared infra).

`compute_shard_counts` turns a pool of these weights into exact shard counts
that sum to `n_shards`. Naively flooring every node's share at 1 shard
silently flattens the split to uniform once the pool gets large enough that
the weakest node's true share rounds below that floor — this is the bug that
once made a 3090 and a 2000 Ada equal. `ProportionalityError` catches that
case before it happens instead of nodes silently getting a mis-sized fair
share. Rounding drift is settled across multiple nodes rather than dumped on
one, since several nodes tied at a `.5` boundary (the normal case here, not
an edge case) can otherwise push a single node's count to zero or negative.
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
    """Split `n_shards` across nodes proportional to `weights`, summing exactly.

    `W_i = max(1, round(E_i/sum(E) * n_shards))`. Any drift left over by
    rounding is then settled across *multiple* nodes, ordered by descending
    weight (highest-`E_i` first), one shard at a time — never dumped onto a
    single node. Several nodes tied at a `.5` rounding boundary (e.g. several
    nodes in the same `SCORE_TABLE` group — the normal case on this cluster,
    not an edge case) can all get rounded the same direction by Python's
    banker's rounding, so the aggregate overshoot/undershoot can exceed what
    any single node can safely absorb without going below
    `min_shards_per_node`; distributing it avoids driving one node's count to
    zero or negative.

    Raises `ProportionalityError` if the weakest node's *unfloored* share
    rounds below `min_shards_per_node` (the `max(1, .)` floor would otherwise
    silently inflate it to a token allocation that looks proportional but
    isn't), or if settling the rounding drift would require pushing some
    node's final count below `min_shards_per_node` no matter how the drift is
    distributed (i.e. the split is genuinely infeasible at this `n_shards`,
    not just unluckily rounded).
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
                    # A full pass with no eligible node: the drift can't be
                    # distributed without pushing someone below the floor,
                    # no matter how it's spread. Genuinely infeasible.
                    raise _proportionality_error(
                        min_shards_per_node, n_shards, len(weights)
                    )
            i += 1

    return counts
