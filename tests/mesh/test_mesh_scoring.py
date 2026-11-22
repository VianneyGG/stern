from __future__ import annotations

import pytest

from stern.mesh.scoring import (
    NodeInfo,
    ProportionalityError,
    compute_effective_weights,
    compute_shard_counts,
    effective_weight,
)


def test_effective_weight_uses_score_table_by_default():
    fish = NodeInfo(hostname="fish", group="fish", subnet="/254", free_gb=100.0)
    orchid = NodeInfo(hostname="orchid", group="orchids", subnet="/254", free_gb=100.0)

    assert effective_weight(fish) == 100.0
    assert effective_weight(orchid) == 15.0


def test_effective_weight_scales_with_gpu_count():
    fish2 = NodeInfo(
        hostname="fish2", group="fish", subnet="/254", free_gb=100.0, gpus=2
    )

    assert effective_weight(fish2) == 200.0


def test_measured_throughput_overrides_static_table():
    fish = NodeInfo(
        hostname="fish",
        group="fish",
        subnet="/254",
        free_gb=100.0,
        measured_samples_per_s=40.0,
    )

    assert effective_weight(fish) == 40.0


def test_shard_counts_sum_to_exactly_n_shards():
    nodes = [
        NodeInfo(hostname="fish", group="fish", subnet="/254", free_gb=100.0),
        NodeInfo(hostname="orchid", group="orchids", subnet="/254", free_gb=100.0),
    ]
    weights = compute_effective_weights(nodes)

    counts = compute_shard_counts(weights, n_shards=260)

    assert sum(counts.values()) == 260


def test_shard_counts_are_proportional_to_score():
    nodes = [
        NodeInfo(hostname="fish", group="fish", subnet="/254", free_gb=100.0),
        NodeInfo(hostname="orchid", group="orchids", subnet="/254", free_gb=100.0),
    ]
    weights = compute_effective_weights(nodes)

    counts = compute_shard_counts(weights, n_shards=260)

    assert counts["fish"] > counts["orchid"] * 5


def test_rejects_node_count_that_collapses_proportionality():
    nodes = [
        NodeInfo(hostname=f"fish{i}", group="fish", subnet="/254", free_gb=100.0)
        for i in range(40)
    ] + [NodeInfo(hostname="orchid", group="orchids", subnet="/254", free_gb=100.0)]
    weights = compute_effective_weights(nodes)

    with pytest.raises(ProportionalityError, match="min_shards_per_node"):
        compute_shard_counts(weights, n_shards=52)


def test_rounding_tie_never_produces_invalid_count():
    """Regression: several nodes tied at a `.5` rounding boundary must never
    settle drift onto a single node hard enough to drive it to 0 or
    negative. 4 equal-weight nodes at n_shards=6 hits exactly this: each
    node's raw share is 1.5, Python's banker's rounding rounds all four up
    to 2, overshooting the sum by 2. Dumping that -2 drift onto one node
    (the pre-fix behavior) drove it to 0. Since 4 nodes * min_shards_per_node
    2 = 8 > 6 shards, no valid split actually exists here, so the only
    correct outcomes are: a fully valid split, or ProportionalityError —
    never a non-positive count slipping through.
    """
    weights = {f"f{i}": 100.0 for i in range(4)}

    try:
        counts = compute_shard_counts(weights, n_shards=6)
    except ProportionalityError as exc:
        assert "min_shards_per_node" in str(exc)
    else:
        assert sum(counts.values()) == 6
        assert all(v >= 2 for v in counts.values())


def test_rounding_tie_settles_across_multiple_nodes_when_feasible():
    """When a valid split *does* exist despite a rounding-tie overshoot, the
    settlement must actually find it by spreading drift across several
    nodes, not just raise. 8 equal-weight nodes at n_shards=28: each raw
    share is 3.5, rounds up to 4, overshooting by 4 -- but every node has
    headroom of 2 above the floor (min_shards_per_node=2), so a valid split
    exists (minimum required is 8*2=16 <= 28). Dumping the whole -4 drift on
    one node (pre-fix behavior) would have driven it to 0.
    """
    weights = {f"n{i}": 1.0 for i in range(8)}

    counts = compute_shard_counts(weights, n_shards=28)

    assert sum(counts.values()) == 28
    assert all(v >= 2 for v in counts.values())
    assert all(v > 0 for v in counts.values())


def test_empty_weights_raises_clear_value_error():
    with pytest.raises(ValueError, match="no nodes provided"):
        compute_shard_counts({}, n_shards=10)


def test_realistic_mixed_pool_gives_meaningful_split():
    nodes = (
        [
            NodeInfo(hostname=f"fish{i}", group="fish", subnet="/254", free_gb=100.0)
            for i in range(4)
        ]
        + [
            NodeInfo(
                hostname=f"countries{i}",
                group="countries",
                subnet="/254",
                free_gb=100.0,
            )
            for i in range(2)
        ]
        + [
            NodeInfo(
                hostname=f"orchids{i}", group="orchids", subnet="/254", free_gb=100.0
            )
            for i in range(2)
        ]
    )
    weights = compute_effective_weights(nodes)

    counts = compute_shard_counts(weights, n_shards=260)

    assert sum(counts.values()) == 260
    fish_total = sum(v for k, v in counts.items() if k.startswith("fish"))
    countries_total = sum(v for k, v in counts.items() if k.startswith("countries"))
    orchids_total = sum(v for k, v in counts.items() if k.startswith("orchids"))
    assert fish_total > countries_total > orchids_total
    assert all(v >= 2 for k, v in counts.items() if k.startswith("orchids"))
