from __future__ import annotations

import pytest

from video_encoding.mesh.placement import InsufficientCapacityError, plan_placement
from video_encoding.mesh.scoring import NodeInfo

SHARDS = [f"shard-{i:06d}.tar" for i in range(20)]


def _holders_by_shard(placement: dict[str, list[str]]) -> dict[str, list[str]]:
    holders: dict[str, list[str]] = {shard: [] for shard in SHARDS}
    for hostname, shards in placement.items():
        for shard in shards:
            holders[shard].append(hostname)
    return holders


def test_every_shard_lands_on_exactly_two_distinct_nodes():
    nodes = [
        NodeInfo(hostname=f"fish{i}", group="fish", subnet="/254", free_gb=400.0)
        for i in range(4)
    ]

    placement = plan_placement(nodes, SHARDS, dataset_gb=100.0)

    holders = _holders_by_shard(placement)
    for shard, hosts in holders.items():
        assert len(hosts) == 2, f"{shard} has {len(hosts)} holders, want 2"
        assert len(set(hosts)) == 2, f"{shard} holders {hosts} are not distinct"


def test_small_disk_node_gets_less_work_not_an_overflow():
    """node003 (fish, ample disk) and node024 (birds, tight disk) at
    dataset_gb=150/replication=1 (with only 2 nodes, replication=2 would
    force *both* to hold every shard regardless of weight -- not a useful
    test of proportionality): node024's capacity_share (0.267) is below
    its compute_share (0.4595), so it is genuinely capacity-bound -- a
    naive compute-only split would give node024 ~9 shards (69 GB),
    overflowing its (90-50)=40 GB cap. Capacity must bound it to <= 5
    shards (37.5 GB) instead, leaving node003 with the rest.
    """
    node003 = NodeInfo(hostname="node003", group="fish", subnet="/254", free_gb=587.0)
    node024 = NodeInfo(hostname="node024", group="birds", subnet="/254", free_gb=90.0)

    placement = plan_placement(
        [node003, node024], SHARDS, dataset_gb=150.0, replication=1
    )

    assert len(placement["node024"]) < len(placement["node003"])
    per_shard_gb = 150.0 / len(SHARDS)
    assert len(placement["node024"]) * per_shard_gb <= 90.0 - 50.0


def test_respects_min_free_headroom():
    """fish (weight 100) and orchid (weight 15) both at 120 G: a
    compute-only split of the dataset_gb=100 would give fish ~87 GB,
    breaching its (120-50)=70 GB cap. Placement must hold fish at exactly
    70 GB (14 shards @ 5 GB) instead.
    """
    fish = NodeInfo(hostname="fish", group="fish", subnet="/254", free_gb=120.0)
    orchid = NodeInfo(hostname="orchid", group="orchids", subnet="/254", free_gb=120.0)

    placement = plan_placement(
        [fish, orchid], SHARDS, dataset_gb=100.0, replication=1, min_free_gb=50.0
    )

    per_shard_gb = 100.0 / len(SHARDS)
    for node in ("fish", "orchid"):
        stored_gb = len(placement[node]) * per_shard_gb
        assert stored_gb <= 120.0 - 50.0


def test_insufficient_total_capacity_raises_instead_of_overfilling():
    nodes = [
        NodeInfo(hostname="n0", group="fish", subnet="/254", free_gb=60.0),
        NodeInfo(hostname="n1", group="fish", subnet="/254", free_gb=60.0),
    ]

    with pytest.raises(InsufficientCapacityError, match="capacity"):
        plan_placement(nodes, SHARDS, dataset_gb=200.0, min_free_gb=50.0)


def test_discretization_never_exceeds_floored_physical_capacity():
    """Regression: sum(capacity_share) can pass the continuous feasibility
    check while per-node floor rounding of `max_slots` still leaves
    `sum(max_slots) < total_slots`. n0=152.5 G, n1=147.5 G at
    dataset_gb=100/replication=2/min_free_gb=50 gives capacity_share
    0.5125 + 0.4875 == 1.0 exactly (passes), but max_slots floors to
    20 + 19 == 39 < total_slots=40. With only 2 nodes and replication=2,
    every shard needs both as its 2 distinct holders, so the 40th slot
    has nowhere to go without exceeding n1's floored 19-slot cap (95 GB
    of dataset on an (147.5-50)=97.5 GB budget -- 19 shards @ 5 GB = 95 GB
    fits, a 20th would not). Must raise, never silently assign n1 a 20th
    shard.
    """
    nodes = [
        NodeInfo(hostname="n0", group="fish", subnet="/254", free_gb=152.5),
        NodeInfo(hostname="n1", group="fish", subnet="/254", free_gb=147.5),
    ]

    with pytest.raises(InsufficientCapacityError):
        plan_placement(nodes, SHARDS, dataset_gb=100.0, replication=2, min_free_gb=50.0)


def test_fewer_nodes_than_replication_factor_raises():
    nodes = [NodeInfo(hostname="n0", group="fish", subnet="/254", free_gb=400.0)]

    with pytest.raises(InsufficientCapacityError):
        plan_placement(nodes, SHARDS, dataset_gb=100.0, replication=2)


def test_placement_is_proportional_among_equal_capacity_nodes():
    """replication=1: with only 2 nodes, replication=2 (the default) would
    force both nodes to hold every shard regardless of weight."""
    fish = NodeInfo(hostname="fish", group="fish", subnet="/254", free_gb=900.0)
    orchid = NodeInfo(hostname="orchid", group="orchids", subnet="/254", free_gb=900.0)

    placement = plan_placement([fish, orchid], SHARDS, dataset_gb=100.0, replication=1)

    assert len(placement["fish"]) > len(placement["orchid"])
