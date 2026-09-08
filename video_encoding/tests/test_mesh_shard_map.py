from __future__ import annotations

import pytest

from video_encoding.mesh.shard_map import (
    Holder,
    find_holders,
    load_shard_map,
    pick_holder,
    save_shard_map,
)


def test_roundtrip_preserves_holders(tmp_path):
    path = tmp_path / "shard_map.json"
    original = {
        "shard-000001.tar": [Holder("node001", "/254"), Holder("node047", "/253")],
        "shard-000002.tar": [Holder("carpe", "/252", port=8001)],
    }

    save_shard_map(path, original)
    loaded = load_shard_map(path)

    assert loaded == original


def test_find_holders_returns_empty_for_unknown_shard():
    shard_map = {"shard-000001.tar": [Holder("node001", "/254")]}
    assert find_holders(shard_map, "shard-999999.tar") == []


def test_pick_holder_prefers_same_subnet():
    local = Holder("node001", "/254")
    remote = Holder("node047", "/253")
    shard_map = {"shard-000001.tar": [remote, local]}

    picked = pick_holder(shard_map, "shard-000001.tar", requester_subnet="/254")

    assert picked == local


def test_pick_holder_falls_back_cross_subnet_when_no_local_holder():
    remote = Holder("node047", "/253")
    shard_map = {"shard-000001.tar": [remote]}

    picked = pick_holder(shard_map, "shard-000001.tar", requester_subnet="/254")

    assert picked == remote


def test_pick_holder_honours_exclude_for_retry():
    first = Holder("node001", "/254")
    second = Holder("node047", "/254")
    shard_map = {"shard-000001.tar": [first, second]}

    picked = pick_holder(
        shard_map,
        "shard-000001.tar",
        requester_subnet="/254",
        exclude=frozenset({first.host}),
    )

    assert picked == second


def test_pick_holder_returns_none_when_all_excluded():
    first = Holder("node001", "/254")
    second = Holder("node047", "/254")
    shard_map = {"shard-000001.tar": [first, second]}

    picked = pick_holder(
        shard_map,
        "shard-000001.tar",
        requester_subnet="/254",
        exclude=frozenset({first.host, second.host}),
    )

    assert picked is None


@pytest.mark.integration
def test_pick_holder_falls_through_same_subnet_to_cross_subnet_across_two_retries():
    """Excluding one same-subnet holder should still prefer the other
    same-subnet holder (not fall back to cross-subnet prematurely); only
    once *both* same-subnet holders are excluded should the third retry
    reach the cross-subnet holder. Only single-exclude and all-excluded
    were covered before -- not this middle, most-common-in-practice step."""
    local_a = Holder("node001", "/254")
    local_b = Holder("node047", "/254")
    remote = Holder("carpe", "/253")
    shard_map = {"shard-000001.tar": [local_a, local_b, remote]}

    after_one_exclude = pick_holder(
        shard_map,
        "shard-000001.tar",
        requester_subnet="/254",
        exclude=frozenset({local_a.host}),
    )
    assert after_one_exclude == local_b

    after_both_local_excluded = pick_holder(
        shard_map,
        "shard-000001.tar",
        requester_subnet="/254",
        exclude=frozenset({local_a.host, local_b.host}),
    )
    assert after_both_local_excluded == remote


def test_pick_holder_spreads_load_across_equal_replicas():
    host_a = Holder("node001", "/254")
    host_b = Holder("node047", "/254")
    shard_map = {f"shard-{i:06d}.tar": [host_a, host_b] for i in range(50)}

    picked_hosts = {
        pick_holder(shard_map, shard, requester_subnet="/254").host
        for shard in shard_map
    }

    assert picked_hosts == {"node001", "node047"}
