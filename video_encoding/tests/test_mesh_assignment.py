from __future__ import annotations

from video_encoding.mesh.assignment import assign_locality_first, reassign_failed

SHARDS = [f"s{i}" for i in range(6)]


def test_nodes_are_given_shards_they_already_hold():
    placement = {"a": ["s0", "s1", "s2"], "b": ["s3", "s4", "s5"]}
    counts = {"a": 3, "b": 3}

    r = assign_locality_first(counts, placement, SHARDS)

    assert set(r["a"]) == {"s0", "s1", "s2"}
    assert set(r["b"]) == {"s3", "s4", "s5"}


def test_every_shard_assigned_exactly_once():
    placement = {"a": ["s0", "s1"], "b": ["s1", "s2"]}
    counts = {"a": 3, "b": 3}

    r = assign_locality_first(counts, placement, SHARDS)

    flattened = sorted(shard for shards in r.values() for shard in shards)
    assert flattened == sorted(SHARDS)


def test_counts_are_honoured_exactly():
    placement = {"a": ["s0"], "b": []}
    counts = {"a": 2, "b": 4}

    r = assign_locality_first(counts, placement, SHARDS)

    assert len(r["a"]) == 2
    assert len(r["b"]) == 4


def test_remote_remainder_is_filled_when_holdings_are_short():
    placement = {"a": ["s0"], "b": ["s1"]}
    counts = {"a": 3, "b": 3}

    r = assign_locality_first(counts, placement, SHARDS)

    assert "s0" in r["a"]
    assert "s1" in r["b"]
    flattened = sorted(shard for shards in r.values() for shard in shards)
    assert flattened == sorted(SHARDS)


def test_reassign_routes_orphans_to_a_replica_holder_for_zero_network():
    placement = {"a": ["s0", "s1"], "b": ["s0", "s1", "s2"], "c": ["s2"]}
    assignments = {"a": ["s0", "s1"], "b": ["s2"], "c": []}

    r = reassign_failed("a", assignments, placement)

    assert "a" not in r
    assert set(r["b"]) >= {"s0", "s1"}
    flattened = sorted(shard for shards in r.values() for shard in shards)
    assert flattened == ["s0", "s1", "s2"]


def test_reassign_falls_back_to_least_loaded_when_no_survivor_holds_it():
    placement = {"a": ["s0"], "b": [], "c": []}
    assignments = {"a": ["s0"], "b": [], "c": []}

    r = reassign_failed("a", assignments, placement)

    assert "a" not in r
    holders = [h for h, shards in r.items() if "s0" in shards]
    assert len(holders) == 1
