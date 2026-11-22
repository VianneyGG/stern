from __future__ import annotations

import json

import pytest

from video_encoding.mesh.assignment import (
    read_my_shards,
    reassign_failed,
)

SHARDS = [f"s{i}" for i in range(6)]


def test_reassign_routes_orphans_to_a_replica_holder_for_zero_network():
    placement = {"a": ["s0", "s1"], "b": ["s0", "s1", "s2"], "c": ["s2"]}
    assignments = {"a": ["s0", "s1"], "b": ["s2"], "c": []}

    r, dropped = reassign_failed("a", assignments, placement)

    assert "a" not in r
    assert set(r["b"]) >= {"s0", "s1"}
    assert dropped == []
    flattened = sorted(shard for shards in r.values() for shard in shards)
    assert flattened == ["s0", "s1", "s2"]


def test_reassign_drops_a_shard_no_survivor_holds_instead_of_faking_an_owner():
    """The 2026-08-20 fault, in its second location.

    reassign_failed used to fall back to the least-loaded survivor when no
    survivor held a replica, exactly as assign_locality_first's phase-2
    remainder-fill did. The named node then had a shard it could not open;
    with the P2P fetch dead that is a hard FileNotFoundError, and before
    that it was a silently skipped shard. Losing coverage of s0 is the
    honest outcome and it is reported, not hidden.
    """
    placement = {"a": ["s0"], "b": [], "c": []}
    assignments = {"a": ["s0"], "b": [], "c": []}

    r, dropped = reassign_failed("a", assignments, placement)

    assert "a" not in r
    assert dropped == ["s0"]
    assert all("s0" not in shards for shards in r.values())


def test_reassign_never_hands_a_survivor_a_shard_it_does_not_hold():
    placement = {"a": ["s0", "s1", "s2"], "b": ["s1"], "c": ["s2"]}
    assignments = {"a": ["s0", "s1", "s2"], "b": [], "c": []}

    r, dropped = reassign_failed("a", assignments, placement)

    for node, shards in r.items():
        assert set(shards) <= set(placement[node]), node
    assert dropped == ["s0"]


def test_reassign_balances_load_among_several_real_holders():
    placement = {"a": ["s0", "s1"], "b": ["s0", "s1"], "c": ["s0", "s1"]}
    assignments = {"a": ["s0", "s1"], "b": [], "c": []}

    r, dropped = reassign_failed("a", assignments, placement)

    assert dropped == []
    assert sorted(len(v) for v in r.values()) == [1, 1]


def test_read_my_shards_raises_on_empty_list_for_present_host(tmp_path):
    path = tmp_path / "shard_assignments_run.json"
    path.write_text(json.dumps({"node040": []}))

    with pytest.raises(KeyError, match="node040.*empty|empty.*node040"):
        read_my_shards(path, "node040")


def test_read_my_shards_raises_on_missing_host_unchanged(tmp_path):
    """Existing missing-host behaviour must be unaffected by the empty-list
    check -- a host absent from the assignments file is a different failure
    mode from a host present with an empty list."""
    path = tmp_path / "shard_assignments_run.json"
    path.write_text(json.dumps({"node011": ["a.tar"]}))

    with pytest.raises(KeyError, match="node040"):
        read_my_shards(path, "node040")


def test_read_my_shards_returns_shards_for_present_nonempty_host(tmp_path):
    path = tmp_path / "shard_assignments_run.json"
    path.write_text(json.dumps({"node011": ["a.tar"]}))

    assert read_my_shards(path, "node011") == ["a.tar"]
