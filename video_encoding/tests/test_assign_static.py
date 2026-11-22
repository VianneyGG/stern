from __future__ import annotations

import pytest

from video_encoding.mesh.assignment import assign_static


def test_every_expected_shard_is_assigned_exactly_once():
    holdings = {
        "node001": ["shard-a", "shard-b"],
        "node002": ["shard-b", "shard-c"],
    }
    result = assign_static(holdings, {"shard-a", "shard-b", "shard-c"})

    assigned = [s for shards in result.values() for s in shards]
    assert sorted(assigned) == ["shard-a", "shard-b", "shard-c"]
    assert len(assigned) == len(set(assigned)), "a shard was assigned twice"


def test_a_node_is_only_ever_assigned_shards_it_physically_holds():
    """The whole point: attribution can no longer disagree with reality."""
    holdings = {
        "node001": ["shard-a", "shard-b"],
        "node002": ["shard-c"],
    }
    result = assign_static(holdings, {"shard-a", "shard-b", "shard-c"})

    for node, shards in result.items():
        assert set(shards) <= set(holdings[node]), f"{node} assigned a shard it lacks"


def test_load_is_balanced_across_holders_rather_than_dumped_on_the_first():
    """Rounds are time-budgeted, so a node handed 3x its share of the data
    simply trains less of it -- the imbalance shows up as silently skipped
    data, not as a longer round."""
    shards = [f"shard-{i:03d}" for i in range(30)]
    holdings = {
        "node001": list(shards),
        "node002": list(shards),
        "node004": list(shards),
    }
    result = assign_static(holdings, set(shards))

    counts = sorted(len(v) for v in result.values())
    assert counts == [10, 10, 10]


def test_a_node_holding_nothing_gets_an_empty_assignment_not_a_missing_key():
    """node017, node030, node069 and friends hold zero shards. They must
    appear in the assignment as empty rather than vanish -- the watcher
    reads this file to decide who to submit."""
    holdings = {"node001": ["shard-a"], "node017": []}
    result = assign_static(holdings, {"shard-a"})

    assert result["node017"] == []
    assert result["node001"] == ["shard-a"]


def test_an_expected_shard_no_node_holds_raises_naming_it():
    """This is the 2026-08-20 incident as a unit test: node055 was assigned 19
    shards it did not hold. That must now be impossible to express."""
    holdings = {"node001": ["shard-a"]}
    with pytest.raises(ValueError, match="shard-b"):
        assign_static(holdings, {"shard-a", "shard-b"})


def test_shards_held_but_not_expected_are_left_unassigned():
    """A leftover from an older seeding is not training data."""
    holdings = {"node001": ["shard-a", "shard-stale"]}
    result = assign_static(holdings, {"shard-a"})
    assert result["node001"] == ["shard-a"]


def test_assignment_is_deterministic_across_calls():
    holdings = {
        "node001": ["shard-a", "shard-b", "shard-c"],
        "node002": ["shard-a", "shard-b", "shard-c"],
    }
    expected = {"shard-a", "shard-b", "shard-c"}
    assert assign_static(holdings, expected) == assign_static(holdings, expected)


def test_every_holder_gets_at_least_one_shard_when_holdings_overlap():
    """A node holding few, heavily-replicated shards must still train.

    Reproduces the 2026-08-21 factorial collapse: `assign_static` broke
    load ties on node name, so a node whose whole (small) holding was also
    held by alphabetically-earlier nodes lost every tie and was assigned
    zero shards. `mesh_round` then reported train_loss=nan / 0.0 samp/s and
    skipped its push, dropping merge quorum to a single contributor.
    """
    # node068 holds only two shards, and four other nodes that sort earlier
    # hold them too. Every one of those competitors is still at load 0 when
    # s1/s2 are handed out, so node068 loses both ties and starves.
    holdings = {
        "node002": ["s1", "s2", "s3", "s4", "s5", "s6"],
        "node004": ["s1", "s2", "s3", "s4", "s5", "s6"],
        "node008": ["s1", "s2", "s3", "s4", "s5", "s6"],
        "node027": ["s1", "s2", "s3", "s4", "s5", "s6"],
        "node068": ["s1", "s2"],
    }
    expected = {"s1", "s2", "s3", "s4", "s5", "s6"}

    assignment = assign_static(holdings, expected)

    assert len(assignment["node068"]) >= 1, (
        "node068 holds shards but was assigned none; it would train on zero "
        f"samples and skip its push. assignment={assignment}"
    )
    # The core invariants must still hold.
    flat = [s for shards in assignment.values() for s in shards]
    assert sorted(flat) == sorted(expected), "shards must be partitioned exactly once"
    for node, shards in assignment.items():
        assert set(shards) <= set(holdings[node]), (
            f"{node} assigned a shard it does not hold"
        )
