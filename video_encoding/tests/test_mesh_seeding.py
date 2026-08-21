from __future__ import annotations

import pytest

from video_encoding.mesh.seeding import (
    holdings_from_scan,
    validate,
)

MANIFEST = {"shard-000000.tar": 100, "shard-000001.tar": 100, "shard-000002.tar": 100}
VAL_MANIFEST = {"shard-000000.tar": 50}
BOTH_MANIFESTS = (True, True)


def _validate(scan, *, val_scan=None, has_manifest=None):
    """Default every node to a complete val/ set and both manifest.json files,
    so each test isolates the one defect it is about."""
    if val_scan is None:
        val_scan = {node: dict(VAL_MANIFEST) for node in scan}
    if has_manifest is None:
        has_manifest = {node: BOTH_MANIFESTS for node in scan}
    return validate("fish", MANIFEST, VAL_MANIFEST, scan, val_scan, has_manifest)


def test_a_fully_seeded_pool_is_ok_with_full_coverage():
    report = _validate(
        {
            "node001": {"shard-000000.tar": 100, "shard-000001.tar": 100},
            "node002": {"shard-000002.tar": 100},
        }
    )
    assert report.ok
    assert report.coverage == 1.0
    assert report.missing_pool_wide == ()


def test_a_shard_absent_from_a_node_is_reported_as_missing_not_silently_dropped():
    report = _validate(
        {
            "node001": {"shard-000000.tar": 100},
            "node002": {"shard-000002.tar": 100},
        }
    )
    assert not report.ok
    assert report.missing_pool_wide == ("shard-000001.tar",)
    assert report.coverage == pytest.approx(2 / 3)


def test_a_truncated_shard_is_wrong_size_and_does_not_count_toward_coverage():
    report = _validate(
        {
            "node001": {"shard-000000.tar": 100, "shard-000001.tar": 7},
            "node002": {"shard-000002.tar": 100},
        }
    )
    assert not report.ok
    by_node = {n.node: n for n in report.nodes}
    assert by_node["node001"].wrong_size == ("shard-000001.tar",)
    # Classified once, as wrong_size -- not double-counted as missing too.
    # Under 1x sharding a node's `missing` is never empty (it holds a slice
    # of the manifest, not all of it), so the claim is about the truncated
    # shard specifically, not about the field being empty.
    assert "shard-000001.tar" not in by_node["node001"].missing
    assert by_node["node001"].missing == ("shard-000002.tar",)
    # A half-transferred shard is worse than an absent one: it resolves
    # locally and then fails mid-epoch. It must not be counted as covered.
    assert report.missing_pool_wide == ("shard-000001.tar",)
    assert report.coverage == pytest.approx(2 / 3)


def test_a_shard_not_in_the_golden_manifest_is_unexpected_but_not_fatal():
    report = _validate(
        {
            "node001": {"shard-000000.tar": 100, "shard-000001.tar": 100, "junk.tar": 3},
            "node002": {"shard-000002.tar": 100},
        }
    )
    by_node = {n.node: n for n in report.nodes}
    assert by_node["node001"].unexpected == ("junk.tar",)
    # Leftovers from an older seeding do not break training.
    assert report.ok


def test_a_missing_val_shard_fails_the_pool():
    """A missing val/ currently yields an empty val dataset and a meaningless
    val_loss in silence (mesh_round.py). It must fail loudly instead."""
    report = _validate(
        {"node001": {"shard-000000.tar": 100, "shard-000001.tar": 100, "shard-000002.tar": 100}},
        val_scan={"node001": {}},
    )
    assert not report.ok
    assert report.nodes[0].val_missing == ("shard-000000.tar",)


def test_a_missing_train_manifest_fails_the_pool():
    report = _validate(
        {"node001": {"shard-000000.tar": 100, "shard-000001.tar": 100, "shard-000002.tar": 100}},
        has_manifest={"node001": (False, True)},
    )
    assert not report.ok
    assert report.nodes[0].manifest_missing == ("manifest.json",)


def test_a_missing_val_manifest_fails_the_pool_separately():
    """val/ has its own non-interchangeable manifest (seed_mesh.py) and
    MeshShardIterableDataset loads it unconditionally."""
    report = _validate(
        {"node001": {"shard-000000.tar": 100, "shard-000001.tar": 100, "shard-000002.tar": 100}},
        has_manifest={"node001": (True, False)},
    )
    assert not report.ok
    assert report.nodes[0].manifest_missing == ("val/manifest.json",)


def test_an_empty_pool_scan_is_zero_coverage_not_a_crash():
    report = validate("fish", MANIFEST, VAL_MANIFEST, {}, {}, {})
    assert not report.ok
    assert report.coverage == 0.0
    assert report.nodes == ()


def test_holdings_from_scan_drops_sizes_and_sorts_deterministically():
    holdings = holdings_from_scan(
        {
            "node002": {"shard-000002.tar": 100},
            "node001": {"shard-000001.tar": 100, "shard-000000.tar": 100},
        }
    )
    assert holdings == {
        "node001": ["shard-000000.tar", "shard-000001.tar"],
        "node002": ["shard-000002.tar"],
    }


def test_holdings_from_scan_output_is_json_round_trippable():
    """verified_holdings.json is the artifact Task 2 writes; tuples would
    silently become lists on reload and break equality comparisons later."""
    import json

    holdings = holdings_from_scan({"node001": {"shard-000000.tar": 100}})
    assert json.loads(json.dumps(holdings)) == holdings
