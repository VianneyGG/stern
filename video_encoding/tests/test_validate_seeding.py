from __future__ import annotations

import json

import pytest

from video_encoding.scripts.validate_seeding import (
    GOLDEN_DIR,
    GOLDEN_HOST,
    build_manifest,
    parse_listing,
    scan_pool,
)

pytestmark = pytest.mark.integration

# `ls -l`-ish output as the remote listing command emits it: "<size> <name>".
GOLDEN_TRAIN = "100 shard-000000.tar\n100 shard-000001.tar\n"
GOLDEN_VAL = "50 shard-000000.tar\n"


def test_parse_listing_reads_size_and_name_ignoring_blank_lines():
    assert parse_listing("100 shard-000000.tar\n\n 50 shard-000001.tar \n") == {
        "shard-000000.tar": 100,
        "shard-000001.tar": 50,
    }


def test_parse_listing_ignores_non_shard_entries():
    """`manifest.json` and the `val` directory itself appear in the listing
    and must not be mistaken for shards."""
    assert parse_listing("12 manifest.json\n100 shard-000000.tar\n4096 val\n") == {
        "shard-000000.tar": 100
    }


def test_parse_listing_of_an_empty_directory_is_an_empty_dict():
    assert parse_listing("") == {}


def _golden_ssh(host, command):
    assert host in ("node001", GOLDEN_HOST)
    return GOLDEN_VAL if "/val" in command else GOLDEN_TRAIN


def test_build_manifest_returns_train_and_val_separately():
    train, val = build_manifest(
        "node001", "/Data/$USER/voxceleb2_shards", run_ssh=_golden_ssh
    )
    assert train == {"shard-000000.tar": 100, "shard-000001.tar": 100}
    assert val == {"shard-000000.tar": 50}


def _pool_ssh(holdings):
    """Fake ssh serving each node its own listing, with both manifests present."""

    def run_ssh(host, command):
        if "manifest.json" in command:
            return "ok\n"
        if "/val" in command:
            return GOLDEN_VAL
        return "".join(f"100 {name}\n" for name in holdings.get(host, []))

    return run_ssh


def test_scan_pool_collects_holdings_val_and_manifest_presence_per_node():
    scan, val_scan, has_manifest = scan_pool(
        ["node001", "node002"],
        run_ssh=_pool_ssh(
            {"node001": ["shard-000000.tar"], "node002": ["shard-000001.tar"]}
        ),
    )
    assert scan == {
        "node001": {"shard-000000.tar": 100},
        "node002": {"shard-000001.tar": 100},
    }
    assert val_scan["node001"] == {"shard-000000.tar": 50}
    assert has_manifest["node001"] == (True, True)


def test_scan_pool_records_an_unreachable_node_as_holding_nothing():
    """An unreachable node must not vanish from the report -- silently
    dropping it is exactly how the stale shard map hid node069's empty disk."""

    def run_ssh(host, command):
        if host == "node002":
            raise OSError("ssh: connect to host node002 port 22: No route to host")
        if "manifest.json" in command:
            return "ok\n"
        if "/val" in command:
            return GOLDEN_VAL
        return GOLDEN_TRAIN

    scan, _val_scan, has_manifest = scan_pool(["node001", "node002"], run_ssh=run_ssh)
    assert scan["node002"] == {}
    assert has_manifest["node002"] == (False, False)


def _run_main(tmp_path, holdings, monkeypatch, *extra):
    import video_encoding.scripts.validate_seeding as vs

    monkeypatch.setattr(vs, "tier_nodes", lambda pool: sorted(holdings))
    pool_ssh = _pool_ssh(holdings)

    def run_ssh(host, command):
        return (
            _golden_ssh(host, command)
            if host == GOLDEN_HOST and GOLDEN_DIR in command
            else pool_ssh(host, command)
        )

    monkeypatch.setattr(vs, "_run_ssh", run_ssh)
    return vs.main(["--pool", "fish", "--metadata-dir", str(tmp_path), *extra])


def test_main_exits_zero_and_writes_both_artifacts_for_a_complete_pool(
    tmp_path, monkeypatch
):
    rc = _run_main(
        tmp_path,
        {"node001": ["shard-000000.tar"], "node002": ["shard-000001.tar"]},
        monkeypatch,
    )
    assert rc == 0

    manifest = json.loads((tmp_path / "shard_manifest.json").read_text())
    assert manifest["train"] == {"shard-000000.tar": 100, "shard-000001.tar": 100}
    assert manifest["val"] == {"shard-000000.tar": 50}

    holdings = json.loads((tmp_path / "verified_holdings.json").read_text())
    assert holdings["fish"] == {
        "node001": ["shard-000000.tar"],
        "node002": ["shard-000001.tar"],
    }


def test_main_exits_one_when_a_shard_is_missing_pool_wide(tmp_path, monkeypatch):
    rc = _run_main(
        tmp_path, {"node001": ["shard-000000.tar"], "node002": []}, monkeypatch
    )
    assert rc == 1


def test_main_writes_holdings_even_on_failure_so_the_true_state_is_recorded(
    tmp_path, monkeypatch
):
    """The artifact is a measurement, not a certificate. A failed pool is
    exactly the case where you most need its real holdings on disk."""
    _run_main(tmp_path, {"node001": ["shard-000000.tar"], "node002": []}, monkeypatch)
    holdings = json.loads((tmp_path / "verified_holdings.json").read_text())
    assert holdings["fish"]["node002"] == []


def test_main_merges_into_an_existing_holdings_file_without_dropping_other_pools(
    tmp_path, monkeypatch
):
    (tmp_path / "verified_holdings.json").write_text(
        json.dumps({"birds": {"node039": ["shard-000009.tar"]}})
    )
    _run_main(
        tmp_path,
        {"node001": ["shard-000000.tar"], "node002": ["shard-000001.tar"]},
        monkeypatch,
    )
    holdings = json.loads((tmp_path / "verified_holdings.json").read_text())
    assert holdings["birds"] == {"node039": ["shard-000009.tar"]}
    assert "node001" in holdings["fish"]


def test_main_scans_an_explicit_roster_instead_of_the_tier(tmp_path, monkeypatch):
    """A cross-tier pool has no tier to look up, and for a tier pool the two
    thirds of the tier the run never touches must not gate it."""
    import video_encoding.scripts.validate_seeding as vs

    monkeypatch.setattr(
        vs, "tier_nodes", lambda pool: pytest.fail("tier_nodes called despite --nodes")
    )
    rc = _run_main(
        tmp_path,
        {"node001": ["shard-000000.tar"], "node002": ["shard-000001.tar"]},
        monkeypatch,
        "--nodes",
        "node001,node002",
    )
    assert rc == 0
    holdings = json.loads((tmp_path / "verified_holdings.json").read_text())
    assert sorted(holdings["fish"]) == ["node001", "node002"]


def test_main_refuses_an_empty_explicit_roster(tmp_path, monkeypatch):
    """An empty scan reports coverage over zero nodes; exiting 0 there would
    green-light a run against nothing."""
    rc = _run_main(
        tmp_path,
        {"node001": ["shard-000000.tar"]},
        monkeypatch,
        "--nodes",
        " , ",
    )
    assert rc == 1


def test_main_evicts_scanned_nodes_from_the_pools_they_left(tmp_path, monkeypatch):
    """mesh_tick flattens every pool into {node: shards} with last-key-wins,
    and pools are written sorted -- so a node left behind in an alphabetically
    later pool would override its own fresh holdings."""
    (tmp_path / "verified_holdings.json").write_text(
        json.dumps({"zzz-old-pool": {"node001": ["shard-999999.tar"], "node069": []}})
    )
    rc = _run_main(
        tmp_path,
        {"node001": ["shard-000000.tar"], "node002": ["shard-000001.tar"]},
        monkeypatch,
    )
    assert rc == 0

    holdings = json.loads((tmp_path / "verified_holdings.json").read_text())
    assert "node001" not in holdings["zzz-old-pool"]
    assert holdings["zzz-old-pool"] == {"node069": []}, "unrelated nodes must survive"
    assert holdings["fish"]["node001"] == ["shard-000000.tar"]
