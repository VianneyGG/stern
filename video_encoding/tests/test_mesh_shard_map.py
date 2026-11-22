from __future__ import annotations

from video_encoding.mesh.shard_map import (
    Holder,
    find_holders,
    load_shard_map,
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
