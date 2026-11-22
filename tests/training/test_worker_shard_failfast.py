from __future__ import annotations

import pytest

from stern.data.mesh_shard_dataset import resolve_shard
from stern.training.mesh_round import _require_local_shards


def test_resolve_shard_returns_the_seeded_copy_when_present(tmp_path):
    (tmp_path / "shard-000000.tar").write_bytes(b"x")
    path, was_fetched = resolve_shard("shard-000000.tar", tmp_path, {}, "/254")
    assert path == tmp_path / "shard-000000.tar"
    assert was_fetched is False


def test_resolve_shard_raises_instead_of_fetching_a_missing_shard(tmp_path):
    """Before 2026-08-20 this fell through to HTTP. Nothing was serving, so
    every miss became a skipped shard and a zero-sample round."""
    with pytest.raises(FileNotFoundError, match="no fetch fallback"):
        resolve_shard("shard-000000.tar", tmp_path, {}, "/254")


def test_require_local_shards_is_silent_when_every_shard_is_present(tmp_path):
    for name in ("shard-000000.tar", "shard-000001.tar"):
        (tmp_path / name).write_bytes(b"x")
    _require_local_shards(
        ["shard-000000.tar", "shard-000001.tar"], tmp_path, run_id="R", node="node001"
    )


def test_require_local_shards_exits_when_an_assigned_shard_is_absent(tmp_path):
    """The regression test for the whole incident: node055 was assigned 19
    shards it held none of and reported it as a warning."""
    (tmp_path / "shard-000000.tar").write_bytes(b"x")
    with pytest.raises(SystemExit, match="shard-000009"):
        _require_local_shards(
            ["shard-000000.tar", "shard-000009.tar"],
            tmp_path,
            run_id="R",
            node="node055",
        )


def test_require_local_shards_truncates_a_long_absent_list(tmp_path):
    with pytest.raises(SystemExit, match="19 of 19"):
        _require_local_shards(
            [f"shard-{i:06d}.tar" for i in range(19)],
            tmp_path,
            run_id="R",
            node="node055",
        )
