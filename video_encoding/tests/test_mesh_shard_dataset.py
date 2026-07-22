"""Tests for `video_encoding.data.mesh_shard_dataset`.

Targets the correctness-critical, easily-fixture-tested invariants --
`resolve_shard` (seeded-copy preference) and `evict_if_fetched` (seeded vs.
fetched-cache eviction) -- per the task-8 brief. `MeshShardIterableDataset`'s
`__iter__` prefetch/threading mechanics are not covered here: they need a
real WDS shard tarball to exercise meaningfully and are exercised by manual
cluster smoke testing instead.
"""

from __future__ import annotations

from video_encoding.data.mesh_shard_dataset import evict_if_fetched, resolve_shard

SUBNET = "/254"


def test_resolve_prefers_the_seeded_copy_and_reports_not_fetched(tmp_path):
    mesh_root = tmp_path / "mesh"
    mesh_root.mkdir()
    (mesh_root / "s0.tar").write_bytes(b"seeded")
    cache_dir = mesh_root / "cache"
    cache_dir.mkdir()
    (cache_dir / "s0.tar").write_bytes(b"also-cached")  # decoy: must not be picked

    path, was_fetched = resolve_shard("s0.tar", mesh_root, {}, SUBNET)

    assert path == mesh_root / "s0.tar"
    assert was_fetched is False


def test_resolve_finds_an_already_cached_shard_without_refetching(tmp_path):
    mesh_root = tmp_path / "mesh"
    cache_dir = mesh_root / "cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "s1.tar").write_bytes(b"cached")

    # empty shard_map: resolve_shard would raise if it actually tried to fetch
    path, was_fetched = resolve_shard("s1.tar", mesh_root, {}, SUBNET)

    assert path == cache_dir / "s1.tar"
    assert was_fetched is True


def test_evict_deletes_a_fetched_shard(tmp_path):
    mesh_root = tmp_path / "mesh"
    cache_dir = mesh_root / "cache"
    cache_dir.mkdir(parents=True)
    fetched = cache_dir / "s2.tar"
    fetched.write_bytes(b"fetched")

    evict_if_fetched(fetched, mesh_root)

    assert not fetched.exists()


def test_evict_never_deletes_a_seeded_shard(tmp_path):
    mesh_root = tmp_path / "mesh"
    mesh_root.mkdir()
    seeded = mesh_root / "s3.tar"
    seeded.write_bytes(b"seeded")

    evict_if_fetched(seeded, mesh_root)

    assert seeded.exists()


def test_evict_is_idempotent_on_a_missing_file(tmp_path):
    mesh_root = tmp_path / "mesh"
    cache_dir = mesh_root / "cache"
    cache_dir.mkdir(parents=True)
    missing = cache_dir / "does-not-exist.tar"

    evict_if_fetched(missing, mesh_root)  # must not raise
