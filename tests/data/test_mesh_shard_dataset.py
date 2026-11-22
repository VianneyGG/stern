"""Tests for `stern.data.mesh_shard_dataset`.

Targets the correctness-critical, easily-fixture-tested invariants --
`resolve_shard` (seeded-copy preference) and `evict_if_fetched` (seeded vs.
fetched-cache eviction) -- per the task-8 brief. `MeshShardIterableDataset`'s
`__iter__` prefetch/threading mechanics are not covered here: they need a
real WDS shard tarball to exercise meaningfully and are exercised by manual
cluster smoke testing instead.
"""

from __future__ import annotations

import pytest

from stern.data.mesh_shard_dataset import (
    MeshShardIterableDataset,
    evict_if_fetched,
    resolve_shard,
)

pytestmark = pytest.mark.integration

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

    assert fetched.exists() is False


def test_evict_never_deletes_a_seeded_shard(tmp_path):
    mesh_root = tmp_path / "mesh"
    mesh_root.mkdir()
    seeded = mesh_root / "s3.tar"
    seeded.write_bytes(b"seeded")

    evict_if_fetched(seeded, mesh_root)

    assert seeded.exists() is True


def test_evict_is_idempotent_on_a_missing_file(tmp_path):
    mesh_root = tmp_path / "mesh"
    cache_dir = mesh_root / "cache"
    cache_dir.mkdir(parents=True)
    missing = cache_dir / "does-not-exist.tar"

    evict_if_fetched(missing, mesh_root)  # must not raise


def test_set_round_shuffles_shard_resolution_order(tmp_path, monkeypatch):
    """Without set_round, __iter__ walks shard_names in the same fixed
    order every round -- combined with mesh_round.py's max_steps cutoff,
    every round trained on the same first windows of the same first shard.
    No real shard tarball needed: every shard is made to fail resolution
    (skip-and-continue path) so only resolution ORDER is observed, not
    decode content -- that's covered with real shards in test_mesh_round.py.
    """
    mesh_root = tmp_path / "mesh"
    mesh_root.mkdir(parents=True)
    (mesh_root / "manifest.json").write_text("{}")

    dataset = MeshShardIterableDataset(
        [f"shard-{i:06d}.tar" for i in range(8)],
        mesh_root=mesh_root,
        shard_map={},
        subnet=SUBNET,
        num_frames=4,
        image_size=32,
        overlap=0.0,
    )

    order_log: dict[int, list[str]] = {}

    def _fake_spawn_resolve(self, shard_name, result_queue):
        order_log.setdefault(self._round, []).append(shard_name)
        result_queue.put((None, RuntimeError("test: unfetchable")))

        class _AlreadyDone:
            def join(self) -> None:
                return None

        return _AlreadyDone()

    monkeypatch.setattr(MeshShardIterableDataset, "_spawn_resolve", _fake_spawn_resolve)

    dataset.set_round(0)
    list(iter(dataset))
    dataset.set_round(1)
    list(iter(dataset))

    assert sorted(order_log[0]) == sorted(order_log[1])
    assert order_log[0] != order_log[1]


def test_set_round_is_deterministic(tmp_path, monkeypatch):
    mesh_root = tmp_path / "mesh"
    mesh_root.mkdir(parents=True)
    (mesh_root / "manifest.json").write_text("{}")

    def _make_dataset():
        return MeshShardIterableDataset(
            [f"shard-{i:06d}.tar" for i in range(8)],
            mesh_root=mesh_root,
            shard_map={},
            subnet=SUBNET,
            num_frames=4,
            image_size=32,
            overlap=0.0,
        )

    def _fake_spawn_resolve(self, shard_name, result_queue, _order=None):
        _order.append(shard_name)
        result_queue.put((None, RuntimeError("test: unfetchable")))

        class _AlreadyDone:
            def join(self) -> None:
                return None

        return _AlreadyDone()

    order_a: list[str] = []
    order_b: list[str] = []

    monkeypatch.setattr(
        MeshShardIterableDataset,
        "_spawn_resolve",
        lambda self, shard_name, result_queue: _fake_spawn_resolve(
            self, shard_name, result_queue, _order=order_a
        ),
    )
    ds_a = _make_dataset()
    ds_a.set_round(3)
    list(iter(ds_a))

    monkeypatch.setattr(
        MeshShardIterableDataset,
        "_spawn_resolve",
        lambda self, shard_name, result_queue: _fake_spawn_resolve(
            self, shard_name, result_queue, _order=order_b
        ),
    )
    ds_b = _make_dataset()
    ds_b.set_round(3)
    list(iter(ds_b))

    assert order_a == order_b
