"""Iterable dataset over mesh-resident WDS shards, with prefetch and eviction.

Two shard populations live under `mesh_root`. They are **not** interchangeable:

- `mesh_root/` itself holds this node's permanent seeded shards. Peers rely on
  them (see `stern/mesh/shard_map.py`), so they must persist across rounds and
  runs. The training loop never deletes them.
- `mesh_root/cache/` holds shards staged for the current round. They are
  evicted right after use to bound local disk.

`resolve_shard` and `evict_if_fetched` are separate functions on purpose. Merging
them into one delete-after-use path would delete seeded shards that peers
depend on.
"""

from __future__ import annotations

import logging
import queue
import random
import threading
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from stern.data.augment import apply_group_augment, sample_group_augment
from stern.data.wds_frames import (
    load_shard_index,
    read_shard_frames_indexed,
)
from stern.data.windows import window_starts
from stern.mesh.shard_map import Holder
from stern.models.videomae.encoder import preprocess_window

logger = logging.getLogger(__name__)

_CACHE_DIRNAME = "cache"


def resolve_shard(
    shard: str, mesh_root: Path, shard_map: dict[str, list[Holder]], subnet: str
) -> tuple[Path, bool]:
    """Locate `shard` on local disk, preferring the permanent seeded copy.

    Checks `mesh_root/shard`, then `mesh_root/cache/shard`, then raises
    FileNotFoundError. Only the seeded copy returns `was_fetched=False`. A
    leftover cache entry counts as fetched and is evicted after use.
    """
    mesh_root = Path(mesh_root)

    seeded_path = mesh_root / shard
    if seeded_path.exists():
        return seeded_path, False

    cache_dir = mesh_root / _CACHE_DIRNAME
    cached_path = cache_dir / shard
    if cached_path.exists():
        return cached_path, True

    # No network fallback. Assignment is static, and a node is only given
    # shards it holds. A miss here means assignment and disk disagree, and
    # fetching would hide that by silently skipping the shard.
    raise FileNotFoundError(
        f"resolve_shard: {shard!r} is not present under {mesh_root} and no "
        "fetch fallback exists. The shard assignment disagrees with this "
        "node's disk -- re-run python -m stern.mesh.validate_seeding."
    )


def evict_if_fetched(path: Path, mesh_root: Path) -> None:
    """Delete `path` only if it lives under `mesh_root/cache/`.

    Seeded shards are never touched. Deleting a missing path is a no-op, since
    eviction can race a concurrent cleanup.
    """
    path = Path(path)
    cache_dir = (Path(mesh_root) / _CACHE_DIRNAME).resolve()
    resolved = path.resolve()

    if cache_dir not in resolved.parents:
        # Not a file strictly inside mesh/cache/ -- either a seeded shard
        # elsewhere under mesh_root, or (defensively) mesh/cache/ itself.
        # Never delete.
        return

    path.unlink(missing_ok=True)


class MeshShardIterableDataset(IterableDataset):
    """Streams VA-probe/SSL windows from a list of mesh shards.

    One shard is active (being decoded) and one is pending (being resolved on a
    background daemon thread). Disk use under `mesh_root/cache/` therefore stays
    bounded however many shards the node holds. A shard that cannot be resolved
    is logged and skipped, so training continues on what is available.
    """

    def __init__(
        self,
        shard_names: list[str],
        *,
        mesh_root: Path,
        shard_map: dict[str, list[Holder]],
        subnet: str,
        num_frames: int,
        image_size: int,
        overlap: float,
        frame_stride: int = 1,
        augment: bool = False,
        seed: int = 0,
        raw_frames: bool = False,
    ):
        self._shard_names = list(shard_names)
        self._mesh_root = Path(mesh_root)
        self._shard_map = shard_map
        self._subnet = subnet
        self._num_frames = num_frames
        self._image_size = image_size
        self._overlap = overlap
        self._frame_stride = frame_stride
        self._augment = augment
        self._seed = seed
        self._round = 0
        # LeVJEPA's build_views takes un-normalised (T, H, W, 3) uint8 clips, as
        # SSLWindowDataset(raw_frames=True) does. It does its own resize, crop
        # and normalisation per view.
        self._raw_frames = raw_frames

    def set_round(self, round_num: int) -> None:
        """Set the round that seeds the shard and window order of the next `__iter__`.

        Without this, every round replays the same first `steps_per_round`
        windows, because the order is fixed and truncation always cuts at the
        same point."""
        self._round = round_num

    def _worker_shards(self) -> list[str]:
        worker_info = get_worker_info()
        if worker_info is None:
            return self._shard_names
        return self._shard_names[worker_info.id :: worker_info.num_workers]

    def _spawn_resolve(
        self, shard_name: str, result_queue: "queue.Queue"
    ) -> threading.Thread:
        def _run() -> None:
            try:
                path, _was_fetched = resolve_shard(
                    shard_name, self._mesh_root, self._shard_map, self._subnet
                )
                result_queue.put((path, None))
            except Exception as exc:  # noqa: BLE001 -- surfaced to the consumer, not fatal
                result_queue.put((None, exc))

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        return thread

    def __iter__(self) -> Iterator[torch.Tensor]:
        shard_names = self._worker_shards()
        if not shard_names:
            return

        # Keyed on (seed, round, worker_id), so a relaunched worker reproduces
        # the order for its round (see mesh_round.py's resume contract) without
        # relying on process state that a crash would lose. `_worker_shards` may
        # return `self._shard_names` itself, so copy before shuffling. Otherwise
        # repeated `__iter__` calls would compound permutations.
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        # random.Random rejects tuple seeds. The colon-joined string keeps the
        # fields distinct, so (1, 23) and (12, 3) cannot collide.
        rng = random.Random(f"{self._seed}:{self._round}:{worker_id}")
        # Augmentation draws use their own stream. Sharing `rng` would make the
        # `augment` flag reorder the windows of every shard after the first.
        # The flag must not change which data is seen in which order.
        aug_rng = random.Random(f"aug:{self._seed}:{self._round}:{worker_id}")
        shard_names = list(shard_names)
        rng.shuffle(shard_names)

        # One index over the whole shard set, loaded from mesh_root/manifest.json.
        # Cached shards have no manifest of their own, so never derive the index
        # from a shard's parent directory.
        shard_index = load_shard_index(self._mesh_root)

        result_queue: "queue.Queue" = queue.Queue()
        pending_thread = self._spawn_resolve(shard_names[0], result_queue)

        for i, shard_name in enumerate(shard_names):
            pending_thread.join()
            path, error = result_queue.get()

            next_thread = None
            if i + 1 < len(shard_names):
                next_thread = self._spawn_resolve(shard_names[i + 1], result_queue)

            if error is not None:
                logger.warning(
                    "mesh_shard_dataset: skipping unfetchable shard %s: %s",
                    shard_name,
                    error,
                )
                pending_thread = next_thread
                continue

            try:
                # A flat list, because the shuffle needs every window of this
                # shard up front. A shard holds about 2k windows, which is cheap.
                # Nothing is buffered across shards, so eviction stays bounded.
                windows = [
                    (clip_key, start)
                    for clip_key, entry in shard_index.items()
                    if entry["shard"] == shard_name
                    for start in window_starts(
                        entry["num_frames"],
                        self._num_frames,
                        self._overlap,
                        frame_stride=self._frame_stride,
                    )
                ]
                rng.shuffle(windows)
                for clip_key, start in windows:
                    idx = list(
                        range(
                            start,
                            start + self._num_frames * self._frame_stride,
                            self._frame_stride,
                        )
                    )
                    # Indexed, not `_at`: the shuffle makes this random access.
                    # `_at` re-walks every tar header on each window, and that
                    # scan dominates the cost, not the JPEG decode.
                    frames = read_shard_frames_indexed(
                        path, clip_key, idx, size=self._image_size
                    )
                    if self._raw_frames:
                        # No augment or normalisation here. build_views does both
                        # per view, downstream.
                        yield torch.from_numpy(np.stack(frames))
                        continue
                    if self._augment:
                        params = sample_group_augment(self._image_size, aug_rng)
                        frames = apply_group_augment(frames, params, self._image_size)
                    pixel_values = preprocess_window(
                        frames, self._num_frames, self._image_size
                    )
                    yield pixel_values.squeeze(0)
            finally:
                evict_if_fetched(path, self._mesh_root)  # no-op for seeded shards

            pending_thread = next_thread
