"""Iterable dataset over mesh-resident WDS shards, with prefetch and eviction.

Two shard populations live under `mesh_root` and are **not** interchangeable:

- `mesh_root/` itself -- this node's permanent seeded contribution. Peers
  pull from it via `serve_shards.sh` (`video_encoding/mesh/shard_map.py`'s
  `shard_map` lookups depend on it persisting across rounds *and runs*).
  Never deleted by the training loop.
- `mesh_root/cache/` -- shards fetched for this round via `fetch_shard`
  (`video_encoding/mesh/fetch.py`). Evicted immediately after use to bound
  `/Data`.

`resolve_shard` and `evict_if_fetched` are kept as separate, directly-tested
functions (not folded into one convenience call) precisely so a refactor
can't accidentally collapse "fetched" and "seeded" into a single
delete-after-use path -- doing so would delete this node's seeded shards out
from under every peer that depends on them.
"""

from __future__ import annotations

import logging
import queue
import random
import threading
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import IterableDataset, get_worker_info

from video_encoding.data.wds_frames import load_shard_index, read_shard_frames_at
from video_encoding.data.windows import window_starts
from video_encoding.mesh.fetch import fetch_shard
from video_encoding.mesh.shard_map import Holder
from video_encoding.models.videomaev2.encoder import preprocess_window

logger = logging.getLogger(__name__)

_CACHE_DIRNAME = "cache"


def resolve_shard(
    shard: str, mesh_root: Path, shard_map: dict[str, list[Holder]], subnet: str
) -> tuple[Path, bool]:
    """Locate `shard` on local disk, preferring the permanent seeded copy.

    Fall-through order: seeded (`mesh_root/shard`) -> already-cached
    (`mesh_root/cache/shard`) -> fetch via P2P into `mesh_root/cache/`.
    Only the seeded copy is reported as `was_fetched=False`: both the
    already-cached and the freshly-fetched cases live under `cache/` and
    must be evicted after use.
    """
    mesh_root = Path(mesh_root)

    seeded_path = mesh_root / shard
    if seeded_path.exists():
        return seeded_path, False

    cache_dir = mesh_root / _CACHE_DIRNAME
    cached_path = cache_dir / shard
    if cached_path.exists():
        return cached_path, True

    fetched_path = fetch_shard(shard, shard_map, subnet, cache_dir)
    return fetched_path, True


def evict_if_fetched(path: Path, mesh_root: Path) -> None:
    """Delete `path` only if it lives under `mesh_root/cache/`.

    A seeded shard (anywhere else under `mesh_root`, notably `mesh_root`
    itself) is never touched. Idempotent: deleting an already-missing path
    is a no-op, not an error, since eviction can race a concurrent cleanup.
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

    Keeps exactly one shard "active" (being decoded) and one "pending"
    (resolving -- local lookup or P2P fetch -- on a background daemon
    thread) at a time, so disk usage under `mesh_root/cache/` stays bounded
    regardless of how many shards this node is assigned. An unfetchable
    shard is logged and skipped rather than raised: the worker keeps
    training on what it has, and the watcher reconciles next round.
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
        seed: int = 0,
    ):
        self._shard_names = list(shard_names)
        self._mesh_root = Path(mesh_root)
        self._shard_map = shard_map
        self._subnet = subnet
        self._num_frames = num_frames
        self._image_size = image_size
        self._overlap = overlap
        self._seed = seed
        self._round = 0

    def set_round(self, round_num: int) -> None:
        """Selects the deterministic shard/window permutation the next
        __iter__ call uses. Without this, a fresh DataLoader built each
        mesh round (mesh_round.py's train_one_round) replayed the exact
        same first `steps_per_round` windows every round -- the node's
        shard/clip/window order is otherwise fixed insertion order, and
        max_steps truncation always cuts at the same point."""
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

        # Keyed on (seed, round, worker_id) so a relaunched worker process
        # reproduces the exact order its round_num implies (mesh_round.py's
        # cold-start/resume contract), rather than depending on process
        # state that a crash would lose. `_worker_shards` may return
        # `self._shard_names` itself (no worker split) -- copy before
        # shuffling so repeated __iter__ calls don't compound permutations
        # on the stored list.
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        # random.Random's seed only accepts int/float/str/bytes/None, not a
        # tuple -- a delimited string keeps the three components distinct
        # (no collision between e.g. (1, 23) and (12, 3)).
        rng = random.Random(f"{self._seed}:{self._round}:{worker_id}")
        shard_names = list(shard_names)
        rng.shuffle(shard_names)

        # Loaded once from mesh_root/manifest.json -- a single index over
        # the whole shard set. A shard fetched into mesh_root/cache/ has no
        # manifest.json of its own, so this must never be re-derived from a
        # shard's own parent directory.
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
                # Built as a flat list (not the nested generator this
                # replaces) because the shuffle below needs the full
                # per-shard window set materialized first -- one shard's
                # windows (~2k for the measured 260-shard corpus) is a
                # trivial in-memory list, and this never buffers across
                # shards so the bounded-disk eviction contract is untouched.
                windows = [
                    (clip_key, start)
                    for clip_key, entry in shard_index.items()
                    if entry["shard"] == shard_name
                    for start in window_starts(
                        entry["num_frames"], self._num_frames, self._overlap
                    )
                ]
                rng.shuffle(windows)
                for clip_key, start in windows:
                    idx = list(range(start, start + self._num_frames))
                    frames = read_shard_frames_at(
                        path, clip_key, idx, size=self._image_size
                    )
                    pixel_values = preprocess_window(
                        frames, self._num_frames, self._image_size
                    )
                    yield pixel_values.squeeze(0)
            finally:
                evict_if_fetched(path, self._mesh_root)

            pending_thread = next_thread
