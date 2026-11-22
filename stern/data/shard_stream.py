"""Shard-major HTTP streaming for single-node runs with no local shard copy.

Shards are fetched one at a time from the shard host. The next one is
prefetched while the current one trains, then evicted. This is separate from
the random-access readers in wds_frames.py and SSLWindowDataset: those open a
different shard on almost every sample, which would re-download multi-GB shards
over a slow uplink. Here one download serves every window in its shard.
Mirrors the prefetch/evict pattern of mesh_shard_dataset.py, without the
mesh placement machinery.
"""

from __future__ import annotations

import logging
import os
import queue
import random
import threading
import time
import uuid
from pathlib import Path
from typing import Iterator

import requests
import torch
from torch.utils.data import IterableDataset, get_worker_info

from stern.data.wds_frames import read_shard_frames_indexed
from stern.data.windows import window_starts
from stern.models.videomae.encoder import preprocess_window

logger = logging.getLogger(__name__)


def fetch_shard(
    shard_name: str,
    *,
    base_url: str,
    cache_dir: Path,
    timeout: float = 30.0,
    max_retries: int = 3,
    retry_backoff_s: float = 1.0,
) -> Path:
    """Download `shard_name` from `base_url` into `cache_dir`, unless cached.

    The size is checked against Content-Length when the server sends one. The
    download goes to a per-process, per-thread tmp name and is then renamed
    atomically. Concurrent calls are safe for different shard names. Workers
    never share a shard.

    Retries transient failures with linear backoff. Quick tunnels drop
    connections under sustained large transfers, and a single attempt
    fails often enough to starve training, so do not drop the retries."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / shard_name
    if dest.exists():
        return dest

    tmp = (
        cache_dir
        / f".{shard_name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex[:8]}.tmp"
    )
    try:
        for attempt in range(1, max_retries + 1):
            try:
                with requests.get(
                    f"{base_url.rstrip('/')}/{shard_name}",
                    stream=True,
                    timeout=timeout,
                ) as resp:
                    resp.raise_for_status()
                    expected = resp.headers.get("Content-Length")
                    written = 0
                    with open(tmp, "wb") as fh:
                        for chunk in resp.iter_content(chunk_size=1 << 20):
                            fh.write(chunk)
                            written += len(chunk)
                if expected is not None and written != int(expected):
                    raise IOError(
                        f"fetch_shard: {shard_name} truncated: got {written} bytes, expected {expected}"
                    )
                break
            except (requests.exceptions.RequestException, IOError):
                if attempt == max_retries:
                    raise
                time.sleep(retry_backoff_s * attempt)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest


class StreamingShardDataset(IterableDataset):
    """Streams SSL windows from an HTTP-hosted shard corpus, shard-major.

    Each shard is evicted from `cache_dir` once its windows are used up. At
    most two shards are resident: the current one and the prefetched next."""

    def __init__(
        self,
        *,
        base_url: str,
        cache_dir: Path,
        num_frames: int,
        image_size: int,
        overlap: float,
        frame_stride: int = 1,
        seed: int = 0,
    ):
        self._base_url = base_url
        self._cache_dir = Path(cache_dir)
        self._num_frames = num_frames
        self._image_size = image_size
        self._overlap = overlap
        self._frame_stride = frame_stride
        self._seed = seed
        self._epoch = 0

        manifest_resp = requests.get(
            f"{base_url.rstrip('/')}/manifest.json", timeout=30.0
        )
        manifest_resp.raise_for_status()
        self._shard_index: dict[str, dict] = manifest_resp.json()
        self._shard_names = sorted(
            {entry["shard"] for entry in self._shard_index.values()}
        )

    def _worker_shards(self) -> list[str]:
        worker_info = get_worker_info()
        if worker_info is None:
            return self._shard_names
        return self._shard_names[worker_info.id :: worker_info.num_workers]

    def _spawn_fetch(
        self, shard_name: str, result_queue: "queue.Queue"
    ) -> threading.Thread:
        def _run() -> None:
            try:
                path = fetch_shard(
                    shard_name, base_url=self._base_url, cache_dir=self._cache_dir
                )
                result_queue.put((path, None))
            except Exception as exc:  # noqa: BLE001 -- passed to the consumer, not fatal
                result_queue.put((None, exc))

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        return thread

    def __iter__(self) -> Iterator[torch.Tensor]:
        shard_names = list(self._worker_shards())
        if not shard_names:
            return

        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        # Reseed per epoch: without this, every epoch replays the exact same
        # shard and window order (a no-op shuffle) while still paying the
        # full corpus re-download cost.
        rng = random.Random(f"{self._seed}:{worker_id}:{self._epoch}")
        self._epoch += 1
        rng.shuffle(shard_names)

        result_queue: "queue.Queue" = queue.Queue()
        pending_thread = self._spawn_fetch(shard_names[0], result_queue)

        for i, shard_name in enumerate(shard_names):
            pending_thread.join()
            path, error = result_queue.get()

            next_thread = None
            if i + 1 < len(shard_names):
                next_thread = self._spawn_fetch(shard_names[i + 1], result_queue)

            if error is not None:
                logger.warning(
                    "shard_stream: skipping unfetchable shard %s: %s", shard_name, error
                )
                pending_thread = next_thread
                continue

            windows = [
                (clip_key, start)
                for clip_key, entry in self._shard_index.items()
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
                frames = read_shard_frames_indexed(
                    path, clip_key, idx, size=self._image_size
                )
                yield preprocess_window(
                    frames, self._num_frames, self._image_size
                ).squeeze(0)

            path.unlink(missing_ok=True)
            pending_thread = next_thread
