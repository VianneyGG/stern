# stern/data/wds_frames.py
"""Read per-clip JPEG-frame samples from WebDataset shards built by
stern/prep/build_wds_shards.py."""

from __future__ import annotations

import functools
import io
import json
import pickle
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image


def load_shard_index(shards_root: Path) -> dict[str, dict]:
    """clip_key -> {"shard": "shard-000042.tar", "num_frames": int, "size": int}"""
    return json.loads((Path(shards_root) / "manifest.json").read_text())


def read_shard_frames_at(
    shard_path: Path, clip_key: str, indices: list[int], size: int = 224
) -> list[np.ndarray]:
    """Read the JPEG frames at `indices` from clip `clip_key`'s shard member,
    resized to size x size. Opens the shard tar fresh each call (simple,
    correct; a per-worker LRU cache is a future optimization if profiling
    shows this is a bottleneck -- not needed for correctness now)."""
    with tarfile.open(shard_path, "r") as tar:
        member = tar.extractfile(f"{clip_key}.pickle")
        frame_jpegs: list[bytes] = pickle.loads(member.read())
    return [
        np.array(
            Image.open(io.BytesIO(frame_jpegs[i]))
            .convert("RGB")
            .resize((size, size), Image.BILINEAR)
        )
        for i in indices
    ]


@functools.lru_cache(maxsize=320)
def _member_offsets(shard_path: str) -> dict[str, tuple[int, int]]:
    """member name -> (data_offset, size) for every file in `shard_path`.

    Scanning a tar's ~4k headers to open one member is the dominant cost of
    `read_shard_frames_at` under random (cross-shard) access -- ~285ms/window
    measured against a real 2.2GB shard, vs ~0.1ms once offsets are cached.
    Cached by path string (not Path, which isn't hashable-stable across
    equal-but-distinct instances) so repeated calls for the same shard reuse
    the scan. maxsize=320 covers the full 260-shard VoxCeleb2 corpus (with
    room to grow) so a globally-shuffled DataLoader doesn't evict shards it
    will revisit later in the same epoch -- a 64-slot cache against 260
    shards measured a 24% hit rate (matches 64/260) and only 18 samp/s
    single-process; each cached entry is small (~2.5MB, offsets only, not
    shard bytes), so 320 slots is ~800MB/worker, not an unbounded cache.
    """
    with tarfile.open(shard_path, "r") as tar:
        return {member.name: (member.offset_data, member.size) for member in tar}


def read_shard_frames_indexed(
    shard_path: Path, clip_key: str, indices: list[int], size: int = 224
) -> list[np.ndarray]:
    """Same contract as `read_shard_frames_at`, but seeks directly to the
    member's data offset instead of re-scanning the tar's headers -- see
    `_member_offsets`. Prefer this for workloads that read shards in random
    (non-shard-major) order, e.g. a globally-shuffled DataLoader."""
    offset, member_size = _member_offsets(str(shard_path))[f"{clip_key}.pickle"]
    with open(shard_path, "rb") as fh:
        fh.seek(offset)
        frame_jpegs: list[bytes] = pickle.loads(fh.read(member_size))
    return [
        np.array(
            Image.open(io.BytesIO(frame_jpegs[i]))
            .convert("RGB")
            .resize((size, size), Image.BILINEAR)
        )
        for i in indices
    ]
