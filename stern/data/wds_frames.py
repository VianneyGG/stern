"""Read per-clip JPEG frame samples from WebDataset shards built by
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
    resized to size x size. Reopens the tar on every call. For random access
    across shards, use `read_shard_frames_indexed`."""
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
    """Map member name to (data_offset, size) for every file in `shard_path`.

    Walking a tar's headers is the dominant cost of random-access reads, so the
    result is cached. The key is the path string, since equal `Path` objects
    are not reliably hash-equal. maxsize covers the full 260-shard corpus, so a
    shuffled epoch does not evict shards it will revisit. Each entry holds
    offsets only, not shard bytes.
    """
    with tarfile.open(shard_path, "r") as tar:
        return {member.name: (member.offset_data, member.size) for member in tar}


def read_shard_frames_indexed(
    shard_path: Path, clip_key: str, indices: list[int], size: int = 224
) -> list[np.ndarray]:
    """Same contract as `read_shard_frames_at`, but seeks straight to the
    member's data offset via `_member_offsets`. Use this for random
    (non-shard-major) access, such as a shuffled DataLoader."""
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
