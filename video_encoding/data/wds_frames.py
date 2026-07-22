# video_encoding/data/wds_frames.py
"""Read per-clip JPEG-frame samples from WebDataset shards built by
video_encoding/scripts/build_wds_shards.py."""

from __future__ import annotations

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
