# common/cache/store.py
"""FP16 HDF5 embedding cache (docs/superpowers/specs/2026-07-01-training-pipeline-design.md §4.3).

Recipe A/C: compute frozen embeddings once, write here, all probe training
reads tensors from this cache. Recipe B: reads cached crops (not embeddings,
the encoder is changing), recomputes live during adaptation, then caches the
adapted encoder's embeddings once after convergence.

One HDF5 file holds the whole dataset, one dataset per clip keyed by
`clip_key(clip_path)`.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def clip_key(clip_path: str) -> str:
    """"train/id00000/vid001/001.mp4" -> "train__id00000__vid001__001"."""
    normalized = clip_path.replace("\\", "/")
    stem = normalized.rsplit(".", 1)[0]
    return stem.replace("/", "__")


def write_embeddings(path: Path, key: str, embeddings: np.ndarray) -> None:
    """Append `embeddings` (cast to FP16) under `key` in the HDF5 file at `path`."""
    with h5py.File(path, "a") as f:
        if key in f:
            del f[key]
        f.create_dataset(key, data=embeddings.astype(np.float16))


def read_embeddings(path: Path, key: str) -> np.ndarray:
    """Read the FP16 embeddings stored under `key`, upcast to float32."""
    with h5py.File(path, "r") as f:
        return f[key][...].astype(np.float32)
