"""Raw-frame decoding for the VideoMAE pipeline (no face detection).

Per docs/superpowers/specs/2026-07-01-training-pipeline-design.md §4.2: this
intentionally skips face cropping (a shared face-crop stage was archived in
SCAFFOLD_ARCHIVE.md, out of scope until a RetinaFace integration is benchmarked)
and just uniformly samples whole frames from the raw clip.
"""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np
from PIL import Image


def decode_frames(video_path: Path, num_frames: int = 16) -> list[np.ndarray]:
    """Uniformly sample `num_frames` frames from `video_path`, resized to 224x224.

    Returns a list of `num_frames` uint8 arrays of shape (224, 224, 3).
    """
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        decoded = [
            frame.to_ndarray(format="rgb24") for frame in container.decode(stream)
        ]

    if not decoded:
        raise ValueError(f"no decodable frames in {video_path}")

    indices = np.linspace(0, len(decoded) - 1, num_frames).round().astype(int)
    return [_resize(decoded[i], 224) for i in indices]


def _resize(frame: np.ndarray, size: int) -> np.ndarray:
    return np.array(Image.fromarray(frame).resize((size, size), Image.BILINEAR))


# --- Frame-directory decoding (AFEW-VA ships extracted PNG frames, not MP4s) ---

_IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def list_frame_files(clip_dir: Path) -> list[Path]:
    """Frame-ordered image files in `clip_dir`.

    Sorts numerically by filename stem when every stem is an integer (VEATIC's
    `video_frame.py` writes unpadded `0.jpg, 1.jpg, ..., 10.jpg`, where a plain
    lexicographic sort would put "10.jpg" before "2.jpg"). Falls back to a
    lexicographic sort for non-integer stems (e.g. AFEW-VA's zero-padded names,
    where both orderings agree anyway).
    """
    files = [p for p in Path(clip_dir).iterdir() if p.suffix.lower() in _IMAGE_EXTS]
    if not files:
        raise ValueError(f"no image frames ({_IMAGE_EXTS}) in {clip_dir}")
    if all(p.stem.isdigit() for p in files):
        files.sort(key=lambda p: int(p.stem))
    else:
        files.sort()
    return files


def read_frames_at(
    frame_files: list[Path], indices: list[int], size: int = 224
) -> list[np.ndarray]:
    """Read the images at `indices` from `frame_files`, resized to `size`x`size`."""
    return [
        _resize(np.array(Image.open(frame_files[i]).convert("RGB")), size)
        for i in indices
    ]


def load_clip_frames(
    clip_path: Path, num_frames: int = 16, size: int = 224
) -> list[np.ndarray]:
    """Uniformly sample `num_frames` frames from `clip_path`, whichever layout
    it is: a video file (`decode_frames`) or a directory of extracted frame
    images (`list_frame_files` + `read_frames_at`, AFEW-VA/VEATIC's layout).

    Dispatches on `clip_path.is_dir()` so `data/loaders.py`'s `load_frames`
    methods work for both dataset shapes without callers needing to know
    which one a given dataset uses.
    """
    clip_path = Path(clip_path)
    if clip_path.is_dir():
        frame_files = list_frame_files(clip_path)
        indices = np.linspace(0, len(frame_files) - 1, num_frames).round().astype(int)
        return read_frames_at(frame_files, indices.tolist(), size=size)
    return decode_frames(clip_path, num_frames=num_frames)
