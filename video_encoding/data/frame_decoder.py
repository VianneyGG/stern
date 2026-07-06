"""Raw-frame decoding for the VideoMAEv2 pipeline (no face detection).

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
