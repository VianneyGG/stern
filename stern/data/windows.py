"""Fixed-length frame windows for the SSL and probe datasets.

Window length (1-4 s, under multi-scale ablation) sits upstream of the SSL
objective. A 1 s window leaves latent prediction little to predict across, and
4 s changes the effective masking ratio. Confirm it against the annotation
granularity before fixing SSL hyperparameters.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from stern.data.frame_decoder import list_frame_files


def make_windows(
    frames: np.ndarray, fps: float, len_s: float, overlap: float
) -> list[np.ndarray]:
    """Slice `frames` into `len_s`-second windows with `overlap` fraction.

    Trailing partial windows are dropped. An empty `frames` array yields no
    windows and is not an error, since empty input is a valid upstream state.
    """
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if len_s <= 0:
        raise ValueError(f"len_s must be positive, got {len_s}")
    window_len = int(round(len_s * fps))
    if window_len <= 0:
        raise ValueError(f"len_s={len_s} at fps={fps} produced a zero-length window")
    step = int(round(window_len * (1 - overlap)))
    if step <= 0:
        raise ValueError(f"overlap={overlap} produced a non-positive step")
    n_frames = frames.shape[0]
    windows = []
    start = 0
    while start + window_len <= n_frames:
        windows.append(frames[start : start + window_len])
        start += step
    return windows


def window_starts(
    n_frames: int, window_len: int, overlap: float, frame_stride: int = 1
) -> list[int]:
    """Start indices of `window_len`-frame windows over `n_frames`.

    The index-level twin of `make_windows`. It aligns a model window with its
    label rows without decoding the clip first. Trailing partial windows are
    dropped.

    With `frame_stride` > 1, a window spans `(window_len - 1) * frame_stride + 1`
    frames but still feeds the model `window_len` of them. Overlap and the
    trailing bound use that span, or strided windows overlap at `overlap=0`.
    """
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    if window_len <= 0:
        raise ValueError(f"window_len must be positive, got {window_len}")
    if frame_stride <= 0:
        raise ValueError(f"frame_stride must be positive, got {frame_stride}")
    span = (window_len - 1) * frame_stride + 1
    step = int(round(span * (1 - overlap)))
    if step <= 0:
        raise ValueError(f"overlap={overlap} produced a non-positive step")
    return list(range(0, n_frames - span + 1, step))


def build_frame_windows(
    clip_dir: Path,
    *,
    num_frames: int,
    overlap: float,
    n_frames: int | None = None,
    frame_stride: int = 1,
) -> list[tuple[list[Path], list[int]]]:
    """List `clip_dir`'s frames and build `(frame_files, idx)` windows.

    Shared by SSLWindowDataset and VAWindowDataset. They differ only in what
    they do with `idx` afterwards.

    `n_frames` caps the frame count used for windowing. The probe passes
    `min(len(frame_files), va.shape[0])` so no window runs past its labels.
    """
    frame_files = list_frame_files(clip_dir)
    n = len(frame_files) if n_frames is None else n_frames
    windows = []
    for start in window_starts(n, num_frames, overlap, frame_stride):
        idx = list(range(start, start + num_frames * frame_stride, frame_stride))
        windows.append((frame_files, idx))
    return windows
