"""Frame windowing per REPO_SETUP.md D9 (window size 1-4s, multi-scale ablation).

Windowing is upstream of the SSL objective config: a 1s window leaves
S-JEPA-style latent prediction little to predict across, while 4s changes
masking ratios. D9 must be confirmed against Aff-Wild2 annotation
granularity before SSL hyperparameters are fixed (REPO_SETUP.md §4 D9).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from video_encoding.data.frame_decoder import list_frame_files


def make_windows(
    frames: np.ndarray, fps: float, len_s: float, overlap: float
) -> list[np.ndarray]:
    """Slice `frames` into windows of `len_s` seconds with `overlap` fraction.

    Trailing partial windows are dropped (every window has identical length).
    An empty `frames` array yields an empty window list (not an error) since
    "no data" is a valid upstream state, not a usage mistake.
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


def window_starts(n_frames: int, window_len: int, overlap: float) -> list[int]:
    """Start indices of `window_len`-long windows over `n_frames` with `overlap`.

    Frame-index analogue of `make_windows` (which slices arrays): used to align
    a fixed `num_frames` model window with its label rows without decoding the
    whole clip into memory first. Trailing partial windows are dropped.
    """
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    if window_len <= 0:
        raise ValueError(f"window_len must be positive, got {window_len}")
    step = int(round(window_len * (1 - overlap)))
    if step <= 0:
        raise ValueError(f"overlap={overlap} produced a non-positive step")
    return list(range(0, n_frames - window_len + 1, step))


def build_frame_windows(
    clip_dir: Path,
    *,
    num_frames: int,
    overlap: float,
    n_frames: int | None = None,
) -> list[tuple[list[Path], list[int]]]:
    """List `clip_dir`'s frame files and build `(frame_files, idx)` windows.

    Shared by SSLWindowDataset and the VA-probe window dataset -- both need
    the same `list_frame_files` -> `window_starts` -> idx-list construction;
    only what happens with `idx` after (mean VA label vs. nothing) differs.

    `n_frames` overrides the frame count used for `window_starts` (the probe
    dataset clamps to `min(len(frame_files), va.shape[0])` so windows never
    run past the label array); defaults to `len(frame_files)`.
    """
    frame_files = list_frame_files(clip_dir)
    n = len(frame_files) if n_frames is None else n_frames
    windows = []
    for start in window_starts(n, num_frames, overlap):
        idx = list(range(start, start + num_frames))
        windows.append((frame_files, idx))
    return windows
