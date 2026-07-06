"""Frame windowing per REPO_SETUP.md D9 (window size 1-4s, multi-scale ablation).

Windowing is upstream of the SSL objective config: a 1s window leaves
S-JEPA-style latent prediction little to predict across, while 4s changes
masking ratios. D9 must be confirmed against Aff-Wild2 annotation
granularity before SSL hyperparameters are fixed (REPO_SETUP.md §4 D9).
"""

from __future__ import annotations

import numpy as np


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
