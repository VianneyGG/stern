"""VA range and annotation-rate harmonisation.

Every dataset is coerced to the [-1, 1] VA range and a single annotation rate
before any cross-corpus CCC comparison. Otherwise the comparison measures the
harmonisation mismatch instead of the domain shift.
"""

from __future__ import annotations

import numpy as np

TARGET_VA_RANGE: tuple[float, float] = (-1.0, 1.0)


def harmonise_va(values: np.ndarray, source_range: tuple[float, float]) -> np.ndarray:
    """Rescale VA values from `source_range` to the project-wide [-1, 1] convention."""
    lo, hi = source_range
    if hi <= lo:
        raise ValueError(f"source_range must have hi > lo, got {source_range}")
    if not np.all(np.isfinite(values)):
        raise ValueError("values must be finite (no NaN/inf)")
    if np.any(values < lo) or np.any(values > hi):
        raise ValueError(f"values fall outside declared source_range {source_range}")
    target_lo, target_hi = TARGET_VA_RANGE
    normalised = (values - lo) / (hi - lo)  # -> [0, 1]
    return normalised * (target_hi - target_lo) + target_lo


def harmonise_rate(
    signal: np.ndarray, source_hz: float, target_hz: float
) -> np.ndarray:
    """Resample a 1D signal from source_hz to target_hz via linear interpolation."""
    if signal.shape[0] == 0:
        raise ValueError("signal must be non-empty")
    if source_hz <= 0 or target_hz <= 0:
        raise ValueError("rates must be positive")
    if source_hz == target_hz:
        return signal
    n_source = signal.shape[0]
    n_target = int(round(n_source * target_hz / source_hz))
    if n_target == 0:
        raise ValueError(
            f"resampling produced zero target samples "
            f"(n_source={n_source}, source_hz={source_hz}, target_hz={target_hz})"
        )
    source_t = np.arange(n_source) / source_hz
    target_t = np.arange(n_target) / target_hz
    # Rounding n_target can push the last target_t slightly past source_t's end.
    # np.interp clamps those samples to the endpoint, which is fine here.
    return np.interp(target_t, source_t, signal)
