"""Concordance correlation coefficient (Lin, 1989) — ABAW-standard VA metric.

CCC(x, y) = 2 * cov(x, y) / (var(x) + var(y) + (mean(x) - mean(y))^2)

Unlike Pearson r, CCC penalizes both shape mismatch (correlation) and
location/scale mismatch (bias), which is why ABAW uses it for VA scoring.
"""

from __future__ import annotations

import numpy as np


def concordance_correlation_coefficient(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")
    if x.ndim != 1:
        raise ValueError(f"expected 1D arrays, got shape {x.shape}")
    if x.size == 0:
        raise ValueError("x and y must be non-empty")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("x and y must be finite (no NaN/inf)")
    mean_x, mean_y = x.mean(), y.mean()
    var_x, var_y = x.var(), y.var()
    covar = np.mean((x - mean_x) * (y - mean_y))
    denom = var_x + var_y + (mean_x - mean_y) ** 2
    if denom == 0:
        # Both signals are constant (zero variance) and means coincide.
        # np.allclose (not ==) tolerates float noise in upstream signals
        # that are constant in principle but not bit-identical in practice.
        return 1.0 if np.allclose(x, y) else 0.0
    return float((2 * covar) / denom)
