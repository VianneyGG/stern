"""Concordance correlation coefficient (Lin, 1989): the headline VA metric and the training loss.

CCC(x, y) = 2 * cov(x, y) / (var(x) + var(y) + (mean(x) - mean(y))^2),

with population (1/N) moments. Unlike Pearson r, CCC penalises both shape
mismatch (low correlation) and location/scale mismatch (bias). ABAW scores VA
with it.

`ccc_loss` computes the same formula with torch ops so autograd works. The
numpy function is the eval metric; the torch version is only the training loss.
"""

from __future__ import annotations

import numpy as np
import torch


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
        # Both signals are constant. allclose, not ==, absorbs float noise in
        # signals that are constant in principle but not bit-identical.
        return 1.0 if np.allclose(x, y) else 0.0
    return float((2 * covar) / denom)


def _torch_ccc(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mean_x, mean_y = x.mean(), y.mean()
    var_x, var_y = x.var(unbiased=False), y.var(unbiased=False)
    covar = ((x - mean_x) * (y - mean_y)).mean()
    denom = var_x + var_y + (mean_x - mean_y) ** 2
    return (2 * covar) / denom


def ccc_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """`-(CCC_valence + CCC_arousal) / 2`, so minimising it maximises concordance.

    CCC is a batch-level statistic, so `pred` and `target` are `(B, 2)` and `B`
    must be large enough to estimate variance.
    """
    ccc_v = _torch_ccc(pred[:, 0], target[:, 0])
    ccc_a = _torch_ccc(pred[:, 1], target[:, 1])
    return -(ccc_v + ccc_a) / 2
