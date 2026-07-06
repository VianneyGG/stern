"""VA head + CCC-based training loop
(docs/superpowers/specs/2026-07-01-training-pipeline-design.md §4.5).

`ccc_loss` reimplements common.eval.ccc's formula with torch ops so autograd
works (the numpy version is correct for eval but not differentiable) --
common.eval.ccc.concordance_correlation_coefficient stays the eval metric.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from eval.ccc import concordance_correlation_coefficient
from video_encoding.training.logger import make_writer


class LinearVAHead(nn.Module):
    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.linear = nn.Linear(embed_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear(x))


def _torch_ccc(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mean_x, mean_y = x.mean(), y.mean()
    var_x, var_y = x.var(unbiased=False), y.var(unbiased=False)
    covar = ((x - mean_x) * (y - mean_y)).mean()
    denom = var_x + var_y + (mean_x - mean_y) ** 2
    return (2 * covar) / denom


def ccc_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """-(CCC_valence + CCC_arousal) / 2 -- maximize concordance by minimizing this."""
    ccc_v = _torch_ccc(pred[:, 0], target[:, 0])
    ccc_a = _torch_ccc(pred[:, 1], target[:, 1])
    return -(ccc_v + ccc_a) / 2


def train_probe(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    val_embeddings: np.ndarray,
    val_labels: np.ndarray,
    *,
    embed_dim: int = 768,
    epochs: int = 50,
    lr: float = 1e-3,
    log_dir: Path | None = None,
) -> dict:
    head = LinearVAHead(embed_dim=embed_dim)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    train_x = torch.from_numpy(train_embeddings).float()
    train_y = torch.from_numpy(train_labels).float()

    writer = make_writer(log_dir)
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = head(train_x)
        loss = ccc_loss(pred, train_y)
        loss.backward()
        optimizer.step()
        if writer is not None:
            writer.add_scalar("train/loss", loss.item(), epoch)

    with torch.no_grad():
        val_pred = head(torch.from_numpy(val_embeddings).float()).numpy()

    val_ccc_v = concordance_correlation_coefficient(val_pred[:, 0], val_labels[:, 0])
    val_ccc_a = concordance_correlation_coefficient(val_pred[:, 1], val_labels[:, 1])

    if writer is not None:
        writer.close()

    return {"val_ccc_v": val_ccc_v, "val_ccc_a": val_ccc_a, "head": head}
