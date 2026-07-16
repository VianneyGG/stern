"""From-scratch VideoMAE encoder + 2-D VA regression head, trained end-to-end.

No pretrained weights, no freezing, no LoRA: the whole model (the
randomly-initialised `VideoMAEEncoder` backbone + a `Linear(embed_dim, 2)` head)
receives gradients. Outputs (valence, arousal) in `[-1, 1]` via `tanh`, matching
the harmonised label range.
"""

from __future__ import annotations

import torch
from omegaconf import DictConfig
from torch import nn

from video_encoding.models.videomaev2.encoder import VideoMAEEncoder, pool_features


class LinearVAHead(nn.Module):
    """`Linear(embed_dim, 2)` -> `tanh`, mapping a pooled feature to
    (valence, arousal) in `[-1, 1]`."""

    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.linear = nn.Linear(embed_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear(x))


class VideoMAEVARegressor(nn.Module):
    def __init__(self, model_cfg: DictConfig, *, backbone: nn.Module | None = None):
        """`backbone` is injectable so tests can substitute a lightweight stub
        for the full video ViT. Production calls leave it None -> a fresh
        `VideoMAEEncoder` built from `model_cfg`."""
        super().__init__()
        self.backbone = backbone if backbone is not None else VideoMAEEncoder(model_cfg)
        self._embed_dim = int(model_cfg.embed_dim)
        self.head = LinearVAHead(embed_dim=self._embed_dim)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """`pixel_values`: `(B, C, T, H, W)` -> `(B, 2)` VA in `[-1, 1]`."""
        feat = pool_features(self.backbone(pixel_values=pixel_values))
        if feat.shape[-1] != self._embed_dim:
            raise ValueError(
                f"backbone feature dim {feat.shape[-1]} != configured "
                f"model.embed_dim {self._embed_dim}"
            )
        return self.head(feat)
