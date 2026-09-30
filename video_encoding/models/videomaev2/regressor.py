"""VideoMAE encoder + frozen-backbone attentive VA head.

The regressor consumes a pretrained video encoder and runs a learnable query
cross-attention over the token sequence before a final linear projection to the
2-D VA output. The attention pooler is intentionally compact: 12 heads, one
learnable query vector, key/value/output matrices, and a final unembed layer.

The backbone stays compatible with the existing test stubs: if a backbone emits a
single pooled embedding `(B, D)`, it is treated as a 1-token sequence. The real
`VideoMAEEncoder` emits `(B, N, D)` tokens and the regressor attends over all of
those tokens before predicting valence/arousal.
"""

from __future__ import annotations

import math

import torch
from omegaconf import DictConfig
from torch import nn

from video_encoding.models.videomaev2.encoder import VideoMAEEncoder


class LinearVAHead(nn.Module):
    """Cross-attention probe head for token sequences.

    Accepts either `(B, D)` or `(B, N, D)` and returns `(B, 2)` VA values in
    `[-1, 1]` after a tanh activation. With `bounded=False` the training
    output is linear and only the eval output is clamped to `[-1, 1]`: a tanh
    that saturates zeroes its own gradient (the ssl-diag-03 probe on AFEW-VA).
    """

    def __init__(self, embed_dim: int = 768, num_heads: int = 12, bounded: bool = True):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}"
            )
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.bounded = bool(bounded)

        self.query = nn.Parameter(torch.zeros(1, self.embed_dim))
        nn.init.xavier_uniform_(self.query)

        self.key = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.value = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.output = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.unembed = nn.Linear(self.embed_dim, 2)

    def _attend(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-head cross-attention from a learned query to token set x."""
        batch_size, num_tokens, _ = x.shape
        q = self.query.expand(batch_size, -1).view(
            batch_size, self.num_heads, self.head_dim
        )
        q = q.unsqueeze(2)  # (B, H, 1, D_h)

        k = self.key(x).view(batch_size, num_tokens, self.num_heads, self.head_dim)
        v = self.value(x).view(batch_size, num_tokens, self.num_heads, self.head_dim)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, v).reshape(batch_size, self.embed_dim)
        return self.output(context)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        elif x.ndim != 3:
            raise ValueError(
                f"expected token tensor of shape (B, N, D) or (B, D), got {tuple(x.shape)}"
            )

        out = self.unembed(self._attend(x))
        if self.bounded:
            return torch.tanh(out)
        return out if self.training else out.clamp(-1.0, 1.0)


class VideoMAEVARegressor(nn.Module):
    def __init__(self, model_cfg: DictConfig, *, backbone: nn.Module | None = None):
        """`backbone` is injectable so tests can substitute a lightweight stub
        for the full video ViT. Production calls leave it None -> a fresh
        `VideoMAEEncoder` built from `model_cfg`."""
        super().__init__()
        self.backbone = backbone if backbone is not None else VideoMAEEncoder(model_cfg)
        self._embed_dim = int(model_cfg.embed_dim)
        self.head = LinearVAHead(
            embed_dim=self._embed_dim,
            num_heads=12,
            bounded=bool(model_cfg.get("head_bounded", True)),
        )

    def freeze_backbone(self) -> None:
        """Freeze the pretrained encoder while leaving the cross-attention VA head trainable."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _backbone_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return token features `(B, N, D)` from the backbone when possible."""
        if hasattr(self.backbone, "patch_embed") and hasattr(
            self.backbone, "pos_embed"
        ):
            x = self.backbone.patch_embed(pixel_values)
            if x.shape[1] != self.backbone.pos_embed.shape[1]:
                raise ValueError(
                    f"got {x.shape[1]} tokens but pos_embed has "
                    f"{self.backbone.pos_embed.shape[1]}; check num_frames/image_size vs "
                    "tubelet_size/patch_size"
                )
            x = x + self.backbone.pos_embed
            x = self.backbone.registers.prepend(x)
            for block in self.backbone.blocks:
                x = block(x)
            x = self.backbone.registers.strip(x)
            return self.backbone.norm(x)

        feat = self.backbone(pixel_values=pixel_values)
        if feat.ndim == 3:
            return feat
        if feat.ndim == 2:
            return feat.unsqueeze(1)
        raise ValueError(f"unexpected backbone feature shape {tuple(feat.shape)}")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """`pixel_values`: `(B, C, T, H, W)` -> `(B, 2)` VA in `[-1, 1]`."""
        tokens = self._backbone_tokens(pixel_values)
        if tokens.shape[-1] != self._embed_dim:
            raise ValueError(
                f"backbone feature dim {tokens.shape[-1]} != configured "
                f"model.embed_dim {self._embed_dim}"
            )
        return self.head(tokens)
