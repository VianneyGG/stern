"""From-scratch VideoMAE-style video ViT encoder (no pretrained weights).

We deliberately do NOT load OpenGVLab/HF checkpoints: the encoder is built and
trained from random init, end-to-end with the VA head. The architecture mirrors
VideoMAE's encoder:

  (B, C, T, H, W)
    -> TubeletEmbed   : Conv3d over non-overlapping (tubelet_size, patch, patch) cubes
    -> + pos_embed    : learned positional embedding
    -> N x Block      : pre-norm MHSA + MLP transformer blocks
    -> LayerNorm
    -> mean over tokens
  -> (B, embed_dim)

Caveat (plan self-critique #1): a ~86M-param ViT trained from scratch on ~30k
AFEW-VA frames will overfit badly -- from-scratch video ViTs expect
Kinetics-scale data. This is an explicit design choice by the user, not an
oversight; expect a low CCC ceiling until more data (or pretraining) is added.
"""

from __future__ import annotations

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn

# Standard ImageNet normalisation (frames arrive as HxWx3 uint8).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class DropPath(nn.Module):
    """Per-sample stochastic depth: drops the whole residual branch for a
    random subset of samples in the batch, scaling survivors by 1/keep so
    the expected output is unchanged (standard inverted-dropout scaling)."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, dtype=x.dtype, device=x.device)
        mask = torch.floor(mask + keep) # binary mask: 1 with prob `keep`, 0 with prob `drop_prob`
        return x / keep * mask


class TubeletEmbed(nn.Module):
    """`(B, C, T, H, W)` -> `(B, N, embed_dim)` via a Conv3d over tubelets."""

    def __init__(
        self, *, in_channels: int, embed_dim: int, tubelet_size: int, patch_size: int
    ):
        super().__init__()
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, D, T', H', W')
        return x.flatten(2).transpose(1, 2)  # (B, N, D), N = T'*H'*W'


class Block(nn.Module):
    """Pre-norm ViT block: `x + DropPath(MHSA(norm(x)))` then `x + DropPath(MLP(norm(x)))`."""

    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.drop_path(self.attn(h, h, h, need_weights=False)[0])
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VideoMAEEncoder(nn.Module):
    """Randomly-initialised video ViT encoder (the "backbone")."""

    def __init__(self, model_cfg: DictConfig):
        super().__init__()
        self.num_frames = int(model_cfg.num_frames)
        self.image_size = int(model_cfg.image_size)
        self.embed_dim = int(model_cfg.embed_dim)
        tubelet_size = int(model_cfg.tubelet_size)
        patch_size = int(model_cfg.patch_size)

        self.patch_embed = TubeletEmbed(
            in_channels=int(model_cfg.in_channels),
            embed_dim=self.embed_dim,
            tubelet_size=tubelet_size,
            patch_size=patch_size,
        )
        n_tubelets = self.num_frames // tubelet_size
        grid = self.image_size // patch_size
        # Token grid (T', H', W') in sequence order. Tube masking needs this to
        # mask the same (H', W') positions across all T' temporal positions.
        self.token_grid = (n_tubelets, grid, grid)
        num_tokens = n_tubelets * grid * grid
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, self.embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02) # 

        dropout = float(model_cfg.get("dropout", 0.0))
        drop_path_rate = float(model_cfg.get("drop_path_rate", 0.0))
        depth = int(model_cfg.depth)
        dpr = (
            torch.linspace(0, drop_path_rate, depth).tolist()
            if depth > 1
            else [0.0] * depth
        )
        self.blocks = nn.ModuleList(
            Block(
                embed_dim=self.embed_dim,
                num_heads=int(model_cfg.num_heads),
                mlp_ratio=float(model_cfg.mlp_ratio),
                dropout=dropout,
                drop_path_rate=dpr[i],
            )
            for i in range(depth)
        )
        self.norm = nn.LayerNorm(self.embed_dim)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """`pixel_values`: `(B, C, T, H, W)` -> `(B, embed_dim)` clip feature."""
        x = self.patch_embed(pixel_values)
        if x.shape[1] != self.pos_embed.shape[1]:
            raise ValueError(
                f"got {x.shape[1]} tokens but pos_embed has "
                f"{self.pos_embed.shape[1]}; check num_frames/image_size vs "
                "tubelet_size/patch_size"
            )
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.norm(x).mean(dim=1)

    def forward_tokens(
        self, x: torch.Tensor, keep_indices: torch.Tensor
    ) -> torch.Tensor:
        """Encode a subset of tokens (used during MAE pretraining).

        Args:
            x: (B, N, D) embedded tokens (after patch_embed + pos_embed)
            keep_indices: (num_kept,) indices of tokens to keep (same for all batch)

        Returns:
            (B, num_kept, D) encoder output on visible tokens only
        """
        # Gather visible tokens from the batch
        x = x[:, keep_indices, :]  # (B, num_kept, D)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


def preprocess_window(
    frames: list[np.ndarray], num_frames: int, image_size: int
) -> torch.Tensor:
    """`frames`: list of `num_frames` HxWx3 uint8 arrays (already `image_size`).

    Returns normalised pixel_values `(1, C, T, H, W)` -- the layout the encoder
    consumes. Replaces the HF `VideoMAEImageProcessor` now that nothing is
    loaded from HF.
    """
    if len(frames) != num_frames:
        raise ValueError(f"expected {num_frames} frames, got {len(frames)}")
    arr = np.stack(frames).astype(np.float32) / 255.0  # (T, H, W, C)
    tensor = torch.from_numpy(arr).permute(3, 0, 1, 2)  # (C, T, H, W)
    mean = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1, 1)
    std = torch.tensor(_IMAGENET_STD).view(3, 1, 1, 1)
    return ((tensor - mean) / std).unsqueeze(0)


def pool_features(outputs: torch.Tensor) -> torch.Tensor:
    """Normalise a backbone output to `(B, D)`.

    The encoder already returns a pooled `(B, D)` feature; this stays as a thin
    seam so a stub backbone (tests) or a future token-sequence variant still
    composes with the regressor.
    """
    if outputs.ndim == 3:  # (B, N, D) token sequence -> mean-pool
        return outputs.mean(dim=1)
    if outputs.ndim == 2:  # (B, D) already pooled
        return outputs
    raise ValueError(f"unexpected backbone feature shape {tuple(outputs.shape)}")
