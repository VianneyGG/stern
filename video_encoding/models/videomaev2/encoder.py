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

import math
from collections.abc import Sequence

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn

from video_encoding.models.ViT.block import BlockConfig, ViTBlock
from video_encoding.models.ViT.init import (
    apply_cutting_skip_to_attention,
    apply_weight_init,
)
from video_encoding.models.ViT.patch_embed import TubeletEmbed
from video_encoding.models.ViT.registers import RegisterTokens

# Standard ImageNet normalisation (frames arrive as HxWx3 uint8).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class VideoMAEEncoder(nn.Module):
    """Randomly-initialised video ViT encoder (the "backbone")."""

    def __init__(self, model_cfg: DictConfig):
        super().__init__()
        self.model_cfg = model_cfg
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
            norm=bool(model_cfg.get("embed_norm", False)),
        )
        n_tubelets = self.num_frames // tubelet_size
        grid = self.image_size // patch_size
        # Token grid (T', H', W') in sequence order. Tube masking needs this to
        # mask the same (H', W') positions across all T' temporal positions.
        self.token_grid = (n_tubelets, grid, grid)
        num_tokens = n_tubelets * grid * grid
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, self.embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)  #

        dropout = float(model_cfg.get("dropout", 0.0))
        drop_path_rate = float(model_cfg.get("drop_path_rate", 0.0))
        depth = int(model_cfg.depth)
        dpr = (
            torch.linspace(0, drop_path_rate, depth).tolist()
            if depth > 1
            else [0.0] * depth
        )

        norm_type = str(model_cfg.get("norm_type", "layernorm"))
        mlp_type = str(model_cfg.get("mlp_type", "gelu"))
        qk_norm = bool(model_cfg.get("qk_norm", False))
        layer_scale_init = model_cfg.get("layer_scale_init", None)
        layer_scale_init = (
            float(layer_scale_init) if layer_scale_init is not None else None
        )
        layernorm_scaling = bool(model_cfg.get("layernorm_scaling", False))
        num_registers = int(model_cfg.get("num_registers", 0))
        self.registers = RegisterTokens(self.embed_dim, num_registers)

        self.blocks = nn.ModuleList(
            ViTBlock(
                BlockConfig(
                    embed_dim=self.embed_dim,
                    num_heads=int(model_cfg.num_heads),
                    mlp_ratio=float(model_cfg.mlp_ratio),
                    dropout=dropout,
                    drop_path_rate=dpr[i],
                    norm_type=norm_type,
                    mlp_type=mlp_type,
                    qk_norm=qk_norm,
                    layer_scale_init=layer_scale_init,
                    ln_scale=1 / math.sqrt(i + 1) if layernorm_scaling else 1.0,
                )
            )
            for i in range(depth)
        )
        self.norm = nn.LayerNorm(self.embed_dim)
        # Pre-rewrite, attention's in_proj/out_proj weights came from
        # nn.MultiheadAttention's own default init (xavier_uniform_ on the raw
        # in_proj_weight Parameter, which self.apply() never reached since it
        # wasn't wrapped in an nn.Linear). SelfAttention's qkv/proj ARE plain
        # nn.Linear, so they now get the configured init below. This keeps the
        # default as trunc_normal(std=0.02) while enabling the paper's
        # Cutting-the-Skip diagonal-dominant init as an explicit ablation axis.
        self.apply(self._init_weights)
        self._init_attention_weights()

    def _init_weights(self, module: nn.Module) -> None:
        method = str(self.model_cfg.get("weight_init", "trunc_normal"))
        std = float(self.model_cfg.get("weight_init_std", 0.02))

        if isinstance(module, nn.Linear):
            apply_weight_init(module, method=method, std=std)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _init_attention_weights(self) -> None:
        """Cutting-the-Skip couples Q with K and V with O and needs the whole
        attention module at once (see apply_cutting_skip_to_attention), so it
        runs here as a second pass over self.blocks rather than through the
        per-nn.Linear self.apply() walk above, which cannot see that pairing."""
        if str(self.model_cfg.get("weight_init", "trunc_normal")) != "cutting_skip":
            return
        alpha = float(self.model_cfg.get("weight_init_alpha", 1.8))
        beta = float(self.model_cfg.get("weight_init_beta", 1.0))
        c = float(self.model_cfg.get("weight_init_c", 3.0))
        for block in self.blocks:
            apply_cutting_skip_to_attention(block.attn, alpha=alpha, beta=beta, c=c)

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
        x = self.registers.prepend(x)
        for block in self.blocks:
            x = block(x)
        x = self.registers.strip(x)
        return self.norm(x).mean(dim=1)

    def forward_tokens(
        self,
        x: torch.Tensor,
        keep_indices: torch.Tensor,
        return_layers: Sequence[int] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Encode a subset of tokens (used during MAE pretraining).

        Args:
            x: (B, N, D) embedded tokens (after patch_embed + pos_embed)
            keep_indices: (B, num_kept) per-sample indices of tokens to keep
            return_layers: optional 1-based block indices whose outputs to also
                return, for deep supervision. Each is register-stripped and
                passed through the *same* `self.norm` as the final output, so a
                shared decoder can consume any of them interchangeably.

        Returns:
            (B, num_kept, D) encoder output on visible tokens only, or
            `(output, intermediates)` when `return_layers` is given. The default
            return type is a bare tensor -- existing callers unpack it directly.
        """
        # keep_indices is (B, num_kept): gather per-sample, not one shared
        # column selection for the whole batch.
        idx = keep_indices.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        x = torch.gather(x, 1, idx)  # (B, num_kept, D)
        x = self.registers.prepend(x)

        if return_layers is None:
            for block in self.blocks:
                x = block(x)
            x = self.registers.strip(x)
            return self.norm(x)

        wanted = set(return_layers)
        intermediates: dict[int, torch.Tensor] = {}
        for i, block in enumerate(self.blocks, start=1):
            x = block(x)
            if i in wanted:
                intermediates[i] = self.norm(self.registers.strip(x))
        x = self.registers.strip(x)
        out = self.norm(x)
        # Ordered by the caller's request, not by block index.
        return out, [intermediates[i] for i in return_layers]


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
