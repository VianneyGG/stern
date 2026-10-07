from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from video_encoding.models.ViT.attention import SelfAttention
from video_encoding.models.ViT.droppath import DropPath
from video_encoding.models.ViT.layerscale import LayerScale
from video_encoding.models.ViT.mlp import build_mlp
from video_encoding.models.ViT.norm import build_norm


@dataclass
class BlockConfig:
    embed_dim: int
    num_heads: int
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    drop_path_rate: float = 0.0
    norm_type: str = "layernorm"
    mlp_type: str = "gelu"
    qk_norm: bool = False
    layer_scale_init: float | None = None
    use_skip: bool = True
    # LayerNorm Scaling (Curse of Depth, arXiv 2502.05795): multiplies both
    # pre-norm outputs by this; the encoder sets 1/sqrt(layer). 1.0 = off.
    ln_scale: float = 1.0


class ViTBlock(nn.Module):
    """Pre-norm ViT block: x + LayerScale(DropPath(Attn(norm(x)))) then
    x + LayerScale(DropPath(MLP(norm(x)))). Every sub-component is swappable
    via BlockConfig; BlockConfig's own field defaults reproduce the original
    nn.MultiheadAttention + LayerNorm + GELU block exactly (see
    test_vit_block.py's parity test)."""

    def __init__(self, cfg: BlockConfig):
        super().__init__()
        self.use_skip = cfg.use_skip
        self.ln_scale = cfg.ln_scale
        self.norm1 = build_norm(cfg.norm_type, cfg.embed_dim)
        self.attn = SelfAttention(
            cfg.embed_dim, cfg.num_heads, dropout=cfg.dropout, qk_norm=cfg.qk_norm
        )
        self.norm2 = build_norm(cfg.norm_type, cfg.embed_dim)
        self.mlp = build_mlp(cfg.mlp_type, cfg.embed_dim, cfg.mlp_ratio, cfg.dropout)
        self.drop_path = DropPath(cfg.drop_path_rate)
        self.layer_scale1 = (
            LayerScale(cfg.embed_dim, cfg.layer_scale_init)
            if cfg.layer_scale_init is not None
            else nn.Identity()
        )
        self.layer_scale2 = (
            LayerScale(cfg.embed_dim, cfg.layer_scale_init)
            if cfg.layer_scale_init is not None
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        rotary: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        h = self.norm1(x) * self.ln_scale
        attn_out, _ = self.attn(
            h, need_weights=False, attn_mask=attn_mask, rotary=rotary
        )
        attn_out = self.drop_path(self.layer_scale1(attn_out))
        x = x + attn_out if self.use_skip else attn_out
        mlp_out = self.drop_path(self.layer_scale2(self.mlp(self.norm2(x) * self.ln_scale)))
        x = x + mlp_out if self.use_skip else mlp_out
        return x
