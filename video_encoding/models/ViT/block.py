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


BLOCK_DEFAULTS = dict(
    norm_type="layernorm", mlp_type="gelu", qk_norm=False, layer_scale_init=None
)


class ViTBlock(nn.Module):
    """Pre-norm ViT block: x + LayerScale(DropPath(Attn(norm(x)))) then
    x + LayerScale(DropPath(MLP(norm(x)))). Every sub-component is swappable
    via BlockConfig; BLOCK_DEFAULTS reproduces the original
    nn.MultiheadAttention + LayerNorm + GELU block exactly (see
    test_vit_block.py's parity test)."""

    def __init__(self, cfg: BlockConfig):
        super().__init__()
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, need_weights=False)
        x = x + self.drop_path(self.layer_scale1(attn_out))
        x = x + self.drop_path(self.layer_scale2(self.mlp(self.norm2(x))))
        return x
