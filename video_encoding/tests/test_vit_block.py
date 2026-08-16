from __future__ import annotations

import pytest
import torch
from torch import nn

from video_encoding.models.ViT.block import BlockConfig, ViTBlock
from video_encoding.models.ViT.layerscale import LayerScale


class _LegacyBlockForParityTest(nn.Module):
    """Faithful copy of the pre-rewrite ViTBlock (nn.MultiheadAttention +
    LayerNorm + GELU), kept ONLY as a test fixture -- the production version
    of this class is deleted from video_encoding/models/ViT/vit.py in this
    same task. This is the reference the new composed ViTBlock is proven
    against before that deletion."""

    def __init__(self, *, embed_dim, num_heads, mlp_ratio, dropout):
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

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


def test_block_defaults_match_legacy_block_given_same_weights():
    torch.manual_seed(0)
    embed_dim, num_heads, mlp_ratio, seq_len, batch = 16, 4, 2.0, 5, 2

    legacy = _LegacyBlockForParityTest(
        embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=0.0
    )
    new = ViTBlock(
        BlockConfig(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
    )

    with torch.no_grad():
        new.attn.qkv.weight.copy_(legacy.attn.in_proj_weight)
        new.attn.qkv.bias.copy_(legacy.attn.in_proj_bias)
        new.attn.proj.weight.copy_(legacy.attn.out_proj.weight)
        new.attn.proj.bias.copy_(legacy.attn.out_proj.bias)
        new.norm1.weight.copy_(legacy.norm1.weight)
        new.norm1.bias.copy_(legacy.norm1.bias)
        new.norm2.weight.copy_(legacy.norm2.weight)
        new.norm2.bias.copy_(legacy.norm2.bias)
        new.mlp.net[0].weight.copy_(legacy.mlp[0].weight)
        new.mlp.net[0].bias.copy_(legacy.mlp[0].bias)
        new.mlp.net[3].weight.copy_(legacy.mlp[3].weight)
        new.mlp.net[3].bias.copy_(legacy.mlp[3].bias)

    x = torch.randn(batch, seq_len, embed_dim)
    legacy.eval()
    new.eval()
    with torch.no_grad():
        legacy_out = legacy(x)
        new_out = new(x)

    assert torch.allclose(legacy_out, new_out, atol=1e-5)


def test_block_defaults_use_identity_layer_scale():
    block = ViTBlock(BlockConfig(embed_dim=16, num_heads=4))
    assert isinstance(block.layer_scale1, nn.Identity)
    assert isinstance(block.layer_scale2, nn.Identity)


@pytest.mark.parametrize("norm_type", ["layernorm", "rmsnorm", "centerednorm"])
def test_block_runs_with_each_norm_type(norm_type):
    block = ViTBlock(BlockConfig(embed_dim=16, num_heads=4, norm_type=norm_type))
    out = block(torch.randn(2, 5, 16))
    assert out.shape == (2, 5, 16)


@pytest.mark.parametrize("mlp_type", ["gelu", "swiglu"])
def test_block_runs_with_each_mlp_type(mlp_type):
    block = ViTBlock(BlockConfig(embed_dim=16, num_heads=4, mlp_type=mlp_type))
    out = block(torch.randn(2, 5, 16))
    assert out.shape == (2, 5, 16)


def test_block_runs_with_qk_norm():
    block = ViTBlock(BlockConfig(embed_dim=16, num_heads=4, qk_norm=True))
    out = block(torch.randn(2, 5, 16))
    assert out.shape == (2, 5, 16)


def test_block_runs_with_layer_scale():
    block = ViTBlock(BlockConfig(embed_dim=16, num_heads=4, layer_scale_init=1e-5))
    assert isinstance(block.layer_scale1, LayerScale)
    out = block(torch.randn(2, 5, 16))
    assert out.shape == (2, 5, 16)
