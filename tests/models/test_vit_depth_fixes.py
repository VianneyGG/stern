"""Phase 2 depth-collapse knobs: LayerNorm Scaling (J2) and patch-embed LayerNorm (J8)."""

import math

import pytest
import torch
from omegaconf import OmegaConf

from stern.models.videomae.encoder import VideoMAEEncoder
from stern.models.vit.block import BlockConfig, ViTBlock
from stern.models.vit.patch_embed import TubeletEmbed


def _cfg(**kw):
    base = OmegaConf.load("configs/model/videomae_base.yaml")
    base.depth, base.embed_dim, base.num_heads = 4, 64, 4
    base.num_frames, base.image_size = 4, 32
    return OmegaConf.merge(base, kw)


def test_block_ln_scale_scales_both_prenorm_outputs():
    torch.manual_seed(0)
    blk = ViTBlock(BlockConfig(embed_dim=16, num_heads=2, ln_scale=0.5))
    x = torch.randn(2, 5, 16)
    h = blk.norm1(x) * 0.5
    x1 = x + blk.attn(h, need_weights=False)[0]
    want = x1 + blk.mlp(blk.norm2(x1) * 0.5)
    torch.testing.assert_close(blk(x), want)


def test_block_ln_scale_default_is_unscaled():
    torch.manual_seed(0)
    a = ViTBlock(BlockConfig(embed_dim=16, num_heads=2))
    b = ViTBlock(BlockConfig(embed_dim=16, num_heads=2, ln_scale=0.5))
    b.load_state_dict(a.state_dict())
    x = torch.randn(2, 5, 16)
    assert a.ln_scale == 1.0
    assert not torch.allclose(a(x), b(x))


def test_encoder_layernorm_scaling_sets_inverse_sqrt_depth():
    enc = VideoMAEEncoder(_cfg(layernorm_scaling=True))
    assert [b.ln_scale for b in enc.blocks] == pytest.approx(
        [1.0, 1 / math.sqrt(2), 1 / math.sqrt(3), 0.5]
    )
    assert [b.ln_scale for b in VideoMAEEncoder(_cfg()).blocks] == [1.0] * 4


def test_embed_norm_gives_unit_scale_tokens_and_no_params_when_off():
    torch.manual_seed(0)
    emb = TubeletEmbed(
        in_channels=3, embed_dim=64, tubelet_size=2, patch_size=16, norm=True
    )
    tok = emb(torch.randn(2, 3, 4, 32, 32) * 5)
    assert tok.shape == (2, 8, 64)
    torch.testing.assert_close(tok.mean(-1), torch.zeros(2, 8), atol=1e-5, rtol=0)
    torch.testing.assert_close(
        tok.std(-1, unbiased=False), torch.ones(2, 8), atol=1e-3, rtol=0
    )
    off = TubeletEmbed(in_channels=3, embed_dim=64, tubelet_size=2, patch_size=16)
    assert sorted(k for k, _ in off.named_parameters()) == ["proj.bias", "proj.weight"]


def test_encoder_default_state_dict_unchanged_by_new_flags():
    # Existing checkpoints must still load: flags off add no parameters.
    keys = set(VideoMAEEncoder(_cfg()).state_dict())
    assert not any(k.startswith("patch_embed.norm") for k in keys)
    on = set(VideoMAEEncoder(_cfg(embed_norm=True)).state_dict())
    assert on - keys == {"patch_embed.norm.weight", "patch_embed.norm.bias"}
