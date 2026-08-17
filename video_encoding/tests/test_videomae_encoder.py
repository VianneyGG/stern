"""From-scratch VideoMAEEncoder: shape, gradient flow, and the token guard.

Uses a tiny ViT config so a real forward/backward runs on CPU in CI -- this is
the real-architecture coverage the HF-loading design could not have (loading
those weights needed network + a GPU).
"""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from video_encoding.models.videomaev2.encoder import VideoMAEEncoder, preprocess_window
from video_encoding.models import VideoMAEVARegressor


def _tiny_cfg(**overrides):
    cfg = {
        "num_frames": 4,
        "image_size": 16,
        "in_channels": 3,
        "tubelet_size": 2,
        "patch_size": 8,
        "embed_dim": 32,
        "depth": 2,
        "num_heads": 4,
        "mlp_ratio": 2.0,
        "dropout": 0.0,
    }
    cfg.update(overrides)
    return OmegaConf.create(cfg)


def test_encoder_returns_pooled_feature_and_flows_grad():
    encoder = VideoMAEEncoder(_tiny_cfg())
    x = torch.randn(2, 3, 4, 16, 16)

    feat = encoder(x)

    assert feat.shape == (2, 32)  # tokens = 2 tubelets * 2*2 grid = 8, pooled away
    feat.sum().backward()
    assert encoder.patch_embed.proj.weight.grad is not None  # random init is trainable


def test_encoder_rejects_wrong_input_geometry():
    encoder = VideoMAEEncoder(_tiny_cfg())
    # 8 frames instead of 4 -> 16 tokens vs pos_embed's 8.
    with pytest.raises(ValueError, match="tokens"):
        encoder(torch.randn(1, 3, 8, 16, 16))


def test_preprocess_window_shape_and_normalisation():
    frames = [torch.zeros(16, 16, 3, dtype=torch.uint8).numpy() for _ in range(4)]

    pv = preprocess_window(frames, num_frames=4, image_size=16)

    assert pv.shape == (1, 3, 4, 16, 16)  # (B, C, T, H, W)
    # zeros -> (0 - mean)/std, i.e. all-negative per-channel constants.
    assert torch.all(pv < 0)


def test_regressor_on_real_encoder_returns_bounded_va():
    model = VideoMAEVARegressor(_tiny_cfg())
    out = model(torch.randn(3, 3, 4, 16, 16))

    assert out.shape == (3, 2)
    assert torch.all(out >= -1.0) and torch.all(out <= 1.0)


def test_drop_path_zeroes_and_scales_samples():
    from video_encoding.models.ViT.droppath import DropPath

    torch.manual_seed(0)
    dp = DropPath(drop_prob=0.5)
    dp.train()
    x = torch.ones(200, 4)
    out = dp(x)

    row_sums = out.sum(dim=1)
    assert torch.all((row_sums == 0.0) | (row_sums == 8.0))  # 0 or scaled by 1/keep=2.0
    assert row_sums.max().item() == 8.0  # confirms scaling happened, not all-zeroed
    zeroed_frac = (row_sums == 0.0).float().mean().item()
    assert 0.3 < zeroed_frac < 0.7  # ~50% with 200 samples, seeded


def test_drop_path_is_identity_at_zero_rate_and_in_eval():
    from video_encoding.models.ViT.droppath import DropPath

    x = torch.randn(10, 4)

    dp = DropPath(drop_prob=0.5)
    dp.eval()
    assert torch.equal(dp(x), x)  # eval mode -> always identity
    assert dp(x).sub(x).abs().max().item() == 0.0

    dp2 = DropPath(drop_prob=0.0)
    dp2.train()
    assert torch.equal(dp2(x), x)  # zero rate -> always identity
    assert dp2(x).sub(x).abs().max().item() == 0.0


def test_encoder_scales_drop_path_linearly_across_blocks():
    encoder = VideoMAEEncoder(_tiny_cfg(drop_path_rate=0.2))
    rates = [blk.drop_path.drop_prob for blk in encoder.blocks]
    assert rates[0] == pytest.approx(0.0)
    assert rates[-1] == pytest.approx(0.2)
    assert rates == sorted(rates)


def test_encoder_builds_from_config_missing_all_new_axis_fields():
    """Historical stored configs (pre-dating this refactor) have none of the
    new keys -- the encoder must still build with today's exact defaults."""
    cfg = _tiny_cfg()  # no norm_type/mlp_type/qk_norm/layer_scale_init/num_registers
    encoder = VideoMAEEncoder(cfg)
    assert encoder.registers.num_registers == 0
    out = encoder(torch.randn(2, 3, 4, 16, 16))
    assert out.shape == (2, 32)


def test_encoder_respects_new_axis_overrides():
    cfg = _tiny_cfg(
        norm_type="rmsnorm",
        mlp_type="swiglu",
        qk_norm=True,
        layer_scale_init=1e-5,
    )
    encoder = VideoMAEEncoder(cfg)
    out = encoder(torch.randn(2, 3, 4, 16, 16))
    assert out.shape == (2, 32)


def test_encoder_with_registers_pooled_output_shape_unaffected():
    cfg = _tiny_cfg(num_registers=2)
    encoder = VideoMAEEncoder(cfg)
    assert encoder.registers.num_registers == 2
    out = encoder(torch.randn(2, 3, 4, 16, 16))
    assert out.shape == (2, 32)  # registers stripped before pooling


def test_forward_tokens_excludes_registers_from_returned_tokens():
    cfg = _tiny_cfg(num_registers=2)
    encoder = VideoMAEEncoder(cfg)
    x = encoder.patch_embed(torch.randn(2, 3, 4, 16, 16)) + encoder.pos_embed
    keep_indices = torch.arange(4)  # keep 4 of the 8 patch tokens
    out = encoder.forward_tokens(x, keep_indices)
    assert out.shape == (2, 4, 32)  # 4 kept patch tokens, no registers leaked through
