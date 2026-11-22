"""From-scratch VideoMAEEncoder: shape, gradient flow, and the token guard.

Uses a tiny ViT config so a real forward/backward runs on CPU in CI -- this is
the real-architecture coverage the HF-loading design could not have (loading
those weights needed network + a GPU).
"""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from stern.models.videomae.encoder import VideoMAEEncoder, preprocess_window
from stern.models import VideoMAEVARegressor


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
    # embed_dim=48: divisible by both the encoder's num_heads=4 and the VA
    # head's fixed num_heads=12.
    torch.manual_seed(0)
    model = VideoMAEVARegressor(_tiny_cfg(embed_dim=48))
    out = model(torch.randn(3, 3, 4, 16, 16))

    assert out.shape == (3, 2)
    assert torch.all(out >= -1.0) and torch.all(out <= 1.0)
    # Measured 0.14 at seed 0. Bounds alone pass for a saturated tanh too.
    assert out.abs().max().item() < 0.99


def test_drop_path_zeroes_and_scales_samples():
    from stern.models.vit.droppath import DropPath

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
    from stern.models.vit.droppath import DropPath

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
    keep_indices = torch.arange(4).unsqueeze(0).expand(2, -1)  # (B=2, 4)
    out = encoder.forward_tokens(x, keep_indices)
    assert out.shape == (2, 4, 32)  # 4 kept patch tokens, no registers leaked through


# --- opt-in intermediate outputs (E2b arm C) ------------------------------


def test_forward_tokens_returns_requested_intermediates():
    cfg = _tiny_cfg(depth=4)
    encoder = VideoMAEEncoder(cfg)
    x = encoder.patch_embed(torch.randn(2, 3, 4, 16, 16)) + encoder.pos_embed
    keep_indices = torch.arange(4).unsqueeze(0).expand(2, -1)

    out, intermediates = encoder.forward_tokens(x, keep_indices, return_layers=[1, 3])

    assert out.shape == (2, 4, 32)
    assert len(intermediates) == 2
    for tokens in intermediates:
        assert tokens.shape == out.shape  # registers stripped, same as the final


def test_forward_tokens_intermediates_do_not_change_the_final_output():
    """Instrumentation must be observation-only: asking for intermediates must
    not perturb the value the encoder returns."""
    cfg = _tiny_cfg(depth=4)
    encoder = VideoMAEEncoder(cfg).eval()
    x = encoder.patch_embed(torch.randn(2, 3, 4, 16, 16)) + encoder.pos_embed
    keep_indices = torch.arange(4).unsqueeze(0).expand(2, -1)

    with torch.no_grad():
        plain = encoder.forward_tokens(x, keep_indices)
        instrumented, _ = encoder.forward_tokens(x, keep_indices, return_layers=[2])

    assert torch.equal(plain, instrumented)


def test_forward_tokens_last_intermediate_matches_the_final_output():
    """Requesting the final layer must hand back the same tensor the function
    returns -- a cheap guard that the layer indices are 1-based and not off by
    one."""
    cfg = _tiny_cfg(depth=4)
    encoder = VideoMAEEncoder(cfg).eval()
    x = encoder.patch_embed(torch.randn(2, 3, 4, 16, 16)) + encoder.pos_embed
    keep_indices = torch.arange(4).unsqueeze(0).expand(2, -1)

    with torch.no_grad():
        out, intermediates = encoder.forward_tokens(x, keep_indices, return_layers=[4])

    assert torch.equal(out, intermediates[0])


def test_use_skip_false_removes_both_residual_adds_in_every_encoder_block():
    torch.manual_seed(0)
    encoder = VideoMAEEncoder(_tiny_cfg(use_skip=False))
    block = encoder.blocks[0]
    x = torch.randn(2, 8, 32)

    attn_out, _ = block.attn(block.norm1(x), need_weights=False)
    expected = block.mlp(block.norm2(attn_out))

    assert [b.use_skip for b in encoder.blocks] == [False, False]
    torch.testing.assert_close(block(x), expected)
    # the same block with its residual restored gives a different output
    block.use_skip = True
    assert (block(x) - expected).abs().max() > 0.1


def test_use_skip_defaults_to_true_and_leaves_init_unchanged():
    torch.manual_seed(0)
    default = VideoMAEEncoder(_tiny_cfg())
    torch.manual_seed(0)
    explicit = VideoMAEEncoder(_tiny_cfg(use_skip=True))
    x = torch.randn(2, 3, 4, 16, 16)

    assert [b.use_skip for b in default.blocks] == [True, True]
    torch.testing.assert_close(default(x), explicit(x), rtol=0, atol=0)


def test_skipless_cutting_skip_encoder_trains_one_step():
    torch.manual_seed(0)
    encoder = VideoMAEEncoder(_tiny_cfg(use_skip=False, weight_init="cutting_skip"))
    x = torch.randn(2, 3, 4, 16, 16)

    feat = encoder(x)
    feat.pow(2).mean().backward()

    assert torch.isfinite(feat).all()
    grad = encoder.blocks[0].attn.qkv.weight.grad
    assert grad is not None and grad.abs().sum() > 0

    torch.manual_seed(0)
    kept = VideoMAEEncoder(_tiny_cfg(weight_init="cutting_skip"))
    kept_feat = kept(x)
    kept_feat.pow(2).mean().backward()
    # same seed, same weights: only the cut skips separate the two
    assert (feat - kept_feat).abs().max() > 1e-3
    assert not torch.allclose(grad, kept.blocks[0].attn.qkv.weight.grad)


def test_mae_pretraining_cuts_encoder_skips_and_keeps_decoder_skips():
    from stern.models.videomae.mae import VideoMAEForPreTraining
    from stern.models.videomae.masking import RandomTubeMasking

    cfg = _tiny_cfg(
        use_skip=False,
        image_size=32,
        patch_size=16,
        decoder_depth=2,
        decoder_embed_dim=8,
        decoder_num_heads=2,
        mask_ratio=0.5,
    )
    torch.manual_seed(0)
    model = VideoMAEForPreTraining(cfg)
    pixel_values = torch.randn(2, 3, 4, 32, 32)
    keep, masked = RandomTubeMasking(mask_ratio=0.5)(
        pixel_values, (2, 2, 2), torch.device("cpu"), batch_size=2
    )

    pred, target = model(pixel_values, keep, masked)
    loss = model.compute_loss(pred, target)

    assert [b.use_skip for b in model.encoder.blocks] == [False, False]
    assert [b.use_skip for b in model.decoder_blocks] == [True, True]
    assert torch.isfinite(loss)


def test_hydra_override_sets_use_skip_false_as_a_bool():
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    conf = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(conf), version_base=None):
        cfg = compose(
            "config",
            overrides=[
                "experiment=videomae_mesh_ssl",
                "+model.use_skip=false",
                "model.weight_init=cutting_skip",
            ],
        )

    assert cfg.model.use_skip is False
    assert cfg.model.weight_init == "cutting_skip"
