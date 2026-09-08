from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from video_encoding.models.levjepa.encoder import LeVJEPAEncoder


def _tiny_cfg(**overrides):
    base = dict(
        num_frames=4,
        image_size=32,
        in_channels=3,
        patch_size=16,
        embed_dim=12,
        depth=2,
        num_heads=2,
        mlp_ratio=2.0,
        dropout=0.0,
        drop_path_rate=0.0,
        drop_ratio=0.5,
    )
    base.update(overrides)
    return OmegaConf.create(base)


def test_levjepa_encoder_has_no_patch_embed_or_pos_embed_toplevel_attrs():
    """Regression guard for the VideoMAEVARegressor drop-in path: if this
    ever gains a `pos_embed` attribute, _backbone_tokens's hasattr check
    would silently switch to the wrong code path."""
    enc = LeVJEPAEncoder(_tiny_cfg())
    assert hasattr(enc, "patch_embed") is False
    assert hasattr(enc, "pos_embed") is False


def test_levjepa_encoder_output_shape_eval_mode_keeps_all_tokens():
    cfg = _tiny_cfg()
    enc = LeVJEPAEncoder(cfg)
    enc.eval()
    x = torch.randn(2, 3, 4, 32, 32)
    out = enc(x)
    grid = 32 // 16
    expected_patches = 4 * grid * grid  # T * H' * W', tubelet_size=1
    assert out.shape == (2, expected_patches + 1, 12)


def test_levjepa_encoder_drops_tokens_in_train_mode():
    cfg = _tiny_cfg(drop_ratio=0.5)
    enc = LeVJEPAEncoder(cfg)
    enc.train()
    x = torch.randn(2, 3, 4, 32, 32)
    out = enc(x)
    grid = 32 // 16
    total_patches = 4 * grid * grid
    assert out.shape[1] < total_patches + 1
    assert out.shape[1] >= 1  # cls token always present


def test_levjepa_encoder_cls_never_dropped():
    cfg = _tiny_cfg(drop_ratio=0.95)
    enc = LeVJEPAEncoder(cfg)
    enc.train()
    x = torch.randn(2, 3, 4, 32, 32)
    out = enc(x)
    assert out.shape[1] >= 1


def test_levjepa_encoder_forward_cls_matches_first_token():
    cfg = _tiny_cfg()
    enc = LeVJEPAEncoder(cfg)
    enc.eval()
    x = torch.randn(1, 3, 4, 32, 32)
    torch.manual_seed(0)
    full = enc(x)
    torch.manual_seed(0)
    cls_only = enc.forward_cls(x)
    assert torch.allclose(full[:, 0], cls_only, atol=1e-6)


def test_levjepa_encoder_builds_with_vit_base_geometry():
    """ViT-B (embed_dim=768, num_heads=12) gives head_dim=64, which broke the
    old Rotary3D(head_dim % 6 == 0) requirement -- the config every real
    training run uses must actually build."""
    cfg = OmegaConf.create(
        dict(
            num_frames=2,
            image_size=32,
            in_channels=3,
            patch_size=16,
            embed_dim=768,
            depth=1,
            num_heads=12,
            mlp_ratio=4.0,
            dropout=0.0,
            drop_path_rate=0.0,
            drop_ratio=0.0,
        )
    )
    enc = LeVJEPAEncoder(cfg)
    out = enc(torch.randn(1, 3, 2, 32, 32))
    assert out.shape[-1] == 768


def test_levjepa_encoder_per_sample_token_dropping():
    """Each batch element must keep a different random subset of tokens --
    a shared drop mask across the batch would correlate every sample's
    augmentation, weakening SIGReg's cross-sample independence assumption."""
    torch.manual_seed(0)
    cfg = _tiny_cfg(drop_ratio=0.5)
    enc = LeVJEPAEncoder(cfg)
    enc.train()
    x = torch.randn(8, 3, 4, 32, 32)
    coords = enc._patch_coords(4, 32, 32, x.device)
    coords = coords.unsqueeze(0).expand(8, -1, 3)
    tokens = torch.arange(8 * coords.shape[1]).float().view(8, coords.shape[1], 1)
    _, kept_coords = enc._drop_tokens(tokens, coords)
    # If every sample kept the same positions, all rows of kept h/w coords
    # would be identical across the batch.
    assert not torch.equal(kept_coords[0], kept_coords[1])


def test_levjepa_encoder_local_crop_coords_map_into_global_grid_frame():
    """A local crop's RoPE coordinates must be expressed in the *global*
    view's patch-grid units (scaled+offset by the crop box), not the local
    view's own 0..grid-1 indices -- otherwise every crop looks like it was
    taken from the top-left corner at the wrong scale."""
    cfg = _tiny_cfg(image_size=32, patch_size=16)  # global_grid=2
    enc = LeVJEPAEncoder(cfg)
    T, H, W = 1, 16, 16  # local view: 1x1 patch grid, single patch coord
    device = torch.device("cpu")

    top_left_box = torch.tensor([[0.0, 0.0, 0.5, 0.5]])
    bottom_right_box = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    coords_tl = enc._patch_coords_for_crop(T, H, W, top_left_box, device)
    coords_br = enc._patch_coords_for_crop(T, H, W, bottom_right_box, device)

    # global_grid=2, so a half-size crop's single patch center should land
    # near grid coordinate 0.5 (top-left) vs 1.5 (bottom-right).
    assert coords_tl[0, 0, 1].item() == pytest.approx(0.5)
    assert coords_tl[0, 0, 2].item() == pytest.approx(0.5)
    assert coords_br[0, 0, 1].item() == pytest.approx(1.5)
    assert coords_br[0, 0, 2].item() == pytest.approx(1.5)


def test_levjepa_encoder_block_causal_mask_blocks_future_frames():
    """A patch token in frame 0 must produce identical output whether or not
    frame 3's content is perturbed, under block-causal attention -- proving
    the mask, not just its construction, is wired into the block stack."""
    torch.manual_seed(0)
    cfg = _tiny_cfg(
        drop_ratio=0.0
    )  # disable dropping to make token identity deterministic
    enc = LeVJEPAEncoder(cfg)
    enc.eval()
    x = torch.randn(1, 3, 4, 32, 32)
    x_perturbed = x.clone()
    x_perturbed[:, :, 3] = torch.randn_like(x_perturbed[:, :, 3])
    with torch.no_grad():
        out_a = enc(x)
        out_b = enc(x_perturbed)
    grid = 32 // 16
    frame0_tokens = grid * grid
    # token index 1 is the first patch token (index 0 is cls); frame-0 patches
    # occupy indices [1, 1+frame0_tokens)
    assert torch.allclose(
        out_a[:, 1 : 1 + frame0_tokens], out_b[:, 1 : 1 + frame0_tokens], atol=1e-4
    )
