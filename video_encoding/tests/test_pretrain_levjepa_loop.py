from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from video_encoding.training.pretrain_levjepa import (
    LeVJEPAViewDataset,
    run_levjepa_pretrain_stage,
)

pytestmark = pytest.mark.integration


class _FakeRawDataset(torch.utils.data.Dataset):
    """Mimics SSLWindowDataset(raw_frames=True): (T, H, W, 3) uint8 per item."""

    def __init__(self, n=4, num_frames=16, size=32):
        self._n = n
        self._num_frames = num_frames
        self._size = size

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        return torch.randint(
            0, 256, (self._num_frames, self._size, self._size, 3), dtype=torch.uint8
        )


def test_levjepa_view_dataset_builds_views_per_item():
    raw = _FakeRawDataset(n=2, num_frames=16, size=32)
    ds = LeVJEPAViewDataset(raw, num_local=3, global_size=32, local_size=16)

    global_view, local_views, crop_boxes = ds[0]

    assert global_view.shape == (3, 16, 32, 32)
    assert local_views.shape == (3, 3, 16, 16, 16)
    assert crop_boxes.shape == (3, 4)


def _fake_loader(
    num_local=2, num_frames=16, image_size=32, local_size=16, batch_size=2
):
    raw = _FakeRawDataset(n=4, num_frames=num_frames, size=image_size)
    ds = LeVJEPAViewDataset(
        raw, num_local=num_local, global_size=image_size, local_size=local_size
    )
    return torch.utils.data.DataLoader(ds, batch_size=batch_size)


def _tiny_levjepa_cfg(tmp_path, **train_overrides):
    train = dict(
        checkpoint_root=str(tmp_path),
        batch_size=2,
        seed=0,
        epochs=1,
        lr=1e-4,
        weight_decay=0.0,
        amp="off",
        resume=False,
    )
    train.update(train_overrides)
    return OmegaConf.create(
        dict(
            experiment=dict(name="levjepa_smoke_test"),
            model=dict(
                encoder="levjepa",
                num_frames=16,
                image_size=32,
                local_image_size=16,
                in_channels=3,
                patch_size=16,
                embed_dim=12,
                depth=1,
                num_heads=2,
                mlp_ratio=2.0,
                dropout=0.0,
                drop_path_rate=0.0,
                drop_ratio=0.5,
                num_local_views=2,
                sigreg_lambda=0.02,
                sigreg_directions=32,
                projector_hidden=16,
                embed_dim_k=8,
                polyak_decay=0.9,
                polyak_update_every=1,
            ),
            train=train,
        )
    )


def test_run_levjepa_pretrain_stage_one_step_smoke(tmp_path, monkeypatch):
    """CPU smoke test on synthetic data: one optimizer step must not NaN and
    the SIGReg term must be strictly positive (regularizer actually engaged,
    not silently short-circuited)."""
    import video_encoding.training.pretrain_levjepa as mod

    monkeypatch.setattr(
        mod, "_build_levjepa_loader", lambda cfg, rank, world_size: _fake_loader()
    )

    cfg = _tiny_levjepa_cfg(tmp_path)
    result = run_levjepa_pretrain_stage(
        cfg, rank=0, world_size=1, local_rank=0, device=torch.device("cpu")
    )
    assert "sigreg_loss" in result
    assert result["sigreg_loss"] > 0
    assert not torch.isnan(torch.tensor(result["total_loss"]))


def test_checkpoint_saved_every_epoch_includes_projector_and_ema(tmp_path, monkeypatch):
    """A run killed mid-training must not lose everything -- and the
    projector (needed to keep training, unlike at Stage-2 inference time)
    and EMA shadow must be resumable, not silently dropped."""
    import video_encoding.training.pretrain_levjepa as mod

    monkeypatch.setattr(
        mod, "_build_levjepa_loader", lambda cfg, rank, world_size: _fake_loader()
    )

    cfg = _tiny_levjepa_cfg(tmp_path, epochs=2)
    run_levjepa_pretrain_stage(
        cfg, rank=0, world_size=1, local_rank=0, device=torch.device("cpu")
    )

    ckpt_root = tmp_path / cfg.experiment.name
    latest = torch.load(ckpt_root / "latest.pt", weights_only=False)
    assert latest["epoch"] == 1  # both epochs ran, 0-indexed
    assert "projector_state_dict" in latest
    assert "ema_state_dict" in latest


def test_resume_continues_from_saved_epoch(tmp_path, monkeypatch):
    import video_encoding.training.pretrain_levjepa as mod

    monkeypatch.setattr(
        mod, "_build_levjepa_loader", lambda cfg, rank, world_size: _fake_loader()
    )

    cfg = _tiny_levjepa_cfg(tmp_path, epochs=1)
    run_levjepa_pretrain_stage(
        cfg, rank=0, world_size=1, local_rank=0, device=torch.device("cpu")
    )

    cfg_resumed = _tiny_levjepa_cfg(tmp_path, epochs=2, resume=True)
    result = run_levjepa_pretrain_stage(
        cfg_resumed, rank=0, world_size=1, local_rank=0, device=torch.device("cpu")
    )

    ckpt_root = tmp_path / cfg.experiment.name
    latest = torch.load(ckpt_root / "latest.pt", weights_only=False)
    assert latest["epoch"] == 1
    assert not torch.isnan(torch.tensor(result["total_loss"]))


def test_grad_clipping_bounds_gradient_norm(tmp_path, monkeypatch):
    """grad_clip_norm=0.5 must be the exact max_norm passed to
    clip_grad_norm_, and the post-clip gradient norm it actually produces
    must be <= 0.5 (not just "clipping ran")."""
    import video_encoding.training.pretrain_levjepa as mod

    captured_max_norms = []
    captured_post_clip_norms = []
    real_clip = torch.nn.utils.clip_grad_norm_

    def _spying_clip(parameters, max_norm, *args, **kwargs):
        real_clip(parameters, max_norm, *args, **kwargs)
        captured_max_norms.append(max_norm)
        params = list(parameters)
        post_norm = torch.sqrt(
            sum(p.grad.detach().pow(2).sum() for p in params if p.grad is not None)
        )
        captured_post_clip_norms.append(float(post_norm))

    monkeypatch.setattr(mod.torch.nn.utils, "clip_grad_norm_", _spying_clip)
    monkeypatch.setattr(
        mod, "_build_levjepa_loader", lambda cfg, rank, world_size: _fake_loader()
    )

    cfg = _tiny_levjepa_cfg(tmp_path, grad_clip_norm=0.5)
    run_levjepa_pretrain_stage(
        cfg, rank=0, world_size=1, local_rank=0, device=torch.device("cpu")
    )

    assert captured_max_norms == [0.5] * len(captured_max_norms)
    assert len(captured_post_clip_norms) > 0
    assert all(n <= 0.5 + 1e-4 for n in captured_post_clip_norms)


def test_epoch_loss_is_mean_over_batches_not_last_batch(tmp_path, monkeypatch):
    """Regression guard: logging the last batch's loss instead of the
    epoch's mean makes TensorBoard curves single-sample noise."""
    import video_encoding.training.pretrain_levjepa as mod

    captured_step_losses = []
    real_step = mod._levjepa_step

    def _spying_step(*args, **kwargs):
        total, components = real_step(*args, **kwargs)
        captured_step_losses.append(float(total.item()))
        return total, components

    monkeypatch.setattr(mod, "_levjepa_step", _spying_step)
    monkeypatch.setattr(
        mod, "_build_levjepa_loader", lambda cfg, rank, world_size: _fake_loader()
    )

    cfg = _tiny_levjepa_cfg(tmp_path)
    result = run_levjepa_pretrain_stage(
        cfg, rank=0, world_size=1, local_rank=0, device=torch.device("cpu")
    )

    # _fake_loader() default: n=4 items, batch_size=2 -> 2 batches/epoch, epochs=1
    assert len(captured_step_losses) == 2
    expected_mean = sum(captured_step_losses) / len(captured_step_losses)
    assert result["total_loss"] == pytest.approx(expected_mean, rel=1e-4)
