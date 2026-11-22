from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from stern.training.pretrain_levjepa import (
    LeVJEPAViewDataset,
    LeVJEPAViewIterableDataset,
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


@pytest.mark.parametrize(
    ("dataset_kind", "num_local", "expected_items"),
    [
        pytest.param("map_style", 3, 1, id="map_style_first_item"),
        pytest.param("iterable", 2, 3, id="iterable_stream"),
    ],
)
def test_levjepa_view_dataset_builds_views_per_item(
    dataset_kind, num_local, expected_items
):
    if dataset_kind == "map_style":
        raw = _FakeRawDataset(n=2, num_frames=16, size=32)
        ds = LeVJEPAViewDataset(raw, num_local=num_local, global_size=32, local_size=16)
        items = [ds[0]]
    else:
        raw = _FakeRawIterableDataset(n=3, num_frames=16, size=32)
        ds = LeVJEPAViewIterableDataset(
            raw, num_local=num_local, global_size=32, local_size=16
        )
        items = list(ds)

    assert len(items) == expected_items
    global_view, local_views, crop_boxes = items[0]
    assert global_view.shape == (3, 16, 32, 32)
    assert local_views.shape == (num_local, 3, 16, 16, 16)
    assert crop_boxes.shape == (num_local, 4)


class _FakeRawIterableDataset(torch.utils.data.IterableDataset):
    """Mimics MeshShardIterableDataset(raw_frames=True): streams (T, H, W,
    3) uint8 clips, with a set_round hook to verify passthrough."""

    def __init__(self, n=3, num_frames=16, size=32):
        self._n = n
        self._num_frames = num_frames
        self._size = size
        self.set_round_calls = []

    def set_round(self, round_num: int) -> None:
        self.set_round_calls.append(round_num)

    def __iter__(self):
        for _ in range(self._n):
            yield torch.randint(
                0, 256, (self._num_frames, self._size, self._size, 3), dtype=torch.uint8
            )


def test_levjepa_view_iterable_dataset_forwards_set_round():
    raw = _FakeRawIterableDataset(n=1, num_frames=16, size=32)
    ds = LeVJEPAViewIterableDataset(raw, num_local=2, global_size=32, local_size=16)

    ds.set_round(7)

    assert raw.set_round_calls == [7]


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
    import stern.training.pretrain_levjepa as mod

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
    import stern.training.pretrain_levjepa as mod

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
    import stern.training.pretrain_levjepa as mod

    monkeypatch.setattr(
        mod, "_build_levjepa_loader", lambda cfg, rank, world_size: _fake_loader()
    )

    # Epoch 0 (2 batches) in the first call leaves AdamW step=2 on disk.
    cfg = _tiny_levjepa_cfg(tmp_path, epochs=1)
    run_levjepa_pretrain_stage(
        cfg, rank=0, world_size=1, local_rank=0, device=torch.device("cpu")
    )

    saved_epochs: list[int] = []
    real_payload = mod._payload

    def _spying_payload(
        encoder, projector, optimizer, ema, epoch, total_loss, **kwargs
    ):
        saved_epochs.append(epoch)
        return real_payload(
            encoder, projector, optimizer, ema, epoch, total_loss, **kwargs
        )

    monkeypatch.setattr(mod, "_payload", _spying_payload)
    cfg_resumed = _tiny_levjepa_cfg(tmp_path, epochs=2, resume=True)
    result = run_levjepa_pretrain_stage(
        cfg_resumed, rank=0, world_size=1, local_rank=0, device=torch.device("cpu")
    )

    ckpt_root = tmp_path / cfg.experiment.name
    latest = torch.load(ckpt_root / "latest.pt", weights_only=False)
    # Only epoch 1 runs on resume; a run that ignores the checkpoint saves [0, 1].
    assert saved_epochs == [1]
    assert latest["epoch"] == 1
    # Epoch 1 adds 2 batches: 2 + 2 = 4 only if the AdamW state was reloaded
    # (a fresh optimizer would end at 2).
    steps = {int(s["step"]) for s in latest["optimizer_state_dict"]["state"].values()}
    assert steps == {4}
    assert not torch.isnan(torch.tensor(result["total_loss"]))


def test_grad_clipping_bounds_gradient_norm(tmp_path, monkeypatch):
    """grad_clip_norm=0.5 must be the exact max_norm passed to
    clip_grad_norm_, and the post-clip gradient norm it actually produces
    must be <= 0.5 (not just "clipping ran")."""
    import stern.training.pretrain_levjepa as mod

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
    import stern.training.pretrain_levjepa as mod

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


def test_lr_follows_warmup_cosine_schedule(tmp_path, monkeypatch):
    """pretrain_levjepa must drive the optimizer LR through
    training.pretrain._build_scheduler's warmup+cosine schedule, not train
    at a flat LR for the whole run -- flat AdamW has no LR decay at all,
    which this regresses against."""
    import stern.training.pretrain_levjepa as mod

    captured_lrs = []
    real_payload = mod._payload

    def _spying_payload(
        encoder, projector, optimizer, ema, epoch, total_loss, **kwargs
    ):
        captured_lrs.append(optimizer.param_groups[0]["lr"])
        return real_payload(
            encoder, projector, optimizer, ema, epoch, total_loss, **kwargs
        )

    monkeypatch.setattr(mod, "_payload", _spying_payload)
    monkeypatch.setattr(
        mod, "_build_levjepa_loader", lambda cfg, rank, world_size: _fake_loader()
    )

    cfg = _tiny_levjepa_cfg(tmp_path, epochs=3, warmup_epochs=1)
    run_levjepa_pretrain_stage(
        cfg, rank=0, world_size=1, local_rank=0, device=torch.device("cpu")
    )

    # Hand-derived for lr=1e-4, epochs=3, warmup_epochs=1. Warmup is a
    # LinearLR for one step (start_factor=1e-2); from step 1 on it is a
    # CosineAnnealingLR with T_max = epochs - warmup = 2. The payload is
    # written after scheduler.step(), so epoch e records the LR after e+1 steps:
    #   after 1 step: cosine at t=0          -> 1e-4
    #   after 2 steps: 1e-4 * (1+cos(pi/2))/2 -> 5e-5
    #   after 3 steps: 1e-4 * (1+cos(pi))/2   -> 0
    assert len(captured_lrs) == 3
    assert captured_lrs == pytest.approx([1e-4, 5e-5, 0.0], rel=1e-6, abs=1e-12)


def _run_capturing_lrs(cfg, monkeypatch, *, mod, real_payload):
    captured_lrs = []

    def _spying_payload(
        encoder, projector, optimizer, ema, epoch, total_loss, **kwargs
    ):
        captured_lrs.append(optimizer.param_groups[0]["lr"])
        return real_payload(
            encoder, projector, optimizer, ema, epoch, total_loss, **kwargs
        )

    monkeypatch.setattr(mod, "_payload", _spying_payload)
    run_levjepa_pretrain_stage(
        cfg, rank=0, world_size=1, local_rank=0, device=torch.device("cpu")
    )
    return captured_lrs


def test_scheduler_state_survives_resume(tmp_path, monkeypatch):
    """Resuming mid-schedule must continue the cosine curve a continuous run
    (same `epochs`/`warmup_epochs` target throughout) would have produced,
    not restart warmup from epoch 0 -- silently doing so would let LR spike
    back up after every preemption.

    Note: this deliberately keeps `epochs` identical between the checkpoint
    being resumed from and the resuming run. Changing `epochs` across a
    resume changes the cosine schedule's T_max, and torch's
    `scheduler.load_state_dict` overwrites the freshly-built scheduler's
    T_max with the checkpoint's old one -- so "resume with a larger epochs
    target" does not actually retarget the schedule. That's a separate,
    out-of-scope concern from what this test guards (same-target resume
    must not silently reset progress).
    """
    import stern.training.pretrain_levjepa as mod

    real_payload = mod._payload

    monkeypatch.setattr(
        mod, "_build_levjepa_loader", lambda cfg, rank, world_size: _fake_loader()
    )

    cfg = _tiny_levjepa_cfg(tmp_path / "continuous", epochs=3, warmup_epochs=1)
    continuous_lrs = _run_capturing_lrs(
        cfg, monkeypatch, mod=mod, real_payload=real_payload
    )
    assert len(continuous_lrs) == 3

    # Build the on-disk checkpoint a real run would leave after epoch 0
    # completes, using the exact same construction path
    # (_build_scheduler + one step + _payload) production code uses --
    # rather than interrupting a full run mid-flight.
    resumed_cfg = _tiny_levjepa_cfg(
        tmp_path / "resumed", epochs=3, warmup_epochs=1, resume=True
    )
    encoder = mod.LeVJEPAEncoder(resumed_cfg.model)
    projector = mod.Projector(
        embed_dim=int(resumed_cfg.model.embed_dim),
        hidden_dim=int(resumed_cfg.model.projector_hidden),
        output_dim=int(resumed_cfg.model.embed_dim_k),
    )
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(projector.parameters()),
        lr=float(resumed_cfg.train.lr),
    )
    scheduler = mod._build_scheduler(optimizer, resumed_cfg)
    ema = mod.WeightEMA(encoder, decay=float(resumed_cfg.model.polyak_decay))
    scheduler.step()  # epoch 0 completing
    payload = real_payload(
        encoder, projector, optimizer, ema, 0, 0.0, scheduler=scheduler
    )
    ckpt_root = (tmp_path / "resumed") / resumed_cfg.experiment.name
    ckpt_root.mkdir(parents=True, exist_ok=True)
    mod.ckpt_io.save(ckpt_root / "latest.pt", payload)

    resumed_lrs = _run_capturing_lrs(
        resumed_cfg, monkeypatch, mod=mod, real_payload=real_payload
    )

    # only epochs 1, 2 run on resume (epoch 0's checkpoint already exists)
    assert resumed_lrs == pytest.approx(continuous_lrs[1:], rel=1e-5)
