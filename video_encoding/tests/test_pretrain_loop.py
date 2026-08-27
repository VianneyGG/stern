"""Unit tests for the SSL training loop instrumentation (_train_epoch_ssl).

Runs the real VideoMAEForPreTraining at a tiny resolution on CPU so the test
exercises the actual forward + compute_loss + encoder.token_grid path (the N=1
single-process branch that the .module guard fixes), plus the max_steps cap,
gradient accumulation, and the throughput/VRAM metrics dict.
"""

import math

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from video_encoding.models.videomaev2.masking import RandomTubeMasking
from video_encoding.models.videomaev2.mae import VideoMAEForPreTraining
from video_encoding.training.pretrain import _train_epoch_ssl


def _tiny_cfg() -> DictConfig:
    # image 32 / patch 16 -> 2x2 spatial; frames 4 / tubelet 2 -> 2 temporal.
    return DictConfig(
        {
            "num_frames": 4,
            "image_size": 32,
            "in_channels": 3,
            "tubelet_size": 2,
            "patch_size": 16,
            "embed_dim": 32,
            "depth": 1,
            "num_heads": 4,
            "mlp_ratio": 2.0,
            "dropout": 0.0,
            "decoder_depth": 1,
            "decoder_embed_dim": 32,
            "decoder_num_heads": 4,
            "mask_ratio": 0.5,
        }
    )


def _loader(n_samples: int, batch_size: int) -> DataLoader:
    clips = [torch.randn(3, 4, 32, 32) for _ in range(n_samples)]
    return DataLoader(clips, batch_size=batch_size, num_workers=0)


def test_train_epoch_reports_metrics_and_respects_max_steps():
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = VideoMAEForPreTraining(_tiny_cfg()).to(device)  # bare module: N=1 path
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    masker = RandomTubeMasking(mask_ratio=0.5)

    stats = _train_epoch_ssl(
        model,
        _loader(n_samples=8, batch_size=2),
        optimizer,
        device,
        masker,
        accum_steps=2,
        grad_clip=1.0,
        max_steps=3,
        timing_warmup_steps=0,  # time from step 0 so throughput is measured
        amp="off",
    )

    # max_steps caps micro-steps; each step is batch_size=2 samples.
    assert stats["steps"] == 3
    assert stats["samples"] == 6
    assert math.isfinite(stats["loss"])
    assert stats["samples_per_s_per_gpu"] > 0.0
    assert stats["step_time_ms_mean"] > 0.0
    assert stats["peak_vram_gb"] == 0.0  # CPU run


def test_train_epoch_full_pass_without_cap():
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = VideoMAEForPreTraining(_tiny_cfg()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    masker = RandomTubeMasking(mask_ratio=0.5)

    stats = _train_epoch_ssl(
        model,
        _loader(n_samples=4, batch_size=2),
        optimizer,
        device,
        masker,
        accum_steps=1,
        max_steps=0,  # full epoch
        timing_warmup_steps=0,
        amp="off",
    )
    assert stats["steps"] == 2  # 4 samples / batch 2
    assert stats["samples"] == 4


def test_train_epoch_skips_nonfinite_loss_without_poisoning_average():
    """A non-finite loss on one step must not corrupt the epoch average or
    crash training -- observed in the wild (bf16 + real data occasionally
    produces a NaN on an isolated batch); backward() on a NaN loss would
    otherwise broadcast NaN into every parameter's .grad permanently."""
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = VideoMAEForPreTraining(_tiny_cfg()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    masker = RandomTubeMasking(mask_ratio=0.5)

    real_compute_loss = model.compute_loss
    calls = {"n": 0}

    def flaky_compute_loss(pred, target, aux=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return real_compute_loss(pred, target, aux=aux) * float("nan")
        return real_compute_loss(pred, target, aux=aux)

    model.compute_loss = flaky_compute_loss

    stats = _train_epoch_ssl(
        model,
        _loader(n_samples=6, batch_size=2),
        optimizer,
        device,
        masker,
        accum_steps=1,
        max_steps=3,
        timing_warmup_steps=0,
        amp="off",
    )

    assert stats["nan_steps"] == 1
    assert stats["steps"] == 3
    assert stats["samples"] == 4  # the NaN step's 2 samples excluded from count
    assert math.isfinite(stats["loss"])  # average over the 2 finite steps only
    for p in model.parameters():
        assert p.grad is None or torch.isfinite(p.grad).all()


def _run(
    loader, *, abandon_check=None, abandon_check_every=50, max_steps=0, accum_steps=1
):
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = VideoMAEForPreTraining(_tiny_cfg()).to(device)
    return _train_epoch_ssl(
        model,
        loader,
        torch.optim.AdamW(model.parameters(), lr=1e-3),
        device,
        RandomTubeMasking(mask_ratio=0.5),
        accum_steps=accum_steps,
        max_steps=max_steps,
        timing_warmup_steps=0,
        amp="off",
        abandon_check=abandon_check,
        abandon_check_every=abandon_check_every,
    )


def test_no_abandon_check_is_unchanged_behaviour():
    stats = _run(_loader(n_samples=8, batch_size=2))
    assert stats["abandoned"] is False
    assert stats["steps"] == 4


def test_abandon_check_never_true_runs_the_full_epoch():
    calls = []

    def _check():
        calls.append(1)
        return False

    stats = _run(
        _loader(n_samples=8, batch_size=2), abandon_check=_check, abandon_check_every=1
    )
    assert stats["abandoned"] is False
    assert stats["steps"] == 4
    assert len(calls) == 4  # polled once per step, never short-circuited


def test_abandon_check_stops_the_epoch_early():
    """The pool merged this round without us -- every further step is waste."""
    state = {"n": 0}

    def _check():
        state["n"] += 1
        return state["n"] >= 2

    stats = _run(
        _loader(n_samples=20, batch_size=2), abandon_check=_check, abandon_check_every=1
    )
    assert stats["abandoned"] is True
    assert stats["steps"] == 2  # stopped well short of the 10 available
    assert math.isfinite(stats["loss"])  # work done before the break still counts


def test_abandon_is_not_polled_mid_accumulation():
    """Breaking between a backward and its boundary step would drop
    accum_steps-1 micro-batches' gradients on the floor."""
    seen_at = []
    state = {"steps": 0}

    def _check():
        seen_at.append(state["steps"])
        return False

    class _CountingLoader:
        def __init__(self, inner):
            self._inner = inner

        def __iter__(self):
            for batch in self._inner:
                state["steps"] += 1
                yield batch

        def __len__(self):
            return len(self._inner)

    _run(
        _CountingLoader(_loader(n_samples=16, batch_size=2)),
        abandon_check=_check,
        abandon_check_every=1,
        accum_steps=4,
    )
    # Polled only on accumulation boundaries: steps 4, 8, ...
    assert seen_at == [4, 8]
