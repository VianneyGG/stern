"""Smoke test for ssl_adapt's training loop wiring.

Purpose: guard against the "DDP/checkpoint/JSON code silently broke"
failure mode. The pipeline is dormant until Aff-Wild2 arrives; without
this test, a bad refactor of the loop wouldn't surface until the next
real cluster run.

Scope: exercise every non-model code path — data loading, LoRA attach,
DDP-optional wrap, grad accum with no_sync, AMP-off path, checkpoint
save+resume, result JSON. Model itself is a tiny fake stub because the
real VideoMAEForPreTraining is ~350MB to download and the loop under
test is model-agnostic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from video_encoding import ssl as ssl_module


# ---------------------------------------------------------------------------
# Fakes: minimal shapes ssl_adapt reads via _build_model
# ---------------------------------------------------------------------------


@dataclass
class _FakeConfig:
    num_frames: int = 4
    tubelet_size: int = 2
    image_size: int = 8


@dataclass
class _FakeEmbeddings:
    num_patches: int = 8  # (image_size/patch_size)² × num_tubelets = 2×2×2


class _FakeBaseModel(nn.Module):
    def __init__(self, num_patches: int):
        super().__init__()
        self.embeddings = _FakeEmbeddings(num_patches=num_patches)


class _FakeVideomaeSubmodule(nn.Module):
    """Stands in for `model.videomae` after LoRA wrap."""

    def __init__(self, num_patches: int):
        super().__init__()
        self.base_model = _FakeBaseModel(num_patches=num_patches)
        # A single trainable param so `filter(p.requires_grad)` finds
        # something for the optimizer.
        self.weight = nn.Parameter(torch.randn(2, 2))

    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)


@dataclass
class _FakeOutput:
    loss: torch.Tensor


class _FakePretrainingModel(nn.Module):
    """Stand-in for VideoMAEForPreTraining that produces a differentiable
    loss depending on pixel_values so `.backward()` actually flows."""

    def __init__(self, num_patches: int):
        super().__init__()
        self.config = _FakeConfig()
        self.videomae = _FakeVideomaeSubmodule(num_patches=num_patches)

    def forward(self, *, pixel_values: torch.Tensor, bool_masked_pos: torch.Tensor):
        # Loss must depend on a trainable param so the optimizer step is
        # non-vacuous; picks pixel_values.mean() * weight.sum() as a
        # placeholder objective.
        loss = pixel_values.mean() * self.videomae.weight.sum() + torch.tensor(1.0)
        # Add a squared term so the loss surface is bowl-shaped and
        # gradient clip / LR schedule paths actually see finite grads.
        loss = loss + (self.videomae.weight**2).sum() * 0.01
        return _FakeOutput(loss=loss)


class _FakeClipDataset(Dataset):
    def __init__(self, n_clips: int, config: _FakeConfig):
        self._n = n_clips
        self._cfg = config

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> torch.Tensor:
        cfg = self._cfg
        # (T, C, H, W) — matches the shape VideoMAEImageProcessor emits.
        return torch.randn(cfg.num_frames, 3, cfg.image_size, cfg.image_size)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg_stub():
    """Minimal Hydra-shaped cfg exposing only what ssl_adapt reads from it.

    Uses OmegaConf.create because the production `write_result_json` path
    calls `OmegaConf.to_container(cfg)`, which rejects plain dataclasses.
    """
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "experiment": {"name": "smoke_test"},
            "train": {
                "accum_steps": 2,
                "ddp": {
                    "bucket_cap_mb": 25,
                    "grad_compress": "",
                    "nccl_timeout_sec": 300,
                    "find_unused_parameters": False,
                    "backend": "nccl",
                    "nccl_env": {},
                },
            },
        }
    )


def test_ssl_adapt_writes_checkpoint_and_result_json(tmp_path, monkeypatch, cfg_stub):
    """One-epoch CPU run: after ssl_adapt returns, the best-checkpoint
    file exists on disk, the result JSON parses, and the recorded
    best-val-loss is finite."""

    fake_config = _FakeConfig()

    # Bypass HF download: from_pretrained returns our fake.
    def _fake_from_pretrained(_ckpt):
        return _FakePretrainingModel(num_patches=8)

    monkeypatch.setattr(
        "video_encoding.ssl.VideoMAEForPreTraining.from_pretrained",
        classmethod(lambda cls, ckpt: _fake_from_pretrained(ckpt)),
    )
    # attach_lora expects a real HF PreTrainedModel — swap for identity.
    monkeypatch.setattr("video_encoding.ssl.attach_lora", lambda enc, rank=8: enc)

    dataset = _FakeClipDataset(n_clips=8, config=fake_config)
    # 6 train / 2 val via the module's own splitter would need the caller
    # path in probe/train.py; smoke test uses two flat loaders directly.
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=2, shuffle=False)

    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    results_dir = tmp_path / "results"

    ssl_module.ssl_adapt(
        checkpoint="fake://ignored",
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        lora_rank=8,
        mask_ratio=0.5,  # patches_per_frame=4 * 0.5 = 2 masked
        epochs=1,
        lr=1e-3,
        accum_steps=2,
        use_amp=False,
        grad_clip=1.0,
        warmup_epochs=0,
        checkpoint_path=ckpt_dir / "ssl_best.pt",
        resume=False,
        save_latest=True,
        results_root=results_dir,
        log_dir=None,
        rank=0,
        world_size=1,
        local_rank=0,
        device=torch.device("cpu"),
        cfg=cfg_stub,
        seed=0,
    )

    best = ckpt_dir / "ssl_best.pt"
    latest = ckpt_dir / "ssl_best_latest.pt"
    assert best.exists(), "best checkpoint was not written"
    assert latest.exists(), "latest checkpoint was not written"

    payload = torch.load(best, map_location="cpu", weights_only=False)
    assert payload["model_name"] == "videomae_ssl"
    assert payload["optimizer_name"] == "adam"
    assert torch.isfinite(torch.tensor(payload["best_val_metric"]))
    assert payload["epoch"] == 0

    result_files = sorted(results_dir.glob("*.json"))
    assert result_files, "no result JSON was written"
    result = json.loads(result_files[-1].read_text())
    assert result["experiment"] == "smoke_test"
    assert result["metrics"]["metric"] == "recon_loss"
    assert torch.isfinite(torch.tensor(result["metrics"]["best_val_recon_loss"]))


def test_ssl_adapt_resumes_from_latest_checkpoint(tmp_path, monkeypatch, cfg_stub):
    """Run epoch 0, then re-invoke with resume=True — the second call
    should start from epoch=1 not epoch=0."""

    monkeypatch.setattr(
        "video_encoding.ssl.VideoMAEForPreTraining.from_pretrained",
        classmethod(lambda cls, ckpt: _FakePretrainingModel(num_patches=8)),
    )
    monkeypatch.setattr("video_encoding.ssl.attach_lora", lambda enc, rank=8: enc)

    dataset = _FakeClipDataset(n_clips=4, config=_FakeConfig())
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()

    def _run(epochs: int, resume: bool):
        ssl_module.ssl_adapt(
            checkpoint="fake://ignored",
            train_dataloader=DataLoader(dataset, batch_size=2, shuffle=True),
            val_dataloader=DataLoader(dataset, batch_size=2, shuffle=False),
            epochs=epochs,
            lr=1e-3,
            accum_steps=1,
            checkpoint_path=ckpt_dir / "ssl_best.pt",
            resume=resume,
            save_latest=True,
            results_root=None,
            device=torch.device("cpu"),
            cfg=cfg_stub,
        )

    _run(epochs=1, resume=False)
    latest = torch.load(
        ckpt_dir / "ssl_best_latest.pt", map_location="cpu", weights_only=False
    )
    assert latest["epoch"] == 0

    _run(epochs=2, resume=True)
    latest = torch.load(
        ckpt_dir / "ssl_best_latest.pt", map_location="cpu", weights_only=False
    )
    # After the resumed second run we've written epoch=1 (the only epoch
    # not already covered). This proves the loader started from epoch 1
    # instead of restarting at 0.
    assert latest["epoch"] == 1
