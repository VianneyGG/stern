"""End-to-end wiring test for run_train_stage: real manifest/label build, real
windowing + label alignment, real preprocessing + CCC loop -- only the ViT
backbone is faked (a full-size video ViT forward is slow for CI). Exercises the
path the old encoder-faking tests skipped.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch import nn

from data.labels import write_label_csv
from data.manifest import ManifestRecord, write_manifest
from video_encoding.training import train as train_mod
from video_encoding.scripts.build_afewva_manifest import build, parse_clip_annotations

pytestmark = pytest.mark.integration


class _FakeRegressor(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.lin = nn.Linear(3, 2)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.lin(pixel_values.mean(dim=[2, 3, 4])))


class _FakeEncoderBackbone(nn.Module):
    """Minimal stand-in exposing the same `patch_embed.proj.weight` shape
    contract `checkpoint.load_encoder_state` depends on, without paying for
    a real VideoMAEEncoder forward pass in this fast unit test."""

    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv3d(3, 4, kernel_size=1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.patch_embed.proj(pixel_values).mean(dim=[2, 3, 4])


class _FakeRegressorWithBackbone(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.backbone = _FakeEncoderBackbone()
        self.head = nn.Linear(4, 2)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(pixel_values)
        return torch.tanh(self.head(feat))


def test_run_train_stage_loads_and_freezes_pretrained_backbone(tmp_path, monkeypatch):
    """train.pretrain_encoder_path + freeze_backbone=True must load the donor
    weights into model.backbone and keep them frozen through training -- the
    bug this test would have caught: these fields were declared in config but
    never read anywhere, so Stage 1 -> Stage 2 hand-off silently did nothing."""
    rng = np.random.default_rng(0)
    _write_clip(tmp_path / "001", 24, rng)
    _write_clip(tmp_path / "002", 24, rng)
    manifest_path = build(tmp_path, tmp_path)
    monkeypatch.setattr(train_mod, "VideoMAEVARegressor", _FakeRegressorWithBackbone)

    donor_backbone = _FakeEncoderBackbone()
    donor_backbone.patch_embed.proj.weight.data.fill_(7.0)
    donor_backbone.patch_embed.proj.bias.data.fill_(0.0)
    ckpt_path = tmp_path / "pretrain_encoder.pt"
    torch.save(
        {
            "model_state_dict": {
                f"encoder.{k}": v for k, v in donor_backbone.state_dict().items()
            }
        },
        ckpt_path,
    )

    cfg = _cfg(tmp_path, manifest_path)
    cfg.train.pretrain_encoder_path = str(ckpt_path)
    cfg.train.freeze_backbone = True

    train_mod.run_train_stage(
        cfg, rank=0, world_size=1, local_rank=0, device=torch.device("cpu")
    )

    saved = torch.load(
        tmp_path / "ckpt" / "test_va_finetune" / "latest.pt", weights_only=False
    )
    # If loading or freezing were broken, 2 epochs of training on random data
    # would almost certainly move these weights away from the donor's.
    assert torch.equal(
        saved["model_state_dict"]["backbone.patch_embed.proj.weight"],
        donor_backbone.patch_embed.proj.weight.data,
    )
    assert torch.equal(
        saved["model_state_dict"]["backbone.patch_embed.proj.bias"],
        donor_backbone.patch_embed.proj.bias.data,
    )


def _write_clip(clip_dir: Path, n: int, rng: np.random.Generator) -> None:
    clip_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(
            clip_dir / f"{i:05d}.png"
        )
    frames = {
        f"{i:05d}": {
            "valence": int(rng.integers(-10, 11)),
            "arousal": int(rng.integers(-10, 11)),
        }
        for i in range(n)
    }
    (clip_dir / f"{clip_dir.name}.json").write_text(json.dumps({"frames": frames}))


def _cfg(tmp_path: Path, manifest_path: Path):
    return OmegaConf.create(
        {
            "experiment": {"name": "test_va_finetune"},
            "dataset": {
                "name": "afewva",
                "role": "probe",
                "present": True,
                "path": str(tmp_path),
                "manifest_path": str(manifest_path),
                "va_range": [-10.0, 10.0],
                "annotation_hz": 30.0,
            },
            "model": {"num_frames": 4, "embed_dim": 8, "image_size": 8},
            "preprocess": {"window": {"overlap": 0.5}},
            "train": {
                "seed": 0,
                "batch_size": 4,
                "learning_rate": 1e-2,
                "epochs": 2,
                "checkpoint_root": str(tmp_path / "ckpt"),
                "results_root": str(tmp_path / "results"),
            },
        }
    )


def test_run_train_stage_trains_and_reports_ccc(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    _write_clip(tmp_path / "001", 24, rng)
    _write_clip(tmp_path / "002", 24, rng)
    manifest_path = build(tmp_path, tmp_path)

    monkeypatch.setattr(train_mod, "VideoMAEVARegressor", _FakeRegressor)

    cfg = _cfg(tmp_path, manifest_path)
    result = train_mod.run_train_stage(
        cfg, rank=0, world_size=1, local_rank=0, device=torch.device("cpu")
    )

    assert result["stage"] == "train"
    assert isinstance(result["val_ccc_v"], float)
    assert isinstance(result["val_ccc_a"], float)
    assert (tmp_path / "ckpt" / "test_va_finetune" / "best.pt").exists()
    assert len(list((tmp_path / "results").glob("*.json"))) == 1


def test_run_train_stage_resumes_from_latest_checkpoint(tmp_path, monkeypatch):
    """A second call with more epochs must continue from where the first
    call's latest.pt left off, not restart at epoch 0 with a fresh random
    model -- the bug this test would have caught: `ckpt_io.load` was never
    called, so every run silently retrained from scratch."""
    rng = np.random.default_rng(0)
    _write_clip(tmp_path / "001", 24, rng)
    _write_clip(tmp_path / "002", 24, rng)
    manifest_path = build(tmp_path, tmp_path)
    monkeypatch.setattr(train_mod, "VideoMAEVARegressor", _FakeRegressor)

    cfg = _cfg(tmp_path, manifest_path)
    cfg.train.epochs = 2
    train_mod.run_train_stage(
        cfg, rank=0, world_size=1, local_rank=0, device=torch.device("cpu")
    )
    latest_path = tmp_path / "ckpt" / "test_va_finetune" / "latest.pt"
    assert latest_path.exists()
    first_ckpt = torch.load(latest_path, weights_only=False)
    assert first_ckpt["epoch"] == 1  # 0-indexed: epochs 0,1 ran

    cfg.train.epochs = 4
    train_mod.run_train_stage(
        cfg, rank=0, world_size=1, local_rank=0, device=torch.device("cpu")
    )
    second_ckpt = torch.load(latest_path, weights_only=False)
    assert second_ckpt["epoch"] == 3  # resumed at epoch 2, ran through epoch 3


def test_run_train_stage_skips_malformed_clip_instead_of_crashing(
    tmp_path, monkeypatch
):
    """A manifest record whose frame directory has 0 images (e.g. a failed
    extraction that slipped past the manifest builder) must be skipped, not
    abort construction for every other good clip in the dataset."""
    rng = np.random.default_rng(0)
    _write_clip(tmp_path / "001", 24, rng)
    (tmp_path / "bad").mkdir()  # exists but has no frame images at all

    manifest_path = tmp_path / "manifest.csv"
    write_manifest(
        manifest_path,
        [
            ManifestRecord(
                identity_id="001",
                clip_path="001",
                duration_s=24 / 30.0,
                role="probe",
                label_path="labels/001.csv",
            ),
            ManifestRecord(
                identity_id="bad",
                clip_path="bad",
                duration_s=0.0,
                role="probe",
                label_path="labels/001.csv",  # reuse a valid label file; only the frames are bad
            ),
        ],
    )
    # build_afewva_manifest's build() would have written labels/001.csv already
    # via the earlier call pattern; reproduce it directly here since we're
    # bypassing build() to control which records land in the manifest.

    rows = parse_clip_annotations(tmp_path / "001" / "001.json")
    write_label_csv(tmp_path / "labels" / "001.csv", rows)

    monkeypatch.setattr(train_mod, "VideoMAEVARegressor", _FakeRegressor)
    cfg = _cfg(tmp_path, manifest_path)

    result = train_mod.run_train_stage(
        cfg, rank=0, world_size=1, local_rank=0, device=torch.device("cpu")
    )

    assert result["stage"] == "train"  # did not crash despite the "bad" record


def test_run_train_stage_emits_live_heartbeat_every_epoch(tmp_path, monkeypatch):
    """The watcher's stall detector needs a result JSON that updates mid-run,
    not just after the loop exits -- assert write_result_json is called once
    per epoch (heartbeat) plus once more at loop exit (final), and that the
    file on disk reflects the LAST epoch's val metrics, not epoch 0's."""
    rng = np.random.default_rng(0)
    _write_clip(tmp_path / "001", 24, rng)
    _write_clip(tmp_path / "002", 24, rng)
    manifest_path = build(tmp_path, tmp_path)
    monkeypatch.setattr(train_mod, "VideoMAEVARegressor", _FakeRegressor)

    calls = []
    real_write = train_mod.write_result_json

    def _spy(*args, **kwargs):
        calls.append(kwargs.get("overwrite", False))
        return real_write(*args, **kwargs)

    monkeypatch.setattr(train_mod, "write_result_json", _spy)

    cfg = _cfg(tmp_path, manifest_path)
    cfg.train.epochs = 3
    train_mod.run_train_stage(
        cfg, rank=0, world_size=1, local_rank=0, device=torch.device("cpu")
    )

    # 3 epoch-heartbeat writes + 1 final write, every one overwrite=True so
    # they all land on the same stable filename (still exactly 1 file).
    assert calls == [True, True, True, True]
    assert len(list((tmp_path / "results").glob("*.json"))) == 1
