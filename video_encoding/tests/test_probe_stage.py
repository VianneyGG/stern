"""Integration test for run_probe_stage's wiring: real manifest/label CSVs,
real synthetic-MP4 decode, real embedding cache (HDF5), real train_probe --
only the encoder is faked (same reasoning as test_ssl_smoke.py: avoids a
~350MB HF download for a loop that's model-agnostic once it has an
embedding vector to work with).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import av
import numpy as np
import torch
from omegaconf import OmegaConf

from data.manifest import ManifestRecord, write_manifest
from video_encoding.probe.train import run_probe_stage


class _FakeEncoder:
    """Stands in for VideoMaeV2Encoder. Tracks call count (shared across
    instances via a class attribute, since run_probe_stage constructs a
    fresh instance per call) so tests can assert the embedding cache is
    actually reused on a second run, not just present."""

    call_count = 0

    def __init__(self, model_cfg):
        self.embed_dim = model_cfg.embed_dim

    def encode_clip(self, frames) -> np.ndarray:
        _FakeEncoder.call_count += 1
        return np.full(self.embed_dim, fill_value=float(len(frames)), dtype=np.float32)


def _write_synthetic_mp4(
    path: Path, num_frames: int = 10, width: int = 32, height: int = 32
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), mode="w")
    stream = container.add_stream("mpeg4", rate=25)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    for i in range(num_frames):
        arr = np.full((height, width, 3), fill_value=i % 256, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def _write_label_csv(path: Path, rng: np.random.Generator) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["valence", "arousal"])
        for _ in range(5):
            writer.writerow(rng.uniform(-1.0, 1.0, size=2))


def _build_probe_dataset(tmp_path: Path, n_clips: int = 6) -> Path:
    rng = np.random.default_rng(0)
    records = []
    for i in range(n_clips):
        clip_path = f"clips/{i:02d}.mp4"
        label_path = f"labels/{i:02d}.csv"
        _write_synthetic_mp4(tmp_path / clip_path)
        _write_label_csv(tmp_path / label_path, rng)
        records.append(
            ManifestRecord(
                identity_id=f"id{i}",
                clip_path=clip_path,
                duration_s=1.0,
                role="probe",
                label_path=label_path,
            )
        )
    manifest_path = tmp_path / "manifest.csv"
    write_manifest(manifest_path, records)
    return manifest_path


def _cfg_stub(tmp_path: Path, manifest_path: Path) -> OmegaConf:
    return OmegaConf.create(
        {
            "experiment": {"name": "test_probe_stage"},
            "dataset": {
                "name": "affwild2",
                "role": "probe",
                "present": True,
                "path": str(tmp_path),
                "manifest_path": str(manifest_path),
                "va_range": [-1.0, 1.0],
                "annotation_hz": 30.0,
            },
            "model": {
                "checkpoint": "fake/checkpoint",
                "num_frames": 4,
                "embed_dim": 8,
            },
            "preprocess": {"window": {"len_s": 2.0, "overlap": 0.5}},
            "train": {
                "seed": 0,
                "checkpoint_root": str(tmp_path / "ckpt"),
                "results_root": str(tmp_path / "results"),
                "probe": {"epochs": 5, "learning_rate": 1e-2},
            },
        }
    )


def test_run_probe_stage_populates_cache_and_trains_head(tmp_path, monkeypatch):
    _FakeEncoder.call_count = 0
    monkeypatch.setattr("video_encoding.probe.train.VideoMaeV2Encoder", _FakeEncoder)

    manifest_path = _build_probe_dataset(tmp_path, n_clips=6)
    cfg = _cfg_stub(tmp_path, manifest_path)

    result = run_probe_stage(
        cfg, rank=0, world_size=1, local_rank=0, device=torch.device("cpu")
    )

    assert _FakeEncoder.call_count == 6
    assert "val_ccc_v" in result and isinstance(result["val_ccc_v"], float)
    assert "val_ccc_a" in result and isinstance(result["val_ccc_a"], float)

    checkpoint_dir = tmp_path / "ckpt" / "test_probe_stage"
    assert (checkpoint_dir / "embeddings.h5").exists()
    assert (checkpoint_dir / "probe_head.pt").exists()

    result_jsons = list((tmp_path / "results").glob("*.json"))
    assert len(result_jsons) == 1
    payload = json.loads(result_jsons[0].read_text())
    assert payload["metrics"]["val_ccc_v"] == result["val_ccc_v"]


def test_run_probe_stage_reuses_cache_on_second_call(tmp_path, monkeypatch):
    _FakeEncoder.call_count = 0
    monkeypatch.setattr("video_encoding.probe.train.VideoMaeV2Encoder", _FakeEncoder)

    manifest_path = _build_probe_dataset(tmp_path, n_clips=6)
    cfg = _cfg_stub(tmp_path, manifest_path)

    run_probe_stage(cfg, rank=0, world_size=1, local_rank=0, device=torch.device("cpu"))
    assert _FakeEncoder.call_count == 6

    run_probe_stage(cfg, rank=0, world_size=1, local_rank=0, device=torch.device("cpu"))
    assert _FakeEncoder.call_count == 6  # second run hits the cache, no re-encoding


def test_run_probe_stage_non_zero_rank_returns_without_side_effects(
    tmp_path, monkeypatch
):
    _FakeEncoder.call_count = 0
    monkeypatch.setattr("video_encoding.probe.train.VideoMaeV2Encoder", _FakeEncoder)

    manifest_path = _build_probe_dataset(tmp_path, n_clips=6)
    cfg = _cfg_stub(tmp_path, manifest_path)

    result = run_probe_stage(
        cfg, rank=1, world_size=2, local_rank=1, device=torch.device("cpu")
    )

    assert result == {"stage": "probe"}
    assert _FakeEncoder.call_count == 0
    assert not (tmp_path / "ckpt").exists()
