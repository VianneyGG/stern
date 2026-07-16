# video_encoding/tests/test_loaders.py
from pathlib import Path

import av
import numpy as np
import pytest
from omegaconf import OmegaConf

from data.manifest import ManifestRecord, write_manifest
from video_encoding.data.loaders import (
    DatasetNotAcquiredError,
    build_eval_loader,
    build_probe_loader,
    build_ssl_loader,
)


def _window_cfg():
    return OmegaConf.create({"len_s": 2.0, "overlap": 0.5})


def _write_synthetic_mp4(path: Path, num_frames: int = 10, width: int = 32, height: int = 32) -> None:
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


def test_build_ssl_loader_rejects_wrong_role():
    dataset_cfg = OmegaConf.create(
        {"name": "voxmini", "role": "probe", "present": True}
    )
    with pytest.raises(ValueError, match="requires role='ssl'"):
        build_ssl_loader(dataset_cfg, _window_cfg())


def test_build_ssl_loader_rejects_not_acquired():
    dataset_cfg = OmegaConf.create({"name": "voxmini", "role": "ssl", "present": False})
    with pytest.raises(DatasetNotAcquiredError):
        build_ssl_loader(dataset_cfg, _window_cfg())


def test_build_ssl_loader_groups_records_by_identity(tmp_path: Path):
    manifest_path = tmp_path / "manifest.csv"
    write_manifest(
        manifest_path,
        [
            ManifestRecord(
                identity_id="id1", clip_path="id1/a.mp4", duration_s=3.0, role="ssl"
            ),
            ManifestRecord(
                identity_id="id1", clip_path="id1/b.mp4", duration_s=4.0, role="ssl"
            ),
            ManifestRecord(
                identity_id="id2", clip_path="id2/a.mp4", duration_s=2.0, role="ssl"
            ),
        ],
    )
    dataset_cfg = OmegaConf.create(
        {
            "name": "voxmini",
            "role": "ssl",
            "present": True,
            "path": str(tmp_path),
            "manifest_path": str(manifest_path),
        }
    )

    dataset = build_ssl_loader(dataset_cfg, _window_cfg())

    assert set(dataset.records_by_identity) == {"id1", "id2"}
    assert len(dataset.records_by_identity["id1"]) == 2


def test_ssl_dataset_load_frames_decodes_real_clip(tmp_path: Path):
    manifest_path = tmp_path / "manifest.csv"
    write_manifest(
        manifest_path,
        [
            ManifestRecord(
                identity_id="id1", clip_path="id1/a.mp4", duration_s=3.0, role="ssl"
            )
        ],
    )
    _write_synthetic_mp4(tmp_path / "id1" / "a.mp4")
    dataset_cfg = OmegaConf.create(
        {
            "name": "voxmini",
            "role": "ssl",
            "present": True,
            "path": str(tmp_path),
            "manifest_path": str(manifest_path),
        }
    )
    dataset = build_ssl_loader(dataset_cfg, _window_cfg())

    frames = dataset.load_frames("id1", "id1/a.mp4", num_frames=4)

    assert len(frames) == 4
    assert frames[0].shape == (224, 224, 3)


def test_build_probe_loader_returns_labelled_records(tmp_path: Path):
    manifest_path = tmp_path / "manifest.csv"
    write_manifest(
        manifest_path,
        [
            ManifestRecord(
                identity_id="vid1",
                clip_path="vid1.mp4",
                duration_s=120.0,
                role="probe",
                label_path="vid1_va.csv",
            ),
        ],
    )
    dataset_cfg = OmegaConf.create(
        {
            "name": "affwild2",
            "role": "probe",
            "present": True,
            "path": str(tmp_path),
            "manifest_path": str(manifest_path),
        }
    )

    dataset = build_probe_loader(dataset_cfg, _window_cfg())

    assert dataset.records[0].label_path == "vid1_va.csv"


def test_labelled_dataset_load_frames_decodes_real_clip(tmp_path: Path):
    manifest_path = tmp_path / "manifest.csv"
    write_manifest(
        manifest_path,
        [
            ManifestRecord(
                identity_id="vid1",
                clip_path="vid1.mp4",
                duration_s=3.0,
                role="probe",
                label_path="vid1_va.csv",
            ),
        ],
    )
    _write_synthetic_mp4(tmp_path / "vid1.mp4")
    dataset_cfg = OmegaConf.create(
        {
            "name": "affwild2",
            "role": "probe",
            "present": True,
            "path": str(tmp_path),
            "manifest_path": str(manifest_path),
        }
    )
    dataset = build_probe_loader(dataset_cfg, _window_cfg())

    frames = dataset.load_frames("vid1.mp4", num_frames=4)

    assert len(frames) == 4
    assert frames[0].shape == (224, 224, 3)


def test_build_eval_loader_rejects_wrong_role():
    dataset_cfg = OmegaConf.create({"name": "sewa", "role": "ssl", "present": True})
    with pytest.raises(ValueError, match="requires role='eval'"):
        build_eval_loader(dataset_cfg, _window_cfg())
