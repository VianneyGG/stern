"""Unit tests for SSLWindowDataset (video_encoding/data/ssl_windows.py).

Covers the bug found during Workstream A/B review: the cache directory was
computed as `VOXMINI_FRAMES_ROOT/<identity_id>/<clip_path>.stem`, which (a)
crashed with AttributeError on `clip_path` (a str, not a Path) and (b) even
fixed, duplicated identity_id and dropped the video_id directory relative to
the layout `cache_voxmini_frames.py` actually writes
(`frames_root / Path(clip_path).with_suffix("")`, where clip_path already
starts with identity_id). Also covers the missing window-start offset: every
window of a clip was reading frames [0, num_frames) regardless of its
intended start index.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from data.manifest import ManifestRecord
from video_encoding.data.loaders import SSLDataset
from video_encoding.data.ssl_windows import SSLWindowDataset


def _write_fake_frames(clip_dir: Path, n: int, size: int = 224) -> None:
    clip_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.fromarray(np.zeros((size, size, 3), dtype=np.uint8)).save(
            clip_dir / f"{i:04d}.jpg"
        )


def _ssl_dataset(root: Path, clip_path: str) -> SSLDataset:
    return SSLDataset(
        records_by_identity={
            "id00012": [
                ManifestRecord(
                    identity_id="id00012",
                    clip_path=clip_path,
                    duration_s=5.0,
                    role="ssl",
                )
            ]
        },
        window_cfg=OmegaConf.create({}),
        root=root,
    )


def test_cache_dir_matches_cache_voxmini_frames_layout(tmp_path, monkeypatch):
    """The reader must resolve the same directory the writer
    (cache_voxmini_frames.transcode_clip) fills in: frames_root /
    Path(clip_path).with_suffix(''), NOT frames_root / identity_id / stem."""
    frames_root = tmp_path / "frames"
    clip_path = "id00012/vid1/001.mp4"
    # Matches cache_voxmini_frames.cache_frames: out_dir = frames_root / Path(clip_path).with_suffix("")
    _write_fake_frames(frames_root / "id00012/vid1/001", n=16)

    monkeypatch.setenv("VOXMINI_FRAMES_ROOT", str(frames_root))
    dataset = _ssl_dataset(tmp_path / "raw", clip_path)

    ds = SSLWindowDataset(dataset, num_frames=16, image_size=224, overlap=0.0)

    assert len(ds) == 1
    pixel_values = ds[0]
    assert pixel_values.shape == (3, 16, 224, 224)


def test_multiple_windows_read_distinct_frame_ranges(tmp_path, monkeypatch):
    """Two overlapping windows over a longer cached clip must read different
    frame slices, not both collapse to frames [0, num_frames)."""
    frames_root = tmp_path / "frames"
    clip_path = "id00012/vid1/001.mp4"
    _write_fake_frames(frames_root / "id00012/vid1/001", n=24)

    monkeypatch.setenv("VOXMINI_FRAMES_ROOT", str(frames_root))
    dataset = _ssl_dataset(tmp_path / "raw", clip_path)

    ds = SSLWindowDataset(dataset, num_frames=16, image_size=224, overlap=0.5)

    # window_starts(24, 16, 0.5) -> step=8 -> starts at [0, 8]
    assert len(ds) == 2
    assert ds._items[0][1] == list(range(0, 16))
    assert ds._items[1][1] == list(range(8, 24))


def test_falls_back_to_live_decode_when_cache_root_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("VOXMINI_FRAMES_ROOT", raising=False)
    clip_path = "id00012/vid1/001.mp4"
    dataset = _ssl_dataset(tmp_path / "raw", clip_path)

    fake_frames = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(16)]
    with mock.patch(
        "video_encoding.data.ssl_windows.decode_frames", return_value=fake_frames
    ) as mock_decode:
        ds = SSLWindowDataset(dataset, num_frames=16, image_size=224, overlap=0.5)
        assert len(ds) == 1  # one window per clip in the live-decode fallback
        pixel_values = ds[0]

    mock_decode.assert_called_once_with(
        tmp_path / "raw" / clip_path, 16
    )
    assert pixel_values.shape == (3, 16, 224, 224)


def test_skips_clip_with_no_cached_frames(tmp_path, monkeypatch):
    frames_root = tmp_path / "frames"
    frames_root.mkdir()
    monkeypatch.setenv("VOXMINI_FRAMES_ROOT", str(frames_root))
    dataset = _ssl_dataset(tmp_path / "raw", "id00012/vid1/001.mp4")

    with pytest.raises(ValueError, match="no usable SSL clips"):
        SSLWindowDataset(dataset, num_frames=16, image_size=224, overlap=0.0)
