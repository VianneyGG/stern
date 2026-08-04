from __future__ import annotations

import io
import pickle
import tarfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from video_encoding.data.wds_frames import (
    _member_offsets,
    read_shard_frames_at,
    read_shard_frames_indexed,
)


def _make_shard(tmp_path: Path, clips: dict[str, list[np.ndarray]]) -> Path:
    shard_path = tmp_path / "shard-000000.tar"
    with tarfile.open(shard_path, "w") as tar:
        for clip_key, frames in clips.items():
            jpegs = []
            for frame in frames:
                buf = io.BytesIO()
                Image.fromarray(frame).save(buf, format="JPEG")
                jpegs.append(buf.getvalue())
            payload = pickle.dumps(jpegs)
            info = tarfile.TarInfo(name=f"{clip_key}.pickle")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    return shard_path


def test_indexed_reader_matches_scanning_reader(tmp_path):
    clip_a_frames = [np.full((8, 8, 3), i, dtype=np.uint8) for i in range(4)]
    clip_b_frames = [np.full((8, 8, 3), 100 + i, dtype=np.uint8) for i in range(4)]
    shard_path = _make_shard(
        tmp_path, {"id1/vid1/clip1": clip_a_frames, "id2/vid2/clip2": clip_b_frames}
    )

    scanned = read_shard_frames_at(shard_path, "id2/vid2/clip2", [0, 2, 3], size=8)
    indexed = read_shard_frames_indexed(shard_path, "id2/vid2/clip2", [0, 2, 3], size=8)

    assert len(scanned) == len(indexed) == 3
    for a, b in zip(scanned, indexed):
        assert np.array_equal(a, b)


def test_indexed_reader_first_clip_in_shard(tmp_path):
    clip_a_frames = [np.full((4, 4, 3), i, dtype=np.uint8) for i in range(2)]
    shard_path = _make_shard(tmp_path, {"only_clip": clip_a_frames})

    frames = read_shard_frames_indexed(shard_path, "only_clip", [0, 1], size=4)

    assert len(frames) == 2
    assert np.array_equal(frames[0], np.full((4, 4, 3), 0, dtype=np.uint8))
    assert np.array_equal(frames[1], np.full((4, 4, 3), 1, dtype=np.uint8))


def test_member_offsets_raises_keyerror_for_missing_clip(tmp_path):
    shard_path = _make_shard(
        tmp_path, {"present": [np.zeros((2, 2, 3), dtype=np.uint8)]}
    )

    with pytest.raises(KeyError):
        read_shard_frames_indexed(shard_path, "absent", [0], size=2)


def test_member_offsets_is_cached_per_shard_path(tmp_path):
    shard_path = _make_shard(tmp_path, {"clip": [np.zeros((2, 2, 3), dtype=np.uint8)]})
    _member_offsets.cache_clear()

    read_shard_frames_indexed(shard_path, "clip", [0], size=2)
    read_shard_frames_indexed(shard_path, "clip", [0], size=2)

    info = _member_offsets.cache_info()
    assert info.hits == 1
    assert info.misses == 1
