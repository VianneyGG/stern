from pathlib import Path

import h5py
import numpy as np
import pytest

from cache.store import clip_key, read_embeddings, write_embeddings


def test_clip_key_converts_path_separators_and_strips_extension():
    assert clip_key("train/id00000/vid001/001.mp4") == "train__id00000__vid001__001"


def test_clip_key_handles_windows_style_backslashes():
    assert clip_key("train\\id00000\\vid001\\001.mp4") == "train__id00000__vid001__001"


def test_write_then_read_embeddings_roundtrip(tmp_path: Path):
    path = tmp_path / "cache.h5"
    embeddings = np.random.default_rng(0).standard_normal((8, 768)).astype(np.float32)

    write_embeddings(path, "train__id00000__vid001__001", embeddings)
    result = read_embeddings(path, "train__id00000__vid001__001")

    assert result.dtype == np.float32
    np.testing.assert_allclose(result, embeddings, atol=1e-2)  # fp16 round-trip tolerance


def test_write_embeddings_stores_as_float16_on_disk(tmp_path: Path):
    path = tmp_path / "cache.h5"
    embeddings = np.ones((8, 768), dtype=np.float32)

    write_embeddings(path, "clip_a", embeddings)

    with h5py.File(path, "r") as f:
        assert f["clip_a"].dtype == np.float16


def test_write_embeddings_appends_multiple_clips_to_same_file(tmp_path: Path):
    path = tmp_path / "cache.h5"
    write_embeddings(path, "clip_a", np.ones((8, 768), dtype=np.float32))
    write_embeddings(path, "clip_b", np.zeros((8, 768), dtype=np.float32))

    a = read_embeddings(path, "clip_a")
    b = read_embeddings(path, "clip_b")

    assert a.shape == (8, 768) and np.all(a == 1)
    assert b.shape == (8, 768) and np.all(b == 0)
