"""VEATIC ingest: synthetic-fixture test for the manifest/label builder.

Reproduces the real VEATIC layout confirmed from the repo's own dataset.py /
video_frame.py: unpadded frame filenames (`0.jpg, 1.jpg, ..., 10.jpg`) and two
headerless 2-column rating CSVs per video (`{id}_valence.csv`,
`{id}_arousal.csv`, col 1 = value). This also regression-tests
`list_frame_files`' numeric sort -- an 11-frame clip forces "10.jpg" past
"2.jpg" under a naive lexicographic sort.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from stern.data.labels import load_per_frame_va
from stern.data.manifest import read_manifest
from stern.data.frame_decoder import list_frame_files
from stern.prep.build_veatic_manifest import build, load_video_ratings


def _write_video(
    veatic_root: Path, video_id: str, valence: list[float], arousal: list[float]
) -> None:
    frame_dir = veatic_root / "frame" / video_id
    frame_dir.mkdir(parents=True, exist_ok=True)
    for i in range(len(valence)):
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(
            frame_dir / f"{i}.jpg"  # unpadded, like the real video_frame.py
        )
    ratings_dir = veatic_root / "rating_averaged"
    ratings_dir.mkdir(parents=True, exist_ok=True)
    _write_rating_csv(ratings_dir / f"{video_id}_valence.csv", valence)
    _write_rating_csv(ratings_dir / f"{video_id}_arousal.csv", arousal)


def _write_rating_csv(path: Path, values: list[float]) -> None:
    with path.open("w") as f:
        for i, v in enumerate(values):
            f.write(f"{i},{v}\n")  # headerless, 2-col, matching np.loadtxt(...)[:, 1]


def test_list_frame_files_sorts_numerically_not_lexicographically(tmp_path: Path):
    # 11 frames (0..10) forces "10.jpg" to sort before "2.jpg" lexicographically.
    _write_video(tmp_path, "001", [0.0] * 11, [0.0] * 11)

    files = list_frame_files(tmp_path / "frame" / "001")

    assert [p.stem for p in files] == [str(i) for i in range(11)]


def test_load_video_ratings_zips_valence_and_arousal_by_row(tmp_path: Path):
    _write_video(tmp_path, "001", [0.1, 0.2, 0.3], [-0.1, -0.2, -0.3])

    rows = load_video_ratings(tmp_path / "rating_averaged", "001")

    assert rows == [(0.1, -0.1), (0.2, -0.2), (0.3, -0.3)]


def test_load_video_ratings_handles_single_row_csv(tmp_path: Path):
    # np.loadtxt returns a 1D array for a single-row CSV; without
    # np.atleast_2d the [:, 1] column slice raises IndexError.
    _write_video(tmp_path, "001", [0.5], [-0.5])

    rows = load_video_ratings(tmp_path / "rating_averaged", "001")

    assert rows == [(0.5, -0.5)]


def test_load_video_ratings_raises_on_length_mismatch(tmp_path: Path):
    ratings_dir = tmp_path / "rating_averaged"
    ratings_dir.mkdir(parents=True)
    _write_rating_csv(ratings_dir / "001_valence.csv", [0.1, 0.2, 0.3])
    _write_rating_csv(ratings_dir / "001_arousal.csv", [-0.1, -0.2])  # one row short

    with pytest.raises(ValueError, match="valence has 3 rows, arousal has 2"):
        load_video_ratings(ratings_dir, "001")


def test_load_video_ratings_raises_on_non_contiguous_frame_index(tmp_path: Path):
    ratings_dir = tmp_path / "rating_averaged"
    ratings_dir.mkdir(parents=True)
    with (ratings_dir / "001_valence.csv").open("w") as f:
        f.write("0,0.1\n2,0.2\n")  # gap: frame index 1 is missing
    _write_rating_csv(ratings_dir / "001_arousal.csv", [-0.1, -0.2])

    with pytest.raises(ValueError, match="frame-index column"):
        load_video_ratings(ratings_dir, "001")


def test_build_emits_probe_manifest_and_harmonisable_labels(tmp_path: Path):
    _write_video(tmp_path, "001", [1.0, 0.0, -1.0], [-1.0, 0.5, 1.0])
    _write_video(tmp_path, "002", [0.2, -0.2], [0.4, -0.4])

    manifest_path = build(tmp_path, tmp_path)

    records = read_manifest(manifest_path, expected_role="probe")
    assert {r.identity_id for r in records} == {"001", "002"}

    rec001 = next(r for r in records if r.identity_id == "001")
    assert rec001.clip_path == "frame/001"
    assert rec001.label_path == "labels/001.csv"

    va = load_per_frame_va(tmp_path / rec001.label_path, (-1.0, 1.0))
    np.testing.assert_allclose(va[0], [1.0, -1.0])
    np.testing.assert_allclose(va[2], [-1.0, 1.0])


def test_build_raises_when_no_matching_videos(tmp_path: Path):
    (tmp_path / "frame" / "001").mkdir(parents=True)
    (tmp_path / "rating_averaged").mkdir(parents=True)
    try:
        build(tmp_path, tmp_path)
    except ValueError as exc:
        assert "no VEATIC videos" in str(exc)
    else:
        raise AssertionError("expected ValueError when no rating files match")
