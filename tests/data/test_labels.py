import csv
from pathlib import Path

import pytest

from stern.data.labels import load_clip_va_label


def _write_label_csv(path: Path, rows: list[tuple[float, float]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["valence", "arousal"])
        writer.writerows(rows)


def test_load_clip_va_label_mean_pools_known_values(tmp_path: Path):
    label_path = tmp_path / "clip.csv"
    _write_label_csv(label_path, [(0.0, 0.0), (1.0, -1.0), (0.5, 0.5)])

    valence, arousal = load_clip_va_label(
        label_path, annotation_hz=30.0, va_range=(-1.0, 1.0)
    )

    assert valence == pytest.approx(0.5)
    assert arousal == pytest.approx(-1.0 / 6.0)


def test_load_clip_va_label_rescales_source_range(tmp_path: Path):
    label_path = tmp_path / "clip.csv"
    # source range [0, 1]: 0.0 -> -1.0, 1.0 -> 1.0 in the harmonised [-1, 1] range.
    _write_label_csv(label_path, [(0.0, 1.0), (1.0, 1.0)])

    valence, arousal = load_clip_va_label(
        label_path, annotation_hz=30.0, va_range=(0.0, 1.0)
    )

    assert valence == pytest.approx(0.0)
    assert arousal == pytest.approx(1.0)


def test_load_clip_va_label_rejects_out_of_range_values(tmp_path: Path):
    label_path = tmp_path / "clip.csv"
    _write_label_csv(label_path, [(2.0, 0.0)])  # outside declared va_range

    with pytest.raises(ValueError, match="fall outside"):
        load_clip_va_label(label_path, annotation_hz=30.0, va_range=(-1.0, 1.0))


def test_load_clip_va_label_rejects_empty_file(tmp_path: Path):
    label_path = tmp_path / "clip.csv"
    _write_label_csv(label_path, [])

    with pytest.raises(ValueError, match="no rows"):
        load_clip_va_label(label_path, annotation_hz=30.0, va_range=(-1.0, 1.0))
