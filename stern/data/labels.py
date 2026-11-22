"""Per-frame VA label CSVs, the `ManifestRecord.label_path` contract.

Each label file has two columns, `valence,arousal`, and one row per annotated
frame at the dataset's annotation rate. There is no timestamp column: the row
index is the frame number.

`load_per_frame_va` keeps the full `(F, 2)` trajectory for the trainer, which
windows frames. `load_clip_va_label` mean-pools a clip to one pair.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from stern.data.harmonise import harmonise_va


def load_clip_va_label(
    label_path: Path, annotation_hz: float, va_range: tuple[float, float]
) -> tuple[float, float]:
    """Read a per-frame VA CSV, harmonise to [-1, 1], and mean-pool to one
    (valence, arousal) pair for the clip.

    `annotation_hz` is unused. It matches `harmonise_rate`'s signature, and
    clip-level pooling has no resampling target.
    """
    with Path(label_path).open(newline="") as f:
        reader = csv.DictReader(f)
        valence: list[float] = []
        arousal: list[float] = []
        for row in reader:
            valence.append(float(row["valence"]))
            arousal.append(float(row["arousal"]))

    if not valence:
        raise ValueError(f"label file {label_path} has no rows")

    harmonised_valence = harmonise_va(np.array(valence), va_range)
    harmonised_arousal = harmonise_va(np.array(arousal), va_range)
    return float(harmonised_valence.mean()), float(harmonised_arousal.mean())


def load_per_frame_va(label_path: Path, va_range: tuple[float, float]) -> np.ndarray:
    """Read a per-frame VA CSV as an `(F, 2)` array of `[valence, arousal]`,
    harmonised to [-1, 1]. Does not mean-pool.

    The trainer takes a per-window mean itself, so it needs every frame.
    """
    with Path(label_path).open(newline="") as f:
        reader = csv.DictReader(f)
        rows = [(float(r["valence"]), float(r["arousal"])) for r in reader]

    if not rows:
        raise ValueError(f"label file {label_path} has no rows")

    arr = np.array(rows, dtype=np.float64)
    valence = harmonise_va(arr[:, 0], va_range)
    arousal = harmonise_va(arr[:, 1], va_range)
    return np.stack([valence, arousal], axis=1)


def write_label_csv(path: Path, rows: list[tuple[float, float]]) -> None:
    """Write the `valence,arousal` per-frame CSV that the readers above expect.

    Every ingestion script in stern/prep uses this, so the write side lives
    next to the read side.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["valence", "arousal"])
        writer.writerows(rows)
