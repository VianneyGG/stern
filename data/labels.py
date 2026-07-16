"""Per-clip VA label loading (docs/superpowers/specs/2026-07-01-training-pipeline-design.md).

Defines the `ManifestRecord.label_path` CSV contract: two columns
`valence,arousal`, one row per annotated frame at the dataset's
`annotation_hz` (matches Aff-Wild2's actual per-frame annotation export --
no explicit timestamp column, row index implies frame number).

Two readers: `load_per_frame_va` keeps the full per-frame `(F, 2)` trajectory
(used by the end-to-end trainer, which windows frames and takes a per-window
mean), while `load_clip_va_label` mean-pools a whole clip to a single
(valence, arousal) pair (legacy clip-level granularity).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from data.harmonise import harmonise_va


def load_clip_va_label(
    label_path: Path, annotation_hz: float, va_range: tuple[float, float]
) -> tuple[float, float]:
    """Read a per-frame VA CSV, harmonise to [-1, 1], mean-pool to one
    (valence, arousal) pair for the whole clip.

    `annotation_hz` is accepted for signature symmetry with
    `harmonise_rate` but not used yet -- there's no resampling target rate
    at clip-label granularity, only the mean-pool.
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
    """Read a per-frame VA CSV and return an `(F, 2)` array of `[valence,
    arousal]` harmonised to `[-1, 1]` -- WITHOUT mean-pooling.

    Used by the end-to-end trainer, which windows frames and takes a per-window
    mean, so it needs the full per-frame trajectory rather than one clip label.
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
    """Write the `valence,arousal` per-frame CSV this module's readers expect.

    Shared by every dataset ingestion script (build_afewva_manifest.py,
    build_veatic_manifest.py, ...) so the write side of the contract lives in
    one place next to the read side, instead of each script re-implementing
    the same few lines.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["valence", "arousal"])
        writer.writerows(rows)
