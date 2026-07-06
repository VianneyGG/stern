"""Per-clip VA label loading (docs/superpowers/specs/2026-07-01-training-pipeline-design.md).

Defines the `ManifestRecord.label_path` CSV contract: two columns
`valence,arousal`, one row per annotated frame at the dataset's
`annotation_hz` (matches Aff-Wild2's actual per-frame annotation export --
no explicit timestamp column, row index implies frame number).

`train_probe`/`LinearVAHead` (src/probe/__init__.py) operate at one
embedding per clip, matching `VideoMaeV2Encoder.encode_clip`'s single
mean-pooled-over-time output -- so a clip's per-frame annotations are
harmonised then mean-pooled down to one (valence, arousal) pair here, not
kept as a per-frame sequence.
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
