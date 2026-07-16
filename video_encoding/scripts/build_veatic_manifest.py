#!/usr/bin/env python
"""Build the STERN manifest + per-clip VA label CSVs from an extracted VEATIC tree.

VEATIC (github.com/AlbusPeter/VEATIC) stores ratings as two separate,
**headerless, 2-column** CSVs per video -- confirmed from the repo's own
`dataset.py` (`np.loadtxt(path, delimiter=',')[:,1]`), not the README's
simplified tree:

    rating_averaged/${video_id}_valence.csv   # col 0 = frame index, col 1 = valence
    rating_averaged/${video_id}_arousal.csv   # col 0 = frame index, col 1 = arousal
    frame/${video_id}/${frame_index}.jpg      # unpadded, 0-indexed (video_frame.py)

This zips the two rating files (row-for-row -- valence/arousal share the same
per-frame indexing) into one `valence,arousal` label CSV under `--out-root`,
matching the `data/labels.py` contract, and writes `manifest.csv`
(role=probe). `conf/dataset/veatic.yaml` declares `va_range=[-1,1]`; this was
an educated guess, since confirmed against the real dataset (observed range
[-0.9146, 0.8893] across all 124 clips).

Run as (repo root, `data.manifest` needs the root on PYTHONPATH -- pytest sets
this via pyproject.toml's `pythonpath`, a bare script invocation does not):
    PYTHONPATH=. uv run python video_encoding/scripts/build_veatic_manifest.py \\
        --veatic-root $VEATIC_ROOT
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from data.labels import write_label_csv
from data.manifest import ManifestRecord, write_manifest
from video_encoding.data.frame_decoder import _IMAGE_EXTS


def load_video_ratings(ratings_dir: Path, video_id: str) -> list[tuple[float, float]]:
    """Read `{video_id}_valence.csv` + `{video_id}_arousal.csv` (headerless,
    2-column: col 0 = frame index, col 1 = value) and zip them into
    `[(valence, arousal), ...]`.

    Validates rather than silently tolerating: both files must have the same
    row count (a length mismatch means one file is corrupt/truncated, not
    something to silently truncate further), and each file's frame-index
    column must be the contiguous sequence 0..N-1 (this code assumes dense
    1:1 frame alignment; an undetected gap would misalign every later label).
    `np.atleast_2d` guards a single-row CSV, which `np.loadtxt` would
    otherwise return as a 1D array and crash the `[:, 1]` column slice.
    """
    valence_path = ratings_dir / f"{video_id}_valence.csv"
    arousal_path = ratings_dir / f"{video_id}_arousal.csv"
    if not valence_path.exists() or not arousal_path.exists():
        raise FileNotFoundError(
            f"missing rating file(s) for video {video_id!r} in {ratings_dir}"
        )
    valence_arr = np.atleast_2d(np.loadtxt(valence_path, delimiter=","))
    arousal_arr = np.atleast_2d(np.loadtxt(arousal_path, delimiter=","))

    if len(valence_arr) != len(arousal_arr):
        raise ValueError(
            f"video {video_id!r}: valence has {len(valence_arr)} rows, "
            f"arousal has {len(arousal_arr)} rows -- expected equal counts"
        )
    for name, arr in (("valence", valence_arr), ("arousal", arousal_arr)):
        if not np.array_equal(arr[:, 0], np.arange(len(arr))):
            raise ValueError(
                f"video {video_id!r}: {name} frame-index column (col 0) is not "
                f"the contiguous sequence 0..{len(arr) - 1}"
            )

    return list(zip(valence_arr[:, 1].tolist(), arousal_arr[:, 1].tolist()))


def build(veatic_root: Path, out_root: Path, *, annotation_hz: float = 30.0) -> Path:
    veatic_root = Path(veatic_root)
    out_root = Path(out_root)
    frame_root = veatic_root / "frame"
    ratings_dir = veatic_root / "rating_averaged"
    if not frame_root.is_dir() or not ratings_dir.is_dir():
        raise ValueError(
            f"{veatic_root} must contain 'frame/' and 'rating_averaged/' "
            "(run scripts/fetch_veatic.py first)"
        )

    labels_dir = out_root / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    records: list[ManifestRecord] = []
    for clip_dir in sorted(
        (p for p in frame_root.iterdir() if p.is_dir()), key=lambda p: p.name
    ):
        video_id = clip_dir.name
        try:
            rows = load_video_ratings(ratings_dir, video_id)
        except FileNotFoundError:
            continue
        n_frames = sum(1 for p in clip_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
        if n_frames == 0:
            print(
                f"WARNING: skipping {video_id!r} -- 0 extracted frames (ratings present)"
            )
            continue
        if n_frames != len(rows):
            print(
                f"WARNING: {video_id!r} has {n_frames} frames but {len(rows)} rating "
                "rows -- will align to the shorter of the two at training time"
            )
        csv_rel = f"labels/{video_id}.csv"
        write_label_csv(out_root / csv_rel, rows)
        records.append(
            ManifestRecord(
                identity_id=video_id,
                clip_path=f"frame/{video_id}",
                duration_s=n_frames / annotation_hz,
                role="probe",
                label_path=csv_rel,
            )
        )

    if not records:
        raise ValueError(
            f"no VEATIC videos with matching rating files found under {veatic_root}"
        )
    manifest_path = out_root / "manifest.csv"
    write_manifest(manifest_path, records)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--veatic-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--annotation-hz", type=float, default=30.0)
    args = parser.parse_args()

    manifest_path = build(
        args.veatic_root,
        args.out_root or args.veatic_root,
        annotation_hz=args.annotation_hz,
    )
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
