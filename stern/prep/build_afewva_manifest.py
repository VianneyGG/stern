#!/usr/bin/env python
"""Build the manifest and per-clip VA label CSVs from an extracted AFEW-VA tree.

AFEW-VA (ibug, EULA-gated) has one folder per clip. Each folder holds frame
images and one JSON file with per-frame integer valence and arousal in
[-10, 10]:

    {"actor": "Actor Name", "video_id": "001",
     "frames": {"00000": {"valence": 2, "arousal": -3, "landmarks": [[x, y] * 68]},
                ...}}

Under `--out-root` this writes:
  labels/<clip>.csv   `valence,arousal` per frame, divided by 10 into [-1, 1]
  manifest.csv        one ManifestRecord per clip, role=probe

Labels are already in [-1, 1], so the probe's `va_range` must be [-1, 1]
(configs/dataset/afewva.yaml). Otherwise harmonise_va would rescale them again.

`identity_id` is the clip's `actor` when present, which allows an actor-disjoint
split. Otherwise it falls back to the clip id.

A clip whose frame and annotation counts differ is skipped and reported, never
aligned to the shorter one. A misaligned label corrupts CCC for every window
in the clip.

Run from the repo root. `stern.data.manifest` must be on PYTHONPATH:
    PYTHONPATH=. uv run python stern/prep/build_afewva_manifest.py \\
        --afewva-root $AFEWVA_ROOT
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stern.data.labels import write_label_csv
from stern.data.manifest import ManifestRecord, write_manifest
from stern.data.frame_decoder import _IMAGE_EXTS

_VA_SCALE = 10.0  # AFEW-VA's native [-10, 10] -> [-1, 1] for PROBE_VA_RANGE


def parse_clip_annotations(
    json_path: Path,
) -> tuple[str | None, list[tuple[float, float]]]:
    """Return `(actor, [(valence, arousal), ...])` in frame order from one clip's JSON.

    Values are scaled from [-10, 10] to [-1, 1]. Frame keys sort numerically when
    they are all integers, as `list_frame_files` does. A plain string sort would
    put "10" before "2" and misalign every label after the first ten frames.
    """
    data = json.loads(Path(json_path).read_text())
    frames = data.get("frames")
    if not isinstance(frames, dict):
        raise ValueError(
            f"{json_path}: expected a 'frames' object, got {type(frames).__name__}"
        )
    keys = (
        sorted(frames, key=int) if all(k.isdigit() for k in frames) else sorted(frames)
    )
    rows = [
        (
            float(frames[key]["valence"]) / _VA_SCALE,
            float(frames[key]["arousal"]) / _VA_SCALE,
        )
        for key in keys
    ]
    if not rows:
        raise ValueError(f"{json_path}: 'frames' is empty")
    actor = data.get("actor") or None
    return actor, rows


def build(afewva_root: Path, out_root: Path, *, annotation_hz: float = 30.0) -> Path:
    """Write label CSVs and `manifest.csv` under `out_root`; return the manifest path."""
    afewva_root = Path(afewva_root)
    out_root = Path(out_root)
    labels_dir = out_root / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    records: list[ManifestRecord] = []
    for clip_dir in sorted(p for p in afewva_root.iterdir() if p.is_dir()):
        if clip_dir.resolve() == labels_dir.resolve():
            continue
        jsons = sorted(clip_dir.glob("*.json"))
        if not jsons:
            continue
        clip = clip_dir.name
        actor, rows = parse_clip_annotations(jsons[0])
        n_frames = sum(1 for p in clip_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
        if n_frames == 0:
            print(f"SKIPPED {clip!r} -- 0 extracted frames (annotations present)")
            continue
        if n_frames != len(rows):
            print(
                f"SKIPPED {clip!r} -- {n_frames} frames but {len(rows)} annotation rows"
            )
            continue
        csv_rel = f"labels/{clip}.csv"
        write_label_csv(out_root / csv_rel, rows)
        records.append(
            ManifestRecord(
                identity_id=actor or clip,
                clip_path=clip,
                duration_s=n_frames / annotation_hz,
                role="probe",
                label_path=csv_rel,
            )
        )

    if not records:
        raise ValueError(f"no AFEW-VA clips (dir + *.json) found under {afewva_root}")
    manifest_path = out_root / "manifest.csv"
    write_manifest(manifest_path, records)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--afewva-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--annotation-hz", type=float, default=30.0)
    args = parser.parse_args()

    manifest_path = build(
        args.afewva_root,
        args.out_root or args.afewva_root,
        annotation_hz=args.annotation_hz,
    )
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
