#!/usr/bin/env python
"""Build a STERN manifest covering the ENTIRE VoxCeleb2 corpus under --raw-root,
with no subset/downsampling.

Unlike select_voxmini_subset.py (which always culls via
select_stratified_subset and raises ValueError if --target-hours exceeds the
real corpus size), this writes every probed clip -- for the decentralized
mesh plan's "full-corpus manifest" prerequisite
(docs/superpowers/plans/2026-07-16-decentralized-mesh-training.md).

Reuses select_voxmini_subset.py's list_clips() (walk + threaded duration
probe, ~28ms/clip via av.open(), parallelized across --probe-workers)
verbatim -- no second decode/probe path. No vox2_meta.csv join needed: the
manifest schema doesn't carry gender, only identity_id/clip_path/duration_s.

Usage:
    PYTHONPATH=. uv run python stern/prep/build_voxceleb2_manifest.py \\
        --raw-root /Data/$USER/voxceleb2_raw/dev/dev/mp4 \\
        --manifest-out /Data/$USER/voxceleb2_raw/dev_manifest.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stern.data.manifest import ManifestRecord, write_manifest
from stern.prep.select_voxmini_subset import list_clips


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument(
        "--probe-workers",
        type=int,
        default=32,
        help="thread-pool size for per-clip duration probing (I/O-bound)",
    )
    args = parser.parse_args()

    clips_by_identity = list_clips(args.raw_root, max_workers=args.probe_workers)

    records = [
        ManifestRecord(
            identity_id=identity_id,
            clip_path=clip_path,
            duration_s=duration_s,
            role="ssl",
        )
        for identity_id, clips in clips_by_identity.items()
        for clip_path, duration_s in clips
    ]
    write_manifest(args.manifest_out, records)
    total_h = sum(r.duration_s for r in records) / 3600.0
    print(f"wrote {len(records)} clips ({total_h:.1f}h) to {args.manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
