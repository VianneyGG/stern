#!/usr/bin/env python
"""Build a manifest covering the entire VoxCeleb2 corpus under --raw-root.

Unlike select_voxmini_subset.py, nothing is subsampled. Every probed clip is
written, which is what the full-corpus mesh run needs.

It reuses select_voxmini_subset.list_clips() for the walk and the threaded
duration probe. The manifest schema has no gender field, so no vox2_meta.csv
join is needed.

Usage (run from the repo root):
    PYTHONPATH=. uv run python stern/prep/build_voxceleb2_manifest.py \\
        --raw-root $DATA_ROOT/voxceleb2_raw/dev/dev/mp4 \\
        --manifest-out $DATA_ROOT/voxceleb2_raw/dev_manifest.csv
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
