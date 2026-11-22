#!/usr/bin/env python
"""Build manifest.csv from an existing WDS shard set's manifest.json.

This works without the raw mp4 corpus, which is useful on nodes that hold
only the shards. The mesh SSL path skips the manifest entirely, but the
single-node SSL path needs one, since build_ssl_loader reads
`dataset.manifest_path`.

clip_key entries in shards_root/manifest.json look like
"<identity_id>/<video_id>/<clip_id>", matching build_voxceleb2_manifest.py.
duration_s comes from num_frames at the shard build's fps (--fps here must match
build_wds_shards.py --fps).

Usage:
    PYTHONPATH=. uv run python stern/prep/manifest_from_shard_index.py \\
        --shards-root $DATA_ROOT/voxceleb2_shards \\
        --manifest-out $DATA_ROOT/voxceleb2/manifest.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stern.data.manifest import ManifestRecord, write_manifest
from stern.data.wds_frames import load_shard_index


def build_records(
    shard_index: dict[str, dict], *, fps: float = 2.0
) -> list[ManifestRecord]:
    """One SSL record per shard index entry."""
    records = []
    for clip_key, entry in shard_index.items():
        identity_id = clip_key.split("/", 1)[0]
        records.append(
            ManifestRecord(
                identity_id=identity_id,
                clip_path=f"{clip_key}.mp4",
                duration_s=entry["num_frames"] / fps,
                role="ssl",
            )
        )
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards-root", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="frame rate the shards were built at (build_wds_shards.py --fps)",
    )
    args = parser.parse_args(argv)

    shard_index = load_shard_index(args.shards_root)
    records = build_records(shard_index, fps=args.fps)
    write_manifest(args.manifest_out, records)
    total_h = sum(r.duration_s for r in records) / 3600.0
    print(f"wrote {len(records)} clips ({total_h:.1f}h) to {args.manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
