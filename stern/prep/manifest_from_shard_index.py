#!/usr/bin/env python
"""Build a STERN manifest.csv from an already-built WDS shard set's
manifest.json, without touching the raw mp4 corpus.

build_voxceleb2_full_manifest.py needs --raw-root (raw mp4 files); on the
mesh fish nodes only the pre-built shards exist, not the raw corpus. The
mesh SSL path (mesh_round.py) bypasses cfg.dataset/manifest.csv entirely, so
this gap was never exercised until the non-mesh (traditional single-node)
SSL path needed it -- build_ssl_loader unconditionally reads
dataset.manifest_path.

clip_key entries in shards_root/manifest.json are
"<identity_id>/<video_id>/<clip_id>" (build_wds_shards.py, matching
list_clips' raw-mp4 walk convention). duration_s is derived from num_frames
at the shard build's fixed 2fps (build_wds_shards.py --fps default), not a
placeholder.

Usage:
    PYTHONPATH=. uv run python stern/prep/manifest_from_shard_index.py \\
        --shards-root /Data/$USER/voxceleb2_shards \\
        --manifest-out /Data/$USER/voxceleb2/manifest.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stern.data.manifest import ManifestRecord, write_manifest
from stern.data.wds_frames import load_shard_index


def build_records(
    shard_index: dict[str, dict], *, fps: float = 2.0
) -> list[ManifestRecord]:
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
