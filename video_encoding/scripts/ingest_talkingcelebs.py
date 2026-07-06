"""Build a manifest.csv for TalkingCelebs from an already-downloaded directory.

Run after download_talkingcelebs.py has finished:
    uv run python video_encoding/scripts/ingest_talkingcelebs.py \\
        --root $TALKINGCELEBS_ROOT/raw \\
        --manifest-out $TALKINGCELEBS_ROOT/manifest.csv

Then flip `present: true` in conf/dataset/talkingcelebs.yaml.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Callable

from data.manifest import ManifestRecord, write_manifest


def _probe_duration_av(clip_path: Path) -> float:
    import av

    with av.open(str(clip_path)) as container:
        return float(container.duration) / 1_000_000.0


def list_clips(
    root: Path,
    probe_duration_s: Callable[[Path], float] | None = None,
) -> dict[str, list[tuple[str, float]]]:
    """Walk root/<split>/<identity_id>/<video_id>/*.mp4.

    Returns identity_id -> [(clip_path relative to root, duration_s)].
    Aggregates clips from all splits so one identity maps to all its clips
    regardless of whether they came from the train or test subdataset.
    """
    if probe_duration_s is None:
        probe_duration_s = _probe_duration_av

    clips_by_identity: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for split_dir in sorted(root.iterdir()):
        if not split_dir.is_dir():
            continue
        for identity_dir in sorted(split_dir.iterdir()):
            if not identity_dir.is_dir():
                continue
            identity_id = identity_dir.name
            for clip_path in sorted(identity_dir.rglob("*.mp4")):
                duration_s = probe_duration_s(clip_path)
                clips_by_identity[identity_id].append(
                    (str(clip_path.relative_to(root)), duration_s)
                )
    return dict(clips_by_identity)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="downloaded TalkingCelebs directory (TALKINGCELEBS_ROOT/raw)")
    parser.add_argument("--manifest-out", type=Path, required=True)
    args = parser.parse_args()

    clips_by_identity = list_clips(args.root)
    records = [
        ManifestRecord(
            identity_id=identity_id,
            clip_path=clip_path,
            duration_s=duration_s,
            role="ssl",
        )
        for identity_id, clips in sorted(clips_by_identity.items())
        for clip_path, duration_s in clips
    ]
    write_manifest(args.manifest_out, records)
    total_h = sum(r.duration_s for r in records) / 3600.0
    print(f"wrote {len(records)} clips ({total_h:.1f}h) to {args.manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
