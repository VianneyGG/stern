"""Stratified subset selection: full VoxCeleb2 dev set -> ~218h VoxMini manifest.

Stratifies by Gender only -- VoxCeleb2's own vox2_meta.csv has no Nationality
field (that's VoxCeleb1-only, a different identity namespace), so gender is the
only demographic axis available without an unreliable name-based proxy.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from stern.data.manifest import ManifestRecord, write_manifest


def _safe_probe(
    clip_path: Path, probe_duration_s: Callable[[Path], float]
) -> float | None:
    """Probe one clip, returning None if it is unreadable.

    Corrupt files do occur in VoxCeleb2, for example truncated mp4s. One bad
    file must not abort a full-corpus run. Skipped clips are dropped from the
    manifest rather than recorded with zero duration.
    """
    try:
        return probe_duration_s(clip_path)
    except Exception as exc:
        print(f"skipping unreadable clip {clip_path}: {exc}", file=sys.stderr)
        return None


@dataclass
class Identity:
    identity_id: str
    gender: str
    clips: list[tuple[str, float]]  # (clip_path relative to raw root, duration_s)

    @property
    def total_duration_s(self) -> float:
        return sum(duration for _, duration in self.clips)


def _probe_duration_av(clip_path: Path) -> float:
    """Duration in seconds. Some mp4s lack a container duration, so fall back to the
    video stream's duration, then to 0.0."""
    import av

    with av.open(str(clip_path)) as container:
        if container.duration is not None:
            return (
                float(container.duration) / 1_000_000.0
            )  # av duration is in microseconds
        video_stream = next((s for s in container.streams if s.type == "video"), None)
        if video_stream is not None and video_stream.duration is not None:
            return float(video_stream.duration * video_stream.time_base)
        return 0.0


def list_clips(
    raw_mp4_root: Path,
    probe_duration_s: Callable[[Path], float] | None = None,
    max_workers: int = 32,
) -> dict[str, list[tuple[str, float]]]:
    """Walk `raw_mp4_root/<identity_id>/<video_id>/<clip_id>.mp4` and return
    identity_id -> [(clip_path relative to raw_mp4_root, duration_s), ...].

    `probe_duration_s` defaults to probing the real file via `av`; injectable
    for testing without real video files.

    Probing is I/O-bound and independent per clip, so it runs on a thread pool.
    Selection happens only after every clip is probed.
    """
    if probe_duration_s is None:
        probe_duration_s = _probe_duration_av

    identity_dirs = sorted(d for d in raw_mp4_root.iterdir() if d.is_dir())
    ordered_clips: list[tuple[str, Path]] = [
        (identity_dir.name, clip_path)
        for identity_dir in identity_dirs
        for clip_path in sorted(identity_dir.rglob("*.mp4"))
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        durations = list(
            pool.map(
                lambda p: _safe_probe(p, probe_duration_s),
                (p for _, p in ordered_clips),
            )
        )

    clips_by_identity: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for (identity_id, clip_path), duration_s in zip(ordered_clips, durations):
        if duration_s is None:
            continue
        clips_by_identity[identity_id].append(
            (str(clip_path.relative_to(raw_mp4_root)), duration_s)
        )
    return dict(clips_by_identity)


def load_identities(
    meta_csv: Path,
    raw_mp4_root: Path,
    probe_duration_s: Callable[[Path], float] | None = None,
    max_workers: int = 32,
) -> list[Identity]:
    """Join `vox2_meta.csv` (identity -> gender) with the actual clip files
    found under `raw_mp4_root`. Identities present on disk but absent from
    `meta_csv` are skipped (no gender to stratify by).

    The official file is tab-delimited, with padding around headers and values,
    despite its `.csv` name. Every field is stripped after parsing.
    """
    genders: dict[str, str] = {}
    with meta_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            clean = {k.strip(): v.strip() for k, v in row.items()}
            genders[clean["VoxCeleb2 ID"]] = clean["Gender"]

    clips_by_identity = list_clips(
        raw_mp4_root, probe_duration_s=probe_duration_s, max_workers=max_workers
    )

    return [
        Identity(identity_id=identity_id, gender=genders[identity_id], clips=clips)
        for identity_id, clips in clips_by_identity.items()
        if identity_id in genders
    ]


def select_stratified_subset(
    identities: list[Identity], target_hours: float, seed: int
) -> list[Identity]:
    """Select whole identities, stratified by gender, until total duration
    crosses target_hours, preserving the corpus's overall gender ratio.

    Whole identities (never partial) are selected so a later train/holdout
    split never splits one person's clips across both sides. Raises
    ValueError if the corpus can't reach target_hours.
    """
    target_s = target_hours * 3600.0
    by_gender: dict[str, list[Identity]] = defaultdict(list)
    for identity in identities:
        by_gender[identity.gender].append(identity)

    total_corpus_s = sum(i.total_duration_s for i in identities)
    if total_corpus_s < target_s:
        raise ValueError(
            f"corpus has only {total_corpus_s / 3600.0:.1f}h, "
            f"cannot reach target_hours={target_hours}"
        )

    rng = random.Random(seed)
    selected: list[Identity] = []
    for gender, gender_identities in by_gender.items():
        gender_corpus_s = sum(i.total_duration_s for i in gender_identities)
        gender_target_s = target_s * (gender_corpus_s / total_corpus_s)
        shuffled = rng.sample(gender_identities, len(gender_identities))
        gender_selected_s = 0.0
        for identity in shuffled:
            if gender_selected_s >= gender_target_s:
                break
            selected.append(identity)
            gender_selected_s += identity.total_duration_s

    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--meta-csv", type=Path, required=True, help="path to vox2_meta.csv"
    )
    parser.add_argument(
        "--raw-root", type=Path, required=True, help="unzipped VoxCeleb2 dev mp4 root"
    )
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--target-hours", type=float, default=218.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--probe-workers",
        type=int,
        default=32,
        help="thread-pool size for per-clip duration probing (I/O-bound)",
    )
    args = parser.parse_args()

    identities = load_identities(
        args.meta_csv, args.raw_root, max_workers=args.probe_workers
    )
    selected = select_stratified_subset(identities, args.target_hours, args.seed)

    records = [
        ManifestRecord(
            identity_id=identity.identity_id,
            clip_path=clip_path,
            duration_s=duration_s,
            role="ssl",
        )
        for identity in selected
        for clip_path, duration_s in identity.clips
    ]
    write_manifest(args.manifest_out, records)
    total_h = sum(r.duration_s for r in records) / 3600.0
    print(f"wrote {len(records)} clips ({total_h:.1f}h) to {args.manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
