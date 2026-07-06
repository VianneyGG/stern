"""Stratified subset selection: full VoxCeleb2 dev set -> ~218h VoxMini manifest.

Stratifies by Gender only -- VoxCeleb2's own vox2_meta.csv has no Nationality
field (that's VoxCeleb1-only, a different identity namespace), so gender is the
only demographic axis available without an unreliable name-based proxy.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from data.manifest import ManifestRecord, write_manifest


@dataclass
class Identity:
    identity_id: str
    gender: str
    clips: list[tuple[str, float]]  # (clip_path relative to raw root, duration_s)

    @property
    def total_duration_s(self) -> float:
        return sum(duration for _, duration in self.clips)


def _probe_duration_av(clip_path: Path) -> float:
    import av

    with av.open(str(clip_path)) as container:
        return float(container.duration) / 1_000_000.0  # av duration is in microseconds


def list_clips(
    raw_mp4_root: Path, probe_duration_s: Callable[[Path], float] | None = None
) -> dict[str, list[tuple[str, float]]]:
    """Walk `raw_mp4_root/<identity_id>/<video_id>/<clip_id>.mp4` and return
    identity_id -> [(clip_path relative to raw_mp4_root, duration_s), ...].

    `probe_duration_s` defaults to probing the real file via `av`; injectable
    for testing without real video files.
    """
    if probe_duration_s is None:
        probe_duration_s = _probe_duration_av

    clips_by_identity: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for identity_dir in sorted(raw_mp4_root.iterdir()):
        if not identity_dir.is_dir():
            continue
        identity_id = identity_dir.name
        for clip_path in sorted(identity_dir.rglob("*.mp4")):
            duration_s = probe_duration_s(clip_path)
            clips_by_identity[identity_id].append(
                (str(clip_path.relative_to(raw_mp4_root)), duration_s)
            )
    return dict(clips_by_identity)


def load_identities(
    meta_csv: Path,
    raw_mp4_root: Path,
    probe_duration_s: Callable[[Path], float] | None = None,
) -> list[Identity]:
    """Join `vox2_meta.csv` (identity -> gender) with the actual clip files
    found under `raw_mp4_root`. Identities present on disk but absent from
    `meta_csv` are skipped (no gender to stratify by)."""
    genders: dict[str, str] = {}
    with meta_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            genders[row["VoxCeleb2 ID"]] = row["Gender"]

    clips_by_identity = list_clips(raw_mp4_root, probe_duration_s=probe_duration_s)

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
    args = parser.parse_args()

    identities = load_identities(args.meta_csv, args.raw_root)
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
