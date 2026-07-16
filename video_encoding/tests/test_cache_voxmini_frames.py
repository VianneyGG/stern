# video_encoding/tests/test_cache_voxmini_frames.py
from pathlib import Path

import pytest

from data.manifest import ManifestRecord, write_manifest
from video_encoding.scripts.cache_voxmini_frames import (
    cache_frames,
    compute_frame_step,
    is_already_cached,
)


@pytest.mark.parametrize(
    "native_fps, target_fps, expected_step",
    [
        (25.0, 8.0, 3),  # round(25/8) = 3
        (8.0, 8.0, 1),  # same rate -> keep every frame
        (4.0, 8.0, 1),  # native slower than target -> never 0, keep every frame
        (0.0, 8.0, 1),  # unknown native fps -> don't drop everything
    ],
)
def test_compute_frame_step(native_fps, target_fps, expected_step):
    assert compute_frame_step(native_fps, target_fps) == expected_step


def test_is_already_cached_false_for_missing_or_empty_dir(tmp_path: Path):
    missing = tmp_path / "missing"
    assert is_already_cached(missing) is False

    empty = tmp_path / "empty"
    empty.mkdir()
    assert is_already_cached(empty) is False


def test_is_already_cached_true_when_frames_present(tmp_path: Path):
    out_dir = tmp_path / "clip"
    out_dir.mkdir()
    (out_dir / "0000.jpg").touch()
    assert is_already_cached(out_dir) is True


def _write_test_manifest(path: Path, clip_paths: list[str]) -> None:
    records = [
        ManifestRecord(
            identity_id="id00012", clip_path=clip_path, duration_s=5.0, role="ssl"
        )
        for clip_path in clip_paths
    ]
    write_manifest(path, records)


def test_cache_frames_skips_clips_already_cached(tmp_path: Path):
    manifest_path = tmp_path / "manifest.csv"
    _write_test_manifest(
        manifest_path, ["id00012/vid1/001.mp4", "id00012/vid1/002.mp4"]
    )

    raw_root = tmp_path / "raw"
    frames_root = tmp_path / "frames"
    already_done = frames_root / "id00012/vid1/001"
    already_done.mkdir(parents=True)
    (already_done / "0000.jpg").touch()

    seen_jobs: list[list[tuple[Path, Path]]] = []

    def fake_run_transcode(jobs):
        seen_jobs.append(jobs)
        return [None for _ in jobs]

    n_ok, n_failed = cache_frames(
        manifest_path, raw_root, frames_root, run_transcode=fake_run_transcode
    )

    assert n_ok == 1
    assert n_failed == 0
    [jobs] = seen_jobs
    assert len(jobs) == 1
    clip_path, out_dir = jobs[0]
    assert clip_path == raw_root / "id00012/vid1/002.mp4"
    assert out_dir == frames_root / "id00012/vid1/002"


def test_cache_frames_counts_failures_without_crashing(tmp_path: Path):
    manifest_path = tmp_path / "manifest.csv"
    _write_test_manifest(
        manifest_path,
        ["id00012/vid1/001.mp4", "id00012/vid1/002.mp4", "id00012/vid1/003.mp4"],
    )

    def fake_run_transcode(jobs):
        return [
            f"{clip_path}: boom" if i == 1 else None
            for i, (clip_path, _) in enumerate(jobs)
        ]

    n_ok, n_failed = cache_frames(
        manifest_path,
        tmp_path / "raw",
        tmp_path / "frames",
        run_transcode=fake_run_transcode,
    )

    assert n_ok == 2
    assert n_failed == 1
