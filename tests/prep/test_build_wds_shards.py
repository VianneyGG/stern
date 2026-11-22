from __future__ import annotations

import io
import json
import pickle
import tarfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from stern.data.manifest import ManifestRecord, write_manifest
from stern.prep.build_wds_shards import (
    BalancedShardWriter,
    build_shards,
    compute_frame_step,
    encode_frames_jpeg,
)


@pytest.mark.parametrize(
    "native_fps, target_fps, expected_step",
    [
        (24.0, 2.0, 12),
        (2.0, 2.0, 1),
        (1.0, 2.0, 1),  # native slower than target -> never 0
        (0.0, 2.0, 1),  # unknown native fps -> don't drop everything
    ],
)
def test_compute_frame_step(native_fps, target_fps, expected_step):
    assert compute_frame_step(native_fps, target_fps) == expected_step


def test_encode_frames_jpeg_roundtrip():
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    frame[0, 0] = [255, 0, 0]
    [jpeg_bytes] = encode_frames_jpeg([frame], quality=95)

    assert jpeg_bytes[:2] == b"\xff\xd8"  # JPEG magic
    decoded = np.array(Image.open(io.BytesIO(jpeg_bytes)).convert("RGB"))
    assert decoded.shape == (8, 8, 3)


def test_balanced_shard_writer_spreads_load_and_keeps_clip_frames_together(tmp_path):
    writer = BalancedShardWriter(tmp_path, n_shards=4)
    # 8 clips, alternating small/large payloads.
    for i in range(8):
        payload = [b"x" * (1000 if i % 2 == 0 else 100)]
        writer.write(f"clip{i}", payload)
    writer.close()

    sizes = writer.sizes
    assert max(sizes) - min(sizes) <= 1000  # greedy min-loaded-bin bound

    # Every clip's single frame-list member landed in exactly one shard, intact.
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert set(manifest) == {f"clip{i}" for i in range(8)}
    for clip_key, entry in manifest.items():
        with tarfile.open(tmp_path / entry["shard"]) as tar:
            member = tar.extractfile(f"{clip_key}.pickle")
            frames = pickle.loads(member.read())
        assert len(frames) == 1


def test_balanced_shard_writer_resumes_without_duplicating(tmp_path):
    writer = BalancedShardWriter(tmp_path, n_shards=2)
    writer.write("clip0", [b"a"])
    writer.close()

    writer2 = BalancedShardWriter(tmp_path, n_shards=2)
    assert "clip0" in writer2.already_written
    writer2.write("clip1", [b"b"])
    writer2.close()

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert set(manifest) == {"clip0", "clip1"}


def _write_test_manifest(path: Path, clip_paths: list[str]) -> None:
    write_manifest(
        path,
        [
            ManifestRecord(
                identity_id="id00012", clip_path=cp, duration_s=5.0, role="ssl"
            )
            for cp in clip_paths
        ],
    )


def test_build_shards_skips_clips_already_in_manifest(tmp_path: Path):
    manifest_path = tmp_path / "manifest.csv"
    _write_test_manifest(
        manifest_path, ["id00012/vid1/001.mp4", "id00012/vid1/002.mp4"]
    )
    out_dir = tmp_path / "shards"
    out_dir.mkdir()
    # Pre-seed as already-converted.
    seed_writer = BalancedShardWriter(out_dir, n_shards=2)
    seed_writer.write("id00012/vid1/001", [b"x"])
    seed_writer.close()

    seen_jobs: list[list[tuple[Path, str]]] = []

    def fake_run_encode(jobs, target_fps, size):
        seen_jobs.append(jobs)
        return [([b"y"], None) for _ in jobs]

    n_ok, n_failed = build_shards(
        manifest_path,
        tmp_path / "raw",
        out_dir,
        n_shards=2,
        run_encode=fake_run_encode,
    )

    assert n_ok == 1
    assert n_failed == 0
    [jobs] = seen_jobs
    assert len(jobs) == 1
    clip_path, clip_key = jobs[0]
    assert clip_path == tmp_path / "raw" / "id00012/vid1/002.mp4"
    assert clip_key == "id00012/vid1/002"


def test_build_shards_chunks_todo_and_persists_manifest_incrementally(tmp_path: Path):
    """A batch larger than chunk_size must be split across multiple
    run_encode calls (bounding memory + kill-loss to one chunk), and the
    manifest.json on disk must reflect each completed chunk's writes as it
    happens, not only once the whole backlog finishes."""
    manifest_path = tmp_path / "manifest.csv"
    clip_paths = [f"id00012/vid1/{i:03d}.mp4" for i in range(5)]
    _write_test_manifest(manifest_path, clip_paths)

    seen_jobs: list[list[tuple[Path, str]]] = []
    manifest_snapshots_before_each_chunk: list[set[str]] = []
    out_dir = tmp_path / "shards"
    manifest_json_path = out_dir / "manifest.json"

    def fake_run_encode(jobs, target_fps, size):
        seen_jobs.append(jobs)
        # run_encode is invoked *before* this chunk's writer.write() calls,
        # so what's durably on disk right now reflects only prior chunks.
        # If build_shards still encoded the whole backlog before writing
        # anything (the pre-fix behavior), every call here would see an
        # empty/missing manifest -- not a cumulative one.
        written_so_far = (
            set(json.loads(manifest_json_path.read_text()))
            if manifest_json_path.exists()
            else set()
        )
        manifest_snapshots_before_each_chunk.append(written_so_far)
        return [([b"y"], None) for _ in jobs]

    n_ok, n_failed = build_shards(
        manifest_path,
        tmp_path / "raw",
        out_dir,
        n_shards=2,
        run_encode=fake_run_encode,
        chunk_size=2,
    )

    assert n_ok == 5
    assert n_failed == 0
    # 5 clips / chunk_size=2 -> 3 calls (2, 2, 1), never one call for everything.
    assert len(seen_jobs) == 3
    assert [len(jobs) for jobs in seen_jobs] == [2, 2, 1]

    # By the time chunk 2 starts, chunk 1's 2 writes are already durably in
    # manifest.json; by chunk 3, both prior chunks' 4 writes are. Proves
    # writer.write() (and the atomic manifest persist) happens per chunk,
    # not once at the very end after all encoding completes.
    assert [len(s) for s in manifest_snapshots_before_each_chunk] == [0, 2, 4]

    final_manifest = json.loads(manifest_json_path.read_text())
    assert set(final_manifest) == {str(Path(cp).with_suffix("")) for cp in clip_paths}


def test_build_shards_counts_failures_without_crashing(tmp_path: Path):
    manifest_path = tmp_path / "manifest.csv"
    _write_test_manifest(
        manifest_path,
        ["id00012/vid1/001.mp4", "id00012/vid1/002.mp4", "id00012/vid1/003.mp4"],
    )

    def fake_run_encode(jobs, target_fps, size):
        return [
            ([], f"{clip_path}: boom") if i == 1 else ([b"y"], None)
            for i, (clip_path, _) in enumerate(jobs)
        ]

    n_ok, n_failed = build_shards(
        manifest_path,
        tmp_path / "raw",
        tmp_path / "shards",
        n_shards=2,
        run_encode=fake_run_encode,
    )

    assert n_ok == 2
    assert n_failed == 1
