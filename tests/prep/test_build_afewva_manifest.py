"""AFEW-VA ingest: synthetic-fixture test for the manifest/label builder.

Builds a tiny tree in the documented AFEW-VA layout (clip dir of PNGs + a JSON
mapping each frame to valence/arousal in [-10,10]) and asserts the builder emits
a role=probe manifest + per-clip label CSVs that satisfy the STERN contract and
harmonise cleanly to [-1,1].
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from stern.data.labels import load_per_frame_va
from stern.data.manifest import read_manifest
from stern.prep.build_afewva_manifest import build, parse_clip_annotations


def _write_clip(clip_dir: Path, va_rows: list[tuple[int, int]]) -> None:
    clip_dir.mkdir(parents=True, exist_ok=True)
    for i, _ in enumerate(va_rows):
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(
            clip_dir / f"{i:05d}.png"
        )
    frames = {
        f"{i:05d}": {"valence": v, "arousal": a, "landmarks": [[0, 0]] * 68}
        for i, (v, a) in enumerate(va_rows)
    }
    (clip_dir / f"{clip_dir.name}.json").write_text(
        json.dumps({"video_id": clip_dir.name, "frames": frames})
    )


def test_parse_clip_annotations_orders_by_frame_key(tmp_path: Path):
    _write_clip(tmp_path / "007", [(1, -2), (3, 4), (-5, 6)])

    actor, rows = parse_clip_annotations(tmp_path / "007" / "007.json")

    assert actor is None
    assert rows == [(0.1, -0.2), (0.3, 0.4), (-0.5, 0.6)]


def test_parse_clip_annotations_sorts_unpadded_keys_numerically(tmp_path: Path):
    # 11 frames with unpadded keys ("0".."10") forces "10" to sort before "2"
    # under a naive lexicographic sorted(frames); this is the real AFEW-VA
    # risk this module's own docstring flags as unconfirmed.
    clip_dir = tmp_path / "010"
    clip_dir.mkdir(parents=True)
    frames = {str(i): {"valence": i, "arousal": -i} for i in range(11)}
    (clip_dir / "010.json").write_text(json.dumps({"frames": frames}))

    _, rows = parse_clip_annotations(clip_dir / "010.json")

    assert rows == [(i / 10.0, -i / 10.0) for i in range(11)]


def test_build_emits_probe_manifest_and_harmonisable_labels(tmp_path: Path):
    _write_clip(tmp_path / "001", [(10, -10), (0, 5), (-10, 10)])
    _write_clip(tmp_path / "002", [(2, 2), (-2, -2)])

    manifest_path = build(tmp_path, tmp_path)

    records = read_manifest(manifest_path, expected_role="probe")
    assert {r.identity_id for r in records} == {"001", "002"}

    rec001 = next(r for r in records if r.identity_id == "001")
    assert rec001.clip_path == "001"
    assert rec001.label_path == "labels/001.csv"

    # Labels are pre-scaled [-10,10] -> [-1,1]: endpoints map to +-1, midpoint to 0.
    va = load_per_frame_va(tmp_path / rec001.label_path, (-1.0, 1.0))
    np.testing.assert_allclose(va[0], [1.0, -1.0])
    np.testing.assert_allclose(va[1], [0.0, 0.5])
    np.testing.assert_allclose(va[2], [-1.0, 1.0])


def test_build_raises_when_no_clips(tmp_path: Path):
    (tmp_path / "not_a_clip").mkdir()
    try:
        build(tmp_path, tmp_path)
    except ValueError as exc:
        assert "no AFEW-VA clips" in str(exc)
    else:
        raise AssertionError("expected ValueError when no clips are present")
