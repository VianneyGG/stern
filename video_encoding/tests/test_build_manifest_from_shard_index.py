from __future__ import annotations

import json
from pathlib import Path

from data.manifest import read_manifest
from video_encoding.scripts.build_manifest_from_shard_index import (
    build_records,
    main,
)


def test_build_records_derives_identity_and_duration_from_shard_index():
    shard_index = {
        "id00012/vid1/001": {"shard": "shard-000000.tar", "num_frames": 32, "size": 4},
        "id00012/vid2/002": {"shard": "shard-000000.tar", "num_frames": 8, "size": 4},
        "id00099/vid1/001": {"shard": "shard-000001.tar", "num_frames": 40, "size": 4},
    }

    records = build_records(shard_index, fps=2.0)

    by_key = {r.clip_path: r for r in records}
    assert by_key["id00012/vid1/001.mp4"].identity_id == "id00012"
    assert by_key["id00012/vid1/001.mp4"].duration_s == 16.0
    assert by_key["id00099/vid1/001.mp4"].identity_id == "id00099"
    assert by_key["id00099/vid1/001.mp4"].duration_s == 20.0
    assert all(r.role == "ssl" for r in records)


def test_records_round_trip_and_clip_path_strips_back_to_shard_key():
    """SSLWindowDataset looks up shard_index[str(Path(clip_path).with_suffix(''))]
    -- the manifest's clip_path must strip back to exactly the original
    shard-index key, or every clip silently fails to load."""
    shard_index = {
        "id00012/vid1/001": {"shard": "shard-000000.tar", "num_frames": 16, "size": 4},
    }
    records = build_records(shard_index, fps=2.0)

    [record] = records
    assert str(Path(record.clip_path).with_suffix("")) == "id00012/vid1/001"


def test_main_writes_manifest_readable_by_read_manifest(tmp_path: Path):
    shards_root = tmp_path / "shards"
    shards_root.mkdir()
    (shards_root / "manifest.json").write_text(
        json.dumps(
            {
                "id00012/vid1/001": {
                    "shard": "shard-000000.tar",
                    "num_frames": 16,
                    "size": 4,
                },
                "id00012/vid1/002": {
                    "shard": "shard-000000.tar",
                    "num_frames": 24,
                    "size": 4,
                },
            }
        )
    )
    manifest_out = tmp_path / "voxceleb2" / "manifest.csv"

    rc = main(
        [
            "--shards-root",
            str(shards_root),
            "--manifest-out",
            str(manifest_out),
        ]
    )

    assert rc == 0
    records = read_manifest(manifest_out, expected_role="ssl")
    assert len(records) == 2
    assert {r.identity_id for r in records} == {"id00012"}
