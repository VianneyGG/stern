from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

from video_encoding.watcher.squeue import load_squeue_snapshot, refresh_squeue_snapshot

FIXTURE_PAYLOAD = {
    "jobs": [
        {"job_id": 36901, "job_state": ["RUNNING"], "nodes": "node001"},
        {"job_id": 36902, "job_state": ["PENDING"], "nodes": ""},
    ]
}


def test_load_squeue_snapshot_parses_jobs(tmp_path):
    path = tmp_path / "squeue.json"
    path.write_text(json.dumps(FIXTURE_PAYLOAD))

    snapshot = load_squeue_snapshot(path)

    assert snapshot == {
        "36901": {"state": "RUNNING", "nodelist": "node001"},
        "36902": {"state": "PENDING", "nodelist": ""},
    }


def test_load_squeue_snapshot_missing_file_returns_empty(tmp_path):
    assert load_squeue_snapshot(tmp_path / "missing.json") == {}


def test_refresh_squeue_snapshot_writes_atomically(tmp_path):
    out_path = tmp_path / "squeue.json"
    fake_result = MagicMock(stdout=json.dumps(FIXTURE_PAYLOAD))
    with patch("subprocess.run", return_value=fake_result) as mock_run:
        refresh_squeue_snapshot(out_path, user="lev")

    mock_run.assert_called_once_with(
        ["squeue", "-u", "lev", "--json"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert not out_path.with_suffix(".json.tmp").exists()
    assert json.loads(out_path.read_text()) == FIXTURE_PAYLOAD


def test_refresh_squeue_snapshot_propagates_subprocess_failure(tmp_path):
    out_path = tmp_path / "squeue.json"
    with patch(
        "subprocess.run",
        side_effect=subprocess.CalledProcessError(1, ["squeue"]),
    ):
        try:
            refresh_squeue_snapshot(out_path, user="lev")
            assert False, "expected CalledProcessError"
        except subprocess.CalledProcessError:
            pass
    assert not out_path.exists()
