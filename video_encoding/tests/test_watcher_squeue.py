from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from video_encoding.watcher.squeue import load_squeue_snapshot, refresh_squeue_snapshot

pytestmark = pytest.mark.integration

FIXTURE_PAYLOAD = {
    "jobs": [
        {
            "job_id": 36901,
            "job_state": ["RUNNING"],
            "nodes": "node001",
            "restart_cnt": 0,
        },
        {"job_id": 36902, "job_state": ["PENDING"], "nodes": "", "restart_cnt": 0},
    ]
}


def test_load_squeue_snapshot_parses_jobs(tmp_path):
    path = tmp_path / "squeue.json"
    path.write_text(json.dumps(FIXTURE_PAYLOAD))

    snapshot = load_squeue_snapshot(path)

    assert snapshot == {
        "36901": {"state": "RUNNING", "nodelist": "node001", "restart_cnt": 0},
        "36902": {"state": "PENDING", "nodelist": "", "restart_cnt": 0},
    }


def test_load_squeue_snapshot_missing_file_returns_empty(tmp_path):
    assert load_squeue_snapshot(tmp_path / "missing.json") == {}


def test_load_squeue_snapshot_captures_restart_cnt(tmp_path):
    path = tmp_path / "squeue.json"
    path.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "job_id": 123,
                        "job_state": ["PENDING"],
                        "nodes": "",
                        "restart_cnt": 2,
                    }
                ]
            }
        )
    )
    snapshot = load_squeue_snapshot(path)
    assert snapshot["123"]["restart_cnt"] == 2


def test_load_squeue_snapshot_defaults_restart_cnt_to_zero(tmp_path):
    path = tmp_path / "squeue.json"
    path.write_text(
        json.dumps({"jobs": [{"job_id": 1, "job_state": ["RUNNING"], "nodes": "a"}]})
    )
    snapshot = load_squeue_snapshot(path)
    assert snapshot["1"]["restart_cnt"] == 0


def test_refresh_squeue_snapshot_writes_atomically(tmp_path):
    out_path = tmp_path / "squeue.json"
    fake_result = MagicMock(stdout="36901|RUNNING|node001|0\n36902|PENDING||0\n")
    with patch("subprocess.run", return_value=fake_result) as mock_run:
        refresh_squeue_snapshot(out_path, user="lev")

    mock_run.assert_called_once_with(
        ["squeue", "-u", "lev", "-h", "-o", "%i|%T|%N|%r"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert not out_path.with_suffix(".json.tmp").exists()
    assert json.loads(out_path.read_text()) == FIXTURE_PAYLOAD


def test_refresh_squeue_snapshot_captures_nonzero_restart_cnt(tmp_path):
    out_path = tmp_path / "squeue.json"
    fake_result = MagicMock(stdout="36901|PENDING||2\n")
    with patch("subprocess.run", return_value=fake_result):
        refresh_squeue_snapshot(out_path, user="lev")

    payload = json.loads(out_path.read_text())
    assert payload["jobs"][0]["restart_cnt"] == 2


def test_refresh_squeue_snapshot_propagates_subprocess_failure(tmp_path):
    out_path = tmp_path / "squeue.json"
    with patch(
        "subprocess.run",
        side_effect=subprocess.CalledProcessError(1, ["squeue"]),
    ):
        with pytest.raises(subprocess.CalledProcessError, match="squeue"):
            refresh_squeue_snapshot(out_path, user="lev")
    assert out_path.exists() is False
