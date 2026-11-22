from __future__ import annotations

import json

from stern.watcher.registry import JobRecord, append_job_record, load_registry


def test_load_registry_missing_file_returns_empty(tmp_path):
    assert load_registry(tmp_path / "missing.json") == []


def test_append_and_load_round_trip(tmp_path):
    path = tmp_path / "jobs_registry.json"
    r1 = JobRecord(
        "36901",
        "videomae_ssl_pretrain_voxmini",
        "p2-eb256",
        "2026-07-15T09:00:00Z",
        "node001",
    )
    r2 = JobRecord(
        "36902", "videomae_va_finetune_veatic", None, "2026-07-15T09:05:00Z", "node047"
    )

    append_job_record(path, r1)
    append_job_record(path, r2)

    records = load_registry(path)
    assert records == [r1, r2]


def test_append_job_record_creates_parent_dir(tmp_path):
    path = tmp_path / "nested" / "state" / "jobs_registry.json"
    append_job_record(path, JobRecord("1", "exp", None, "2026-07-15T09:00:00Z", ""))
    payload = json.loads(path.read_text())
    assert payload[0]["slurm_job_id"] == "1"


def test_append_job_record_is_valid_json_list(tmp_path):
    path = tmp_path / "jobs_registry.json"
    append_job_record(
        path, JobRecord("1", "exp", "tag", "2026-07-15T09:00:00Z", "node001")
    )
    payload = json.loads(path.read_text())
    assert payload == [
        {
            "slurm_job_id": "1",
            "experiment": "exp",
            "run_tag": "tag",
            "submitted_at": "2026-07-15T09:00:00Z",
            "nodelist": "node001",
        }
    ]
