from __future__ import annotations

from video_encoding.scripts.append_job_registry import main
from video_encoding.watcher.registry import load_registry


def test_main_appends_record_from_argv(tmp_path):
    out_path = tmp_path / "jobs_registry.json"
    argv = [
        "--registry-path",
        str(out_path),
        "--job-id",
        "36901",
        "--experiment",
        "videomae_ssl_pretrain_voxmini",
        "--run-tag",
        "p2-eb256",
        "--nodelist",
        "node001",
    ]
    rc = main(argv)
    assert rc == 0
    records = load_registry(out_path)
    assert len(records) == 1
    assert records[0].slurm_job_id == "36901"
    assert records[0].run_tag == "p2-eb256"


def test_main_treats_empty_run_tag_as_none(tmp_path):
    out_path = tmp_path / "jobs_registry.json"
    argv = [
        "--registry-path",
        str(out_path),
        "--job-id",
        "1",
        "--experiment",
        "exp",
        "--run-tag",
        "",
        "--nodelist",
        "node001",
    ]
    main(argv)
    assert load_registry(out_path)[0].run_tag is None
