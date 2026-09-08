from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from video_encoding.watcher.registry import JobRecord
from video_encoding.watcher.remediate import remediate_dead_job

pytestmark = pytest.mark.integration


def test_remediate_dead_job_parses_new_job_id_and_passes_nodelist_flag():
    record = JobRecord(
        "100",
        "videomae_ssl_pretrain_voxmini",
        "p2-eb256",
        "2026-07-15T09:00:00Z",
        "node001,node002",
    )
    fake = MagicMock(returncode=0, stdout="...\nSubmitted batch job 101\n")

    with patch("subprocess.run", return_value=fake) as mock_run:
        new_id = remediate_dead_job(record, repo_root=Path("/repo"))

    assert new_id == "101"
    args, kwargs = mock_run.call_args
    cmd = args[0]
    assert cmd == [
        "sbatch",
        "--nodes=2",
        "--ntasks=2",
        "--nodelist=node001,node002",
        "video_encoding/slurm/train.sbatch",
        "videomae_ssl_pretrain_voxmini",
        "+run_tag=p2-eb256",
    ]
    assert kwargs["cwd"] == Path("/repo")


def test_remediate_dead_job_omits_run_tag_override_when_none():
    record = JobRecord("100", "exp", None, "2026-07-15T09:00:00Z", "node001")
    fake = MagicMock(returncode=0, stdout="Submitted batch job 200\n")
    with patch("subprocess.run", return_value=fake) as mock_run:
        remediate_dead_job(record, repo_root=Path("/repo"))
    cmd = mock_run.call_args[0][0]
    assert cmd == [
        "sbatch",
        "--nodes=1",
        "--ntasks=1",
        "--nodelist=node001",
        "video_encoding/slurm/train.sbatch",
        "exp",
    ]


def test_remediate_dead_job_omits_nodelist_flag_when_falsy():
    record = JobRecord("100", "exp", None, "2026-07-15T09:00:00Z", "")
    fake = MagicMock(returncode=0, stdout="Submitted batch job 300\n")
    with patch("subprocess.run", return_value=fake) as mock_run:
        remediate_dead_job(record, repo_root=Path("/repo"))
    cmd = mock_run.call_args[0][0]
    assert cmd == [
        "sbatch",
        "--nodes=1",
        "--ntasks=1",
        "video_encoding/slurm/train.sbatch",
        "exp",
    ]


def test_remediate_dead_job_returns_none_on_nonzero_exit():
    record = JobRecord("100", "exp", None, "2026-07-15T09:00:00Z", "node001")
    fake = MagicMock(returncode=1, stdout="ERROR: no idle nodes\n")
    with patch("subprocess.run", return_value=fake):
        assert remediate_dead_job(record, repo_root=Path("/repo")) is None


def test_remediate_dead_job_returns_none_when_job_id_unparseable():
    record = JobRecord("100", "exp", None, "2026-07-15T09:00:00Z", "node001")
    fake = MagicMock(returncode=0, stdout="submission accepted, no id printed\n")
    with patch("subprocess.run", return_value=fake):
        assert remediate_dead_job(record, repo_root=Path("/repo")) is None
