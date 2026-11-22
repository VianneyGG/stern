"""Launcher invariants for the cluster constraints.

These guard the --no-sync and one-GPU-no-DDP cluster constraints: the
project venv lives on NFS, so every launcher must run uv with --no-sync, and
each mesh worker owns exactly one node and one GPU with no DDP rendezvous.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SBATCH_PATH = REPO_ROOT / "cluster" / "slurm" / "mesh_train.sbatch"


def test_watcher_cron_entry_uses_no_sync():
    """This cron fires once a minute; `uv sync --dry-run` reports an
    unconditional reinstall into the NFS-hosted venv even against an
    up-to-date lockfile, so a bare `uv run` here is ~1440 rewrites/day."""
    script = (
        Path(__file__).resolve().parents[2] / "cluster" / "launch_mesh_pool.sh"
    ).read_text()
    cron_lines = [
        ln for ln in script.splitlines() if "watch_training" in ln and "* * * * *" in ln
    ]
    assert cron_lines, "no watcher cron line found in launch_mesh_pool.sh"
    for line in cron_lines:
        assert "uv run --no-sync" in line


@pytest.mark.system
def test_sbatch_requests_exactly_one_node_and_no_ddp():
    text = SBATCH_PATH.read_text()

    assert "--nodes=1" in text
    assert "--ntasks=1" in text
    assert "MASTER_ADDR" not in text
    assert "MASTER_PORT" not in text


@pytest.mark.system
def test_worker_script_runs_uv_with_no_sync():
    """The project venv is on the NFS home and `uv sync` reinstalls into it
    unconditionally, so without --no-sync every worker in the pool rewrites
    the same shared venv at launch, simultaneously."""
    script = (
        Path(__file__).resolve().parents[2] / "cluster" / "run_mesh_worker.sh"
    ).read_text()
    assert "uv run --no-sync" in script
    assert "uv run python" not in script


@pytest.mark.system
def test_designated_host_gets_the_unit_files(install_watcher):
    install_watcher("node089")

    unit_dir = install_watcher.home / ".config" / "systemd" / "user"
    assert (unit_dir / "stern-watcher.service").exists()
    assert (unit_dir / "stern-watcher.timer").exists()
    assert "--no-sync" in (unit_dir / "stern-watcher.service").read_text()
