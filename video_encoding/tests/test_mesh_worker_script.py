from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from video_encoding.mesh.assignment import read_my_shards

pytestmark = pytest.mark.system

REPO_ROOT = Path(__file__).resolve().parents[2]
SBATCH_PATH = REPO_ROOT / "video_encoding" / "slurm" / "mesh_train.sbatch"
WORKER_SCRIPT_PATH = REPO_ROOT / "video_encoding" / "scripts" / "run_mesh_worker.sh"


def _write_assignments(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "shard_assignments_RUN1.json"
    path.write_text(json.dumps(payload))
    return path


def test_read_my_shards_returns_only_this_hosts_list(tmp_path):
    path = _write_assignments(tmp_path, {"node001": ["s0", "s1"], "node002": ["s2"]})

    assert read_my_shards(path, "node001") == ["s0", "s1"]


def test_read_my_shards_raises_for_an_unassigned_host(tmp_path):
    path = _write_assignments(tmp_path, {"node001": ["s0", "s1"], "node002": ["s2"]})

    with pytest.raises(KeyError, match="node089"):
        read_my_shards(path, "node089")


def test_sbatch_script_has_no_syntax_errors():
    result = subprocess.run(["bash", "-n", str(SBATCH_PATH)], capture_output=True)

    assert result.returncode == 0, result.stderr.decode()


def test_sbatch_requests_exactly_one_node_and_no_ddp():
    text = SBATCH_PATH.read_text()

    assert "--nodes=1" in text
    assert "--ntasks=1" in text
    assert "MASTER_ADDR" not in text
    assert "MASTER_PORT" not in text


def test_worker_script_has_no_syntax_errors():
    result = subprocess.run(
        ["bash", "-n", str(WORKER_SCRIPT_PATH)], capture_output=True
    )

    assert result.returncode == 0, result.stderr.decode()


def test_worker_script_runs_uv_with_no_sync():
    """The project venv is on the NFS home and `uv sync` reinstalls into it
    unconditionally, so without --no-sync every worker in the pool rewrites
    the same shared venv at launch, simultaneously."""
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "run_mesh_worker.sh"
    ).read_text()
    assert "uv run --no-sync" in script
    assert "uv run python" not in script
