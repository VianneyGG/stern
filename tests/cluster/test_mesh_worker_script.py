from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest


pytestmark = pytest.mark.system

REPO_ROOT = Path(__file__).resolve().parents[2]
SBATCH_PATH = REPO_ROOT / "cluster" / "slurm" / "mesh_train.sbatch"
WORKER_SCRIPT_PATH = REPO_ROOT / "cluster" / "run_mesh_worker.sh"


def _write_assignments(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "shard_assignments_RUN1.json"
    path.write_text(json.dumps(payload))
    return path


def test_sbatch_script_has_no_syntax_errors():
    result = subprocess.run(["bash", "-n", str(SBATCH_PATH)], capture_output=True)

    assert result.returncode == 0, result.stderr.decode()



def test_worker_script_has_no_syntax_errors():
    result = subprocess.run(
        ["bash", "-n", str(WORKER_SCRIPT_PATH)], capture_output=True
    )

    assert result.returncode == 0, result.stderr.decode()
