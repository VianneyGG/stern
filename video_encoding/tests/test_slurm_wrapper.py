# video_encoding/tests/test_slurm_wrapper.py
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.system

WRAPPER = Path(__file__).resolve().parents[1] / "scripts" / "slurm_wrapper.sh"


def _run(*args, path_dir: Path):
    env = {"PATH": f"{path_dir}:/usr/bin:/bin"}
    return subprocess.run(
        ["bash", str(WRAPPER), *args], capture_output=True, text=True, env=env
    )


def _fake_bin(path_dir: Path, name: str) -> None:
    script = path_dir / name
    script.write_text(f'#!/usr/bin/env bash\necho "{name} called with: $*"\n')
    script.chmod(0o755)


def test_dispatches_allowed_commands(tmp_path):
    for name in ("sbatch", "scancel", "squeue", "sacct"):
        _fake_bin(tmp_path, name)

    for name in ("sbatch", "scancel", "squeue", "sacct"):
        result = _run(name, "--foo", "bar", path_dir=tmp_path)
        assert result.returncode == 0
        assert f"{name} called with: --foo bar" in result.stdout


def test_rejects_disallowed_command(tmp_path):
    result = _run("rm", "-rf", "/", path_dir=tmp_path)
    assert result.returncode == 2
    assert "not allowed" in result.stderr


def test_rejects_no_args(tmp_path):
    result = _run(path_dir=tmp_path)
    assert result.returncode != 0
