# video_encoding/tests/test_slurm_wrapper.py
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.system

WRAPPER = Path(__file__).resolve().parents[1] / "scripts" / "slurm_wrapper.sh"


def _run(*args, path_dir: Path, gate_exit: int = 0):
    # The wrapper gates sbatch on its sibling class_hours.py; run a copy beside
    # a stub gate so the result doesn't depend on the hour the suite runs at.
    scripts = path_dir / "scripts"
    scripts.mkdir(exist_ok=True)
    shutil.copy2(WRAPPER, scripts / WRAPPER.name)
    (scripts / "class_hours.py").write_text(f"raise SystemExit({gate_exit})\n")
    env = {"PATH": f"{path_dir}:/usr/bin:/bin"}
    return subprocess.run(
        ["bash", str(scripts / WRAPPER.name), *args],
        capture_output=True,
        text=True,
        env=env,
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


def test_sbatch_refused_by_class_hours_gate_but_not_scancel(tmp_path):
    for name in ("sbatch", "scancel"):
        _fake_bin(tmp_path, name)
    refused = _run("sbatch", "x", path_dir=tmp_path, gate_exit=1)
    assert refused.returncode == 1
    assert "called with" not in refused.stdout
    assert _run("scancel", "7", path_dir=tmp_path, gate_exit=1).returncode == 0


def test_rejects_disallowed_command(tmp_path):
    result = _run("rm", "-rf", "/", path_dir=tmp_path)
    assert result.returncode == 2
    assert "not allowed" in result.stderr


def test_rejects_no_args(tmp_path):
    result = _run(path_dir=tmp_path)
    assert result.returncode != 0
