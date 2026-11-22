"""Fixtures shared by the cluster launcher tests."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

INSTALL_WATCHER_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "cluster"
    / "install_watcher.sh"
)


@pytest.fixture
def install_watcher(tmp_path):
    """Mirror the repo layout so the script's two-levels-up REPO_ROOT lands in
    tmp_path, and shim `hostname`/`systemctl` onto PATH so the test can assert
    on what the script asked systemd to do."""
    dest = tmp_path / "cluster" / "install_watcher.sh"
    dest.parent.mkdir(parents=True)
    shutil.copy2(INSTALL_WATCHER_SCRIPT, dest)
    dest.chmod(0o755)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "systemctl_calls.txt"
    (bin_dir / "systemctl").write_text(f'#!/bin/sh\necho "$@" >> {calls}\nexit 0\n')
    (bin_dir / "systemctl").chmod(0o755)

    home = tmp_path / "home"
    home.mkdir()

    def run(*args, hostname="node089.example.edu"):
        (bin_dir / "hostname").write_text(f'#!/bin/sh\necho "{hostname}"\n')
        (bin_dir / "hostname").chmod(0o755)
        return subprocess.run(
            ["bash", str(dest), *args],
            capture_output=True,
            text=True,
            env={
                "PATH": f"{bin_dir}:/usr/bin:/bin",
                "HOME": str(home),
            },
        )

    run.host_file = tmp_path / "runs" / "state" / "watcher_host"
    run.calls = calls
    run.home = home
    return run
