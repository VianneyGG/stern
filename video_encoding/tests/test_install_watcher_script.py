"""install_watcher.sh -- converges one host onto the watcher designation.

Replaces the timer-install block that submit_login-node.sh used to run
unconditionally, which silently made any host that ran a submit a second
watcher writer.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REAL_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "video_encoding"
    / "scripts"
    / "install_watcher.sh"
)


@pytest.fixture
def env(tmp_path):
    """Mirror the repo layout so the script's two-levels-up REPO_ROOT lands in
    tmp_path, and shim `hostname`/`systemctl` onto PATH so the test can assert
    on what the script asked systemd to do."""
    dest = tmp_path / "video_encoding" / "scripts" / "install_watcher.sh"
    dest.parent.mkdir(parents=True)
    shutil.copy2(REAL_SCRIPT, dest)
    dest.chmod(0o755)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "systemctl_calls.txt"
    (bin_dir / "systemctl").write_text(f'#!/bin/sh\necho "$@" >> {calls}\nexit 0\n')
    (bin_dir / "systemctl").chmod(0o755)

    home = tmp_path / "home"
    home.mkdir()

    def run(*args, hostname="node089.polytechnique.fr"):
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


def test_script_has_no_syntax_errors():
    result = subprocess.run(
        ["bash", "-n", str(REAL_SCRIPT)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_designating_this_host_writes_the_short_name_and_enables_the_timer(env):
    result = env("node089")

    assert result.returncode == 0, result.stderr
    assert env.host_file.read_text().strip() == "node089"
    assert "enable --now stern-watcher.timer" in env.calls.read_text()


def test_non_designated_host_stops_but_never_disables_the_timer(env):
    """`enable` writes a symlink under ~/.config/systemd/user, which is on the
    same shared NFS home -- so enablement is not a per-host property. A
    `disable` here deletes the designated host's symlink too, and whichever
    host converged last would win. `stop` is per-host runtime state, so it
    quiets this machine without touching what the other one reads."""
    result = env("node089", hostname="login-node.polytechnique.fr")

    assert result.returncode == 0, result.stderr
    assert env.host_file.read_text().strip() == "node089"
    calls = env.calls.read_text()
    assert "stop stern-watcher.timer" in calls
    assert "disable" not in calls
    assert "enable" not in calls


def test_no_argument_converges_against_the_existing_designation(env):
    env.host_file.parent.mkdir(parents=True, exist_ok=True)
    env.host_file.write_text("node089\n")

    result = env(hostname="login-node.polytechnique.fr")

    assert result.returncode == 0, result.stderr
    assert env.host_file.read_text().strip() == "node089"
    assert "stop stern-watcher.timer" in env.calls.read_text()


def test_no_argument_without_a_designation_fails_rather_than_guessing(env):
    result = env()

    assert result.returncode != 0
    assert not env.host_file.exists()
    assert not env.calls.exists()


def test_designated_host_gets_the_unit_files(env):
    env("node089")

    unit_dir = env.home / ".config" / "systemd" / "user"
    assert (unit_dir / "stern-watcher.service").exists()
    assert (unit_dir / "stern-watcher.timer").exists()
    assert "--no-sync" in (unit_dir / "stern-watcher.service").read_text()
