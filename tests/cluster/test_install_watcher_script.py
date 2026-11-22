"""install_watcher.sh -- converges one host onto the watcher designation.

Replaces the timer-install block that submit_to_cluster.sh used to run
unconditionally, which silently made any host that ran a submit a second
watcher writer.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.system


def test_designating_this_host_writes_the_short_name_and_enables_the_timer(install_watcher):
    result = install_watcher("node089")

    assert result.returncode == 0, result.stderr
    assert install_watcher.host_file.read_text().strip() == "node089"
    assert "enable --now stern-watcher.timer" in install_watcher.calls.read_text()


def test_non_designated_host_stops_but_never_disables_the_timer(install_watcher):
    """`enable` writes a symlink under ~/.config/systemd/user, which is on the
    same shared NFS home -- so enablement is not a per-host property. A
    `disable` here deletes the designated host's symlink too, and whichever
    host converged last would win. `stop` is per-host runtime state, so it
    quiets this machine without touching what the other one reads."""
    result = install_watcher("node089", hostname="login-node.example.edu")

    assert result.returncode == 0, result.stderr
    assert install_watcher.host_file.read_text().strip() == "node089"
    calls = install_watcher.calls.read_text()
    assert "stop stern-watcher.timer" in calls
    assert "disable" not in calls
    assert "enable" not in calls


def test_no_argument_converges_against_the_existing_designation(install_watcher):
    install_watcher.host_file.parent.mkdir(parents=True, exist_ok=True)
    install_watcher.host_file.write_text("node089\n")

    result = install_watcher(hostname="login-node.example.edu")

    assert result.returncode == 0, result.stderr
    assert install_watcher.host_file.read_text().strip() == "node089"
    assert "stop stern-watcher.timer" in install_watcher.calls.read_text()


def test_no_argument_without_a_designation_fails_rather_than_guessing(install_watcher):
    result = install_watcher()

    assert result.returncode != 0
    assert not install_watcher.host_file.exists()
    assert not install_watcher.calls.exists()
