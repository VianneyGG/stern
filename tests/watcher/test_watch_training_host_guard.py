"""Host designation for the mesh watcher.

The watcher runs from a systemd --user timer on a machine whose $HOME is the
shared NFS mount, so two hosts can have the timer enabled at once and
`fcntl.flock` provides no mutual exclusion across them on this NFSv3 export
(verified experimentally 2026-08-20). Designation is therefore file-based:
runs/state/watcher_host names the single host allowed to tick.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

pytestmark = pytest.mark.integration


def test_log_line_carries_the_short_hostname(tmp_path, monkeypatch):
    """Without attribution a shared watcher.log cannot say which host acted."""
    import stern.watcher.watch_training as watch_mod

    monkeypatch.setattr(watch_mod, "LOG_PATH", tmp_path / "watcher.log")
    monkeypatch.setattr(watch_mod, "HOSTNAME", "node089")

    watch_mod._log("flags=0 actions=none")

    assert "[node089] flags=0 actions=none" in (tmp_path / "watcher.log").read_text()


def test_designated_when_file_names_this_host(tmp_path):
    from stern.watcher.watch_training import is_designated_watcher

    host_file = tmp_path / "watcher_host"
    host_file.write_text("node089\n")

    assert is_designated_watcher(host_file, "node089") is True


def test_not_designated_when_file_names_another_host(tmp_path):
    from stern.watcher.watch_training import is_designated_watcher

    host_file = tmp_path / "watcher_host"
    host_file.write_text("node089\n")

    assert is_designated_watcher(host_file, "login-node") is False


def test_designation_ignores_surrounding_whitespace_and_case(tmp_path):
    from stern.watcher.watch_training import is_designated_watcher

    host_file = tmp_path / "watcher_host"
    host_file.write_text("  Node089  \n")

    assert is_designated_watcher(host_file, "node089") is True


def test_designation_compares_short_names_when_either_side_is_an_fqdn(tmp_path):
    """`hostname` is short here but socket.gethostname() returns the FQDN, and
    a hand-edited designation file may carry either form."""
    from stern.watcher.watch_training import is_designated_watcher

    host_file = tmp_path / "watcher_host"
    host_file.write_text("node089.example.edu\n")

    assert is_designated_watcher(host_file, "node089.example.edu") is True
    assert is_designated_watcher(host_file, "node089") is True


def test_missing_designation_file_raises_rather_than_defaulting(tmp_path):
    """Fail closed: a silent default would resurrect the two-writer case the
    designation file exists to prevent."""
    from stern.watcher.watch_training import is_designated_watcher

    with pytest.raises(OSError, match="absent"):
        is_designated_watcher(tmp_path / "absent", "node089")


def test_empty_designation_file_is_not_a_designation(tmp_path):
    from stern.watcher.watch_training import is_designated_watcher

    host_file = tmp_path / "watcher_host"
    host_file.write_text("\n")

    assert is_designated_watcher(host_file, "node089") is False


def _stub_paths(watch_mod, tmp_path, monkeypatch):
    """Point every module-level path at tmp_path without creating anything."""
    state_dir = tmp_path / "runs" / "state"
    monkeypatch.setattr(watch_mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(watch_mod, "STATE_DIR", state_dir)
    monkeypatch.setattr(watch_mod, "RESULTS_DIR", tmp_path / "runs" / "results")
    monkeypatch.setattr(watch_mod, "LOG_PATH", state_dir / "watcher.log")
    monkeypatch.setattr(watch_mod, "LOCK_PATH", state_dir / "watch_training.lock")
    monkeypatch.setattr(watch_mod, "HOST_PATH", tmp_path / "watcher_host")
    return state_dir


def test_non_designated_host_ticks_without_any_side_effect(tmp_path, monkeypatch):
    """`main` truncates the shared lock file with open(..., "w") before it even
    attempts flock, so the guard has to run ahead of that, not just ahead of
    squeue."""
    import stern.watcher.watch_training as watch_mod

    state_dir = _stub_paths(watch_mod, tmp_path, monkeypatch)
    monkeypatch.setattr(watch_mod, "HOSTNAME", "login-node")
    (tmp_path / "watcher_host").write_text("node089\n")

    with patch(
        "stern.watcher.watch_training.refresh_squeue_snapshot"
    ) as mock_refresh:
        rc = watch_mod.main()

    assert rc == 0
    mock_refresh.assert_not_called()
    assert not state_dir.exists()


def test_non_designated_host_does_not_write_the_shared_log(tmp_path, monkeypatch):
    """One line a minute from a decommissioned host would swamp the very log
    used to audit the migration; the rejection belongs on stderr."""
    import stern.watcher.watch_training as watch_mod

    state_dir = _stub_paths(watch_mod, tmp_path, monkeypatch)
    state_dir.mkdir(parents=True)
    monkeypatch.setattr(watch_mod, "HOSTNAME", "login-node")
    (tmp_path / "watcher_host").write_text("node089\n")

    watch_mod.main()

    assert (state_dir / "watcher.log").exists() is False


def test_missing_designation_file_exits_nonzero(tmp_path, monkeypatch):
    import stern.watcher.watch_training as watch_mod

    state_dir = _stub_paths(watch_mod, tmp_path, monkeypatch)
    monkeypatch.setattr(watch_mod, "HOSTNAME", "node089")

    with patch(
        "stern.watcher.watch_training.refresh_squeue_snapshot"
    ) as mock_refresh:
        rc = watch_mod.main()

    assert rc == 1
    mock_refresh.assert_not_called()
    assert not state_dir.exists()


def test_designated_host_proceeds_past_the_guard(tmp_path, monkeypatch):
    import stern.watcher.watch_training as watch_mod

    _stub_paths(watch_mod, tmp_path, monkeypatch)
    monkeypatch.setattr(watch_mod, "HOSTNAME", "node089")
    (tmp_path / "watcher_host").write_text("node089\n")

    with patch(
        "stern.watcher.watch_training.refresh_squeue_snapshot"
    ) as mock_refresh:
        rc = watch_mod.main()

    assert rc == 0
    mock_refresh.assert_called_once()
