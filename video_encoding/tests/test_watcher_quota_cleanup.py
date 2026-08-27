from __future__ import annotations

import os
import subprocess
from unittest.mock import MagicMock, patch

from video_encoding.watcher.quota_cleanup import (
    orphaned_tmp_checkpoints,
    remediate_quota_pressure,
    safe_node_checkpoints,
    stale_outer_momentum_checkpoints,
)

NOW = 10_000.0
_REAL_SUBPROCESS_RUN = subprocess.run  # captured before any test patches subprocess.run


def _touch(path, *, age_s):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")
    mtime = NOW - age_s
    os.utime(path, (mtime, mtime))


def _run_via_real_shell(cmd, **kwargs):
    """subprocess.run stand-in that mimics what an *actual* ssh does to a
    multi-element argv: sshd joins every argument after the hostname with a
    single space and hands the result to ONE remote shell invocation, so any
    quoting the caller built into separate argv elements (e.g. ["sh", "-c",
    "..."]) is destroyed before it's ever parsed. `["ssh", host, *rest]` ->
    `sh -c "<' '.join(rest)>"`, run for real against a local temp dir so a
    regression re-introduces the exact bug caught in review: quota_cleanup's
    old form built a correctly-quoted command as separate argv elements and
    it silently listed/deleted in $HOME instead of STERN_CHECKPOINT_ROOT."""
    assert cmd[0] == "ssh"
    remote_command = " ".join(cmd[2:])
    return _REAL_SUBPROCESS_RUN(
        ["sh", "-c", remote_command], capture_output=True, text=True, timeout=10
    )


# --- orphaned_tmp_checkpoints -------------------------------------------


def test_orphaned_tmp_checkpoints_includes_old_tmp_for_inactive_run(tmp_path):
    _touch(tmp_path / "dead-run_3_node001.pt.tmp", age_s=7200)
    result = orphaned_tmp_checkpoints(tmp_path, {"live-run"}, max_age_s=3600, now=NOW)
    assert [p.name for p in result] == ["dead-run_3_node001.pt.tmp"]


def test_orphaned_tmp_checkpoints_excludes_active_run(tmp_path):
    _touch(tmp_path / "live-run_3_node001.pt.tmp", age_s=7200)
    result = orphaned_tmp_checkpoints(tmp_path, {"live-run"}, max_age_s=3600, now=NOW)
    assert result == []


def test_orphaned_tmp_checkpoints_excludes_recent_tmp(tmp_path):
    _touch(tmp_path / "dead-run_3_node001.pt.tmp", age_s=100)
    result = orphaned_tmp_checkpoints(tmp_path, {"live-run"}, max_age_s=3600, now=NOW)
    assert result == []


def test_orphaned_tmp_checkpoints_plain_pt_is_never_matched(tmp_path):
    # sweep_orphaned_pending already owns *.pt -- this function only ever
    # globs *.pt.tmp, so a plain .pt file is invisible to it regardless of
    # age/activity.
    _touch(tmp_path / "dead-run_3_node001.pt", age_s=7200)
    result = orphaned_tmp_checkpoints(tmp_path, {"live-run"}, max_age_s=3600, now=NOW)
    assert result == []


def test_orphaned_tmp_checkpoints_empty_active_set_is_unknown_not_nothing(tmp_path):
    _touch(tmp_path / "dead-run_3_node001.pt.tmp", age_s=7200)
    result = orphaned_tmp_checkpoints(tmp_path, set(), max_age_s=3600, now=NOW)
    assert result == []


def test_orphaned_tmp_checkpoints_missing_dir_returns_empty(tmp_path):
    result = orphaned_tmp_checkpoints(
        tmp_path / "does_not_exist", {"live-run"}, max_age_s=3600, now=NOW
    )
    assert result == []


# --- stale_outer_momentum_checkpoints --------------------------------------


def test_stale_outer_momentum_checkpoints_includes_done_run(tmp_path):
    mesh_shared = tmp_path / "stern_mesh"
    path = mesh_shared / "checkpoints" / "done-run" / "outer_momentum.pt"
    path.parent.mkdir(parents=True)
    path.write_text("x")

    result = stale_outer_momentum_checkpoints(mesh_shared, {"done-run"})
    assert len(result) == 1
    assert result[0].name == "outer_momentum.pt"
    assert result[0].parent.name == "done-run"


def test_stale_outer_momentum_checkpoints_excludes_non_done_run(tmp_path):
    mesh_shared = tmp_path / "stern_mesh"
    path = mesh_shared / "checkpoints" / "live-run" / "outer_momentum.pt"
    path.parent.mkdir(parents=True)
    path.write_text("x")

    result = stale_outer_momentum_checkpoints(mesh_shared, {"done-run"})
    assert result == []


def test_stale_outer_momentum_checkpoints_never_touches_master_pt(tmp_path):
    mesh_shared = tmp_path / "stern_mesh"
    run_dir = mesh_shared / "checkpoints" / "done-run"
    run_dir.mkdir(parents=True)
    (run_dir / "master.pt").write_text("x")

    result = stale_outer_momentum_checkpoints(mesh_shared, {"done-run"})
    assert result == []


def test_stale_outer_momentum_checkpoints_missing_file_is_skipped(tmp_path):
    mesh_shared = tmp_path / "stern_mesh"
    result = stale_outer_momentum_checkpoints(mesh_shared, {"done-run"})
    assert result == []


# --- safe_node_checkpoints ------------------------------------------------


def test_safe_node_checkpoints_matches_done_run_pt():
    assert safe_node_checkpoints(["done-run.pt"], {"done-run"}) == ["done-run.pt"]


def test_safe_node_checkpoints_matches_done_run_push_residual():
    names = ["done-run_push_residual.pt"]
    assert safe_node_checkpoints(names, {"done-run"}) == names


def test_safe_node_checkpoints_matches_done_run_tmp():
    names = ["done-run.pt.tmp"]
    assert safe_node_checkpoints(names, {"done-run"}) == names


def test_safe_node_checkpoints_excludes_non_done_run():
    result = safe_node_checkpoints(["running-run.pt"], {"done-run"})
    assert result == []


def test_safe_node_checkpoints_always_includes_tb_dir():
    result = safe_node_checkpoints(["tb"], done_run_ids=set())
    assert result == ["tb"]


def test_safe_node_checkpoints_excludes_unrelated_names():
    names = ["master.pt", "outer_momentum.pt", "notes.txt"]
    result = safe_node_checkpoints(names, {"master", "outer_momentum", "notes.txt"})
    # "master.pt" -> run_id "master" (in done set) -- still legitimately
    # matched by the .pt suffix; the real master.pt/outer_momentum.pt files
    # live under a different path entirely (mesh_shared/checkpoints/, not
    # STERN_CHECKPOINT_ROOT) and are never listed as inputs to this
    # function in production, so this only proves the suffix logic itself.
    assert "notes.txt" not in result


# --- remediate_quota_pressure ---------------------------------------------


def test_remediate_quota_pressure_home_deletes_stale_tmp(tmp_path):
    mesh_shared = tmp_path / "stern_mesh"
    pending = mesh_shared / "checkpoints" / "pending"
    stale = pending / "dead-run_3_node001.pt.tmp"
    _touch(stale, age_s=7200)

    summary = remediate_quota_pressure(
        "home",
        mesh_shared=mesh_shared,
        done_run_ids=set(),
        active_run_ids={"live-run"},
        now=NOW,
    )

    assert summary is not None
    assert "dead-run_3_node001.pt.tmp" in summary
    assert not stale.exists()


def test_remediate_quota_pressure_home_deletes_stale_outer_momentum(tmp_path):
    mesh_shared = tmp_path / "stern_mesh"
    momentum = mesh_shared / "checkpoints" / "done-run" / "outer_momentum.pt"
    momentum.parent.mkdir(parents=True)
    momentum.write_text("x")
    master = mesh_shared / "checkpoints" / "done-run" / "master.pt"
    master.write_text("x")

    summary = remediate_quota_pressure(
        "home",
        mesh_shared=mesh_shared,
        done_run_ids={"done-run"},
        active_run_ids={"live-run"},
        now=NOW,
    )

    assert summary is not None
    assert "outer_momentum.pt" in summary
    assert not momentum.exists()
    assert master.exists()  # the trained-model result is never touched


def test_remediate_quota_pressure_home_returns_none_when_nothing_stale(tmp_path):
    mesh_shared = tmp_path / "stern_mesh"
    summary = remediate_quota_pressure(
        "home",
        mesh_shared=mesh_shared,
        done_run_ids=set(),
        active_run_ids={"live-run"},
        now=NOW,
    )
    assert summary is None


def test_remediate_quota_pressure_node_lists_then_deletes(tmp_path):
    list_result = MagicMock(returncode=0, stdout="done-run.pt\nrunning-run.pt\ntb\n")
    delete_result = MagicMock(returncode=0, stdout="")
    with patch("subprocess.run", side_effect=[list_result, delete_result]) as mock_run:
        summary = remediate_quota_pressure(
            "node:node058",
            mesh_shared=tmp_path,
            done_run_ids={"done-run"},
            active_run_ids=set(),
            now=NOW,
        )

    assert summary is not None
    assert "done-run.pt" in summary and "tb" in summary
    assert "running-run.pt" not in summary
    assert mock_run.call_count == 2
    list_cmd = mock_run.call_args_list[0][0][0]
    assert list_cmd[:2] == ["ssh", "node058"]
    delete_cmd = mock_run.call_args_list[1][0][0]
    assert delete_cmd[:2] == ["ssh", "node058"]
    # The remote command must be ONE string (not split across further argv
    # elements) -- ssh joins everything after the hostname with a plain
    # space before a real remote shell ever sees it, so splitting it here
    # would silently re-parse wrong on a real host (see
    # test_remediate_quota_pressure_node_real_shell_round_trip).
    assert len(delete_cmd) == 3
    remote_command = delete_cmd[2]
    assert "done-run.pt" in remote_command
    assert "tb" in remote_command
    assert "running-run.pt" not in remote_command


def test_remediate_quota_pressure_node_returns_none_when_nothing_matches():
    list_result = MagicMock(returncode=0, stdout="running-run.pt\n")
    with patch("subprocess.run", return_value=list_result) as mock_run:
        summary = remediate_quota_pressure(
            "node:node058",
            mesh_shared=None,
            done_run_ids={"done-run"},
            active_run_ids=set(),
            now=NOW,
        )
    assert summary is None
    mock_run.assert_called_once()  # delete step never attempted


def test_remediate_quota_pressure_node_returns_none_when_list_fails():
    with patch("subprocess.run", side_effect=OSError("ssh unreachable")):
        summary = remediate_quota_pressure(
            "node:deadnode",
            mesh_shared=None,
            done_run_ids={"done-run"},
            active_run_ids=set(),
            now=NOW,
        )
    assert summary is None


def test_remediate_quota_pressure_node_returns_none_when_delete_fails():
    list_result = MagicMock(returncode=0, stdout="done-run.pt\n")
    delete_result = MagicMock(returncode=1, stdout="rm: permission denied")
    with patch("subprocess.run", side_effect=[list_result, delete_result]):
        summary = remediate_quota_pressure(
            "node:node058",
            mesh_shared=None,
            done_run_ids={"done-run"},
            active_run_ids=set(),
            now=NOW,
        )
    assert summary is None


def test_remediate_quota_pressure_node_real_shell_round_trip(tmp_path, monkeypatch):
    """End-to-end regression for the ssh-argv-flattening bug caught in
    review: quota_cleanup used to build `["ssh", node, "sh", "-c", "cd ... &&
    rm ..."]`, which is correctly quoted as a Python list but not as what
    sshd actually runs -- ssh joins every arg after the hostname with a
    single space and re-parses the result in ONE remote shell, which turned
    `sh -c ls -1 "$STERN..."` into a bare `ls` of $HOME (via `-c`'s own
    `[command_name [args...]]` form swallowing the real path as $0/$1) and
    the delete into `cd` running in a throwaway subshell followed by
    `rm -r -f -- sh <names>` in the OUTER shell's cwd -- i.e. $HOME, not
    the checkpoint dir. _run_via_real_shell reproduces that join+reparse for
    real; this test would delete `done-run.pt` from $HOME (or fail to find
    it) under the pre-fix code instead of deleting it from ckpt_root."""
    ckpt_root = tmp_path / "checkpoints"
    ckpt_root.mkdir()
    (ckpt_root / "done-run.pt").write_text("x")
    (ckpt_root / "running-run.pt").write_text("x")
    (ckpt_root / "tb").mkdir()
    (ckpt_root / "tb" / "events.out").write_text("x")
    monkeypatch.setenv("STERN_CHECKPOINT_ROOT", str(ckpt_root))

    with patch("subprocess.run", side_effect=_run_via_real_shell):
        summary = remediate_quota_pressure(
            "node:node058",
            mesh_shared=None,
            done_run_ids={"done-run"},
            active_run_ids=set(),
            now=NOW,
        )

    assert summary is not None
    remaining = {p.name for p in ckpt_root.iterdir()}
    assert remaining == {"running-run.pt"}


def test_remediate_quota_pressure_unknown_location_returns_none():
    summary = remediate_quota_pressure(
        "mystery",
        mesh_shared=None,
        done_run_ids=set(),
        active_run_ids=set(),
        now=NOW,
    )
    assert summary is None
