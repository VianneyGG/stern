from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from stern.watcher.shard_transfer import copy_shards

pytestmark = pytest.mark.integration


def _ok() -> MagicMock:
    return MagicMock(returncode=0, stdout="", stderr="")


def _fail(stderr: str = "rsync: connection unexpectedly closed") -> MagicMock:
    return MagicMock(returncode=12, stdout="", stderr=stderr)


def test_copy_shards_returns_the_shards_it_transferred():
    with patch(
        "stern.watcher.shard_transfer.subprocess.run", return_value=_ok()
    ):
        copied = copy_shards(
            "node040", "node032", ["shard-000009.tar", "shard-000057.tar"]
        )
    assert copied == ["shard-000009.tar", "shard-000057.tar"]


def test_copy_shards_returns_empty_when_rsync_fails():
    """A failed copy must report nothing copied -- assigning a shard that did
    not land is exactly the 2026-08-20 fault (a node named for a shard it does
    not have, which now raises inside the worker instead of skipping)."""
    with patch(
        "stern.watcher.shard_transfer.subprocess.run", return_value=_fail()
    ):
        copied = copy_shards("node040", "node032", ["shard-000009.tar"])
    assert copied == []


def test_copy_shards_returns_empty_when_rsync_is_missing():
    with patch(
        "stern.watcher.shard_transfer.subprocess.run",
        side_effect=OSError("rsync not found"),
    ):
        copied = copy_shards("node040", "node032", ["shard-000009.tar"])
    assert copied == []


def test_copy_shards_returns_empty_on_timeout():
    import subprocess as sp

    with patch(
        "stern.watcher.shard_transfer.subprocess.run",
        side_effect=sp.TimeoutExpired(cmd="rsync", timeout=600),
    ):
        copied = copy_shards("node040", "node032", ["shard-000009.tar"])
    assert copied == []


def test_copy_shards_is_a_noop_for_an_empty_shard_list():
    with patch(
        "stern.watcher.shard_transfer.subprocess.run", return_value=_ok()
    ) as run:
        assert copy_shards("node040", "node032", []) == []
    assert run.call_count == 0


def test_copy_shards_runs_rsync_on_the_destination_node():
    """Pins the whole argv: an ssh into dst_node running a target-side pull.

    rsync rejects a command with two remote endpoints, so building
    `rsync src:... dst:...` fails on every single call -- and since this
    function never raises, it failed silently (48/48 transfers returned []
    against the live cluster on 2026-08-22 with nothing in the log).
    """
    with patch(
        "stern.watcher.shard_transfer.subprocess.run", return_value=_ok()
    ) as run:
        copy_shards(
            "node040",
            "node032",
            ["shard-000009.tar", "shard-000057.tar"],
            mesh_data_dir="/Data/u/mesh",
        )
    assert run.call_args[0][0] == [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "node032",
        "mkdir -p /Data/u/mesh && rsync -aL --partial --files-from=- "
        "-e 'ssh -o StrictHostKeyChecking=no' node040:/Data/u/mesh/ /Data/u/mesh/",
    ]


def test_the_rsync_command_has_exactly_one_remote_endpoint():
    """The regression guard for the remote-to-remote bug, independent of the
    exact argv above: rsync refuses `host_a:... host_b:...`, so at most one of
    its two path arguments may carry a `host:` prefix."""
    with patch(
        "stern.watcher.shard_transfer.subprocess.run", return_value=_ok()
    ) as run:
        copy_shards("node040", "node032", ["shard-000009.tar"], mesh_data_dir="/D/m")
    remote_cmd = run.call_args[0][0][-1]
    rsync_args = remote_cmd.split("rsync ", 1)[1].split()
    paths = [a for a in rsync_args if "/D/m" in a]
    assert [p for p in paths if ":" in p] == ["node040:/D/m/"], paths


def test_copy_shards_sends_shard_names_over_stdin():
    """--files-from=- rather than interpolated filenames: the nested
    ssh/rsync quoting is a hazard once a batch has many shards."""
    shards = ["shard-000009.tar", "shard-000057.tar", "shard-000089.tar"]
    with patch(
        "stern.watcher.shard_transfer.subprocess.run", return_value=_ok()
    ) as run:
        copy_shards("node040", "node032", shards, mesh_data_dir="/Data/u/mesh")
    assert run.call_args.kwargs["input"] == (
        "shard-000009.tar\nshard-000057.tar\nshard-000089.tar\n"
    )
    assert not any("shard-" in arg for arg in run.call_args[0][0])


def test_a_failed_transfer_explains_itself_on_stderr(capsys):
    """Returning [] with no explanation is what let the remote-to-remote bug
    survive: the watcher recorded an under-filled pool and no reason."""
    with patch(
        "stern.watcher.shard_transfer.subprocess.run",
        return_value=_fail("The source and destination cannot both be remote."),
    ):
        assert copy_shards("node040", "node032", ["shard-000009.tar"]) == []
    err = capsys.readouterr().err
    assert "node040 -> node032" in err
    assert "cannot both be remote" in err


def test_copy_shards_honours_an_explicit_timeout():
    with patch(
        "stern.watcher.shard_transfer.subprocess.run", return_value=_ok()
    ) as run:
        copy_shards("node040", "node032", ["shard-000009.tar"], timeout_s=42)
    assert run.call_args.kwargs["timeout"] == 42


def test_copy_shards_dereferences_symlinked_shards():
    """A seeded shard is often a symlink into the source's own
    voxceleb2_shards, and several nodes had that wiped by the 2026-08-17
    reboot -- copying the link instead of the bytes lands a shard that lists
    as present and fails to open (node066/node056, 2026-08-22)."""
    with patch(
        "stern.watcher.shard_transfer.subprocess.run", return_value=_ok()
    ) as run:
        copy_shards("node040", "node032", ["shard-000009.tar"])
    remote_cmd = run.call_args[0][0][-1]
    flags = remote_cmd.split("rsync ", 1)[1].split()[0]
    assert "L" in flags, remote_cmd
