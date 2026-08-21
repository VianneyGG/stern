"""Tests for `video_encoding/scripts/launch_node_scaling_sweep.sh`.

Every path the launcher touches is env-overridable (the `SWEEP_*` pattern
from `sweep_queue.sh`), so these drive the real script against a fake
register command and a tmp_path roster -- no live coordinator state is
read or written.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "video_encoding"
    / "scripts"
    / "launch_node_scaling_sweep.sh"
)


@pytest.fixture
def env(tmp_path):
    """Returns (make_roster, run, calls_path, state_dir)."""
    log = tmp_path / "register_calls.txt"
    fake = tmp_path / "fake_register.sh"
    fake.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> ' + f'"{log}"\n'
    )
    fake.chmod(0o755)

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    roster = tmp_path / "ready.txt"

    def make_roster(n: int) -> None:
        roster.write_text("".join(f"fish{i:02d}\n" for i in range(n)))

    def run(wave: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(SCRIPT), wave],
            capture_output=True,
            text=True,
            env={
                "PATH": "/usr/bin:/bin",
                "NSS_READY_FILE": str(roster),
                "NSS_STATE_DIR": str(state_dir),
                "NSS_REGISTER_CMD": str(fake),
            },
        )

    return make_roster, run, log, state_dir


def _calls(log: Path) -> dict[str, list[str]]:
    """run_id -> the argv the launcher passed for that cell."""
    return {
        line.split()[0]: line.split()
        for line in log.read_text().splitlines()
        if line.strip()
    }


def test_script_has_no_syntax_errors():
    result = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_wave_a_registers_four_cells_with_disjoint_nodes(env):
    make_roster, run, log, _ = env
    make_roster(20)

    result = run("a")
    assert result.returncode == 0, result.stderr

    calls = _calls(log)
    assert sorted(calls) == ["nsl-n01a", "nsl-n02", "nsl-n04", "nsl-n08"]

    node_sets = {rid: argv[1].split(",") for rid, argv in calls.items()}
    assert [len(node_sets[r]) for r in ("nsl-n01a", "nsl-n02", "nsl-n04", "nsl-n08")] == [
        1,
        2,
        4,
        8,
    ]
    union = [n for nodes in node_sets.values() for n in nodes]
    assert len(union) == len(set(union)) == 15


def test_wave_b_registers_sixteen_and_one(env):
    make_roster, run, log, _ = env
    make_roster(20)

    result = run("b")
    assert result.returncode == 0, result.stderr

    calls = _calls(log)
    assert sorted(calls) == ["nsl-n01b", "nsl-n16"]
    big = set(calls["nsl-n16"][1].split(","))
    control = set(calls["nsl-n01b"][1].split(","))
    assert len(big) == 16
    assert len(control) == 1
    assert big.isdisjoint(control)


def test_passes_quorum_and_weighting_per_cell(env):
    make_roster, run, log, _ = env
    make_roster(20)
    assert run("a").returncode == 0

    calls = _calls(log)
    # argv: <run_id> <csv> <rounds> <outer_lr> <outer_momentum> <quorum_k> <weighting>
    assert calls["nsl-n08"][2:] == ["24", "0.7", "0.9", "6", "token"]
    assert calls["nsl-n02"][2:] == ["24", "0.7", "0.9", "2", "data"]


def test_exits_nonzero_and_registers_nothing_when_roster_too_small(env):
    """A partially registered wave would train cells on node sets the
    analysis cannot interpret, so the size check precedes every call."""
    make_roster, run, log, state_dir = env
    make_roster(10)

    result = run("a")
    assert result.returncode == 1
    assert not log.exists()
    assert list(state_dir.iterdir()) == []


def test_writes_the_partition_file(env):
    make_roster, run, log, state_dir = env
    make_roster(20)
    assert run("a").returncode == 0

    lines = (state_dir / "node_scaling_wave_a.txt").read_text().splitlines()
    assert len(lines) == 4

    calls = _calls(log)
    recorded = {line.split()[0]: line.split()[1] for line in lines}
    assert recorded == {rid: argv[1] for rid, argv in calls.items()}


def test_rejects_an_unknown_wave(env):
    make_roster, run, log, _ = env
    make_roster(20)

    result = run("c")
    assert result.returncode == 1
    assert not log.exists()
