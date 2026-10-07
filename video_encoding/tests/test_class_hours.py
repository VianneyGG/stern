"""The class-hours rule (scripts/class_hours.py) and every place it must hold.

No cluster job may run Mon-Fri 08:00-20:00 Europe/Paris. conftest.py pins the
gate open for the rest of the suite; this module is exempt and uses the real
function, or pins it CLOSED to prove each launch path honours it.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from video_encoding.scripts import class_hours
from video_encoding.scripts.class_hours import PARIS, blocked, parse_walltime
from video_encoding.tests.test_mesh_tick import _run_config, _seed_shard_map
from video_encoding.watcher.mesh_tick import _submit_worker_job, advance_one_run
from video_encoding.watcher.registry import JobRecord
from video_encoding.watcher.remediate import remediate_dead_job
from video_encoding.watcher.state import WatcherState

REPO = Path(__file__).resolve().parents[2]
H = 3600


def _at(y, mo, d, h, mi=0):
    return datetime(y, mo, d, h, mi, tzinfo=PARIS)


# --- the rule ---------------------------------------------------------------

# 2026-09-21 is a Monday.
@pytest.mark.parametrize(
    ("when", "walltime", "refused"),
    [
        (_at(2026, 9, 21, 7, 59), 0, False),  # instant job before class
        (_at(2026, 9, 21, 6, 0), 2 * H, False),  # ends exactly at 08:00
        (_at(2026, 9, 21, 6, 1), 2 * H, True),  # would run into 08:00
        (_at(2026, 9, 21, 8, 0), 0, True),
        (_at(2026, 9, 21, 12, 0), 60, True),
        (_at(2026, 9, 25, 19, 59), 0, True),  # Friday, last class minute
        (_at(2026, 9, 25, 20, 0), 2 * H, False),  # Friday evening
        (_at(2026, 9, 26, 10, 0), 2 * H, False),  # Saturday
        (_at(2026, 9, 27, 23, 0), 2 * H, False),  # Sunday night into Monday 01:00
        (_at(2026, 9, 27, 23, 0), 10 * H, True),  # ...but not into Monday 09:00
        (_at(2026, 9, 21, 22, 0), 2 * H, False),  # weekday night
        (_at(2026, 9, 25, 7, 0), 4 * 86400, True),  # multi-day job spanning class
    ],
)
def test_rule(when, walltime, refused):
    assert (blocked(when, walltime) is not None) is refused


def test_rule_uses_paris_wall_clock_across_dst():
    # 2026-03-29 CET->CEST. Monday 06:00 local is 04:00 UTC after the switch;
    # a UTC-naive rule would put class start at 07:00 UTC and let this through.
    assert blocked(_at(2026, 3, 30, 6, 0), 2 * H) is None
    assert blocked(_at(2026, 3, 30, 6, 30), 2 * H) is not None
    # 2026-10-25 CEST->CET: Monday 07:30 local + 1h reaches class.
    assert blocked(_at(2026, 10, 26, 7, 30), H) is not None
    # Walltime is real elapsed time. Sunday 2026-10-25 01:00 CEST to Monday
    # 08:00 CET is 32h, not the 31h the wall clock shows (the fall-back hour
    # repeats), so a 32h job ends exactly at class start and 32h+1s overlaps.
    assert blocked(_at(2026, 10, 25, 1, 0), 32 * H) is None
    assert blocked(_at(2026, 10, 25, 1, 0), 32 * H + 1) is not None


def test_rule_accepts_epoch_seconds():
    ts = _at(2026, 9, 21, 10, 0).timestamp()
    assert blocked(ts) is not None
    assert blocked(_at(2026, 9, 21, 21, 0).timestamp()) is None


@pytest.mark.parametrize(
    ("spec", "secs"),
    [
        ("1:59:58", 7198),
        ("59:58", 3598),
        ("30", 1800),
        ("2-00:00:00", 2 * 86400),
        ("UNLIMITED", class_hours.MAX_WALLTIME_S),  # fail closed
        ("NOT_SET", class_hours.MAX_WALLTIME_S),
        ("", class_hours.MAX_WALLTIME_S),
    ],
)
def test_parse_walltime(spec, secs):
    assert parse_walltime(spec) == secs


# --- every launch path honours it --------------------------------------------


@pytest.fixture
def closed(monkeypatch):
    monkeypatch.setattr(class_hours, "blocked", lambda *a, **k: "class hours: test")


CLASS_TIME = _at(2026, 9, 21, 10, 0).timestamp()


def test_watcher_holds_a_new_run_without_submitting_or_advancing(tmp_path):
    mesh_shared = tmp_path / "mesh_shared"
    _seed_shard_map(mesh_shared)
    state = WatcherState()
    kw = dict(
        state=state,
        repo_root=tmp_path,
        mesh_shared=mesh_shared,
        mesh_root=tmp_path / "mesh_root",
        now=CLASS_TIME,
    )
    with patch("video_encoding.watcher.mesh_tick.subprocess.run") as run:
        first = advance_one_run(_run_config(), **kw)
        second = advance_one_run(_run_config(), **kw)
    run.assert_not_called()
    assert state.run("RUN1").status == "ASSIGNING"
    assert "launch held" in first
    assert second is None  # reported once, not every tick
    assert not (mesh_shared / "metadata" / "shard_assignments_RUN1.json").exists()


def test_watcher_submits_the_same_run_outside_class_hours(tmp_path):
    # Control for the test above: same inputs, evening -> it launches.
    mesh_shared = tmp_path / "mesh_shared"
    _seed_shard_map(mesh_shared)
    state = WatcherState()
    fake = MagicMock(returncode=0, stdout="Submitted batch job 101\n")
    evening = _at(2026, 9, 21, 21, 0).timestamp()
    with (
        patch("video_encoding.watcher.mesh_tick.subprocess.run", return_value=fake) as run,
        patch("video_encoding.watcher.mesh_tick.time.time", return_value=evening),
    ):
        advance_one_run(
            _run_config(),
            state=state,
            repo_root=tmp_path,
            mesh_shared=mesh_shared,
            mesh_root=tmp_path / "mesh_root",
            now=evening,
        )
    assert any(c.args[0][0] == "sbatch" for c in run.call_args_list)
    assert state.run("RUN1").status == "RUNNING"


def test_watcher_skips_chunk_relaunch_and_recovery_in_class_hours(tmp_path):
    state = WatcherState()
    state.run("RUN1").status = "RUNNING"
    with (
        patch("video_encoding.watcher.mesh_tick._relaunch_chunk_if_finished") as chunk,
        patch("video_encoding.watcher.mesh_tick._recover_pool") as recover,
    ):
        advance_one_run(
            _run_config(chunk_rounds=5),
            state=state,
            repo_root=tmp_path,
            mesh_shared=tmp_path / "mesh_shared",
            mesh_root=tmp_path / "mesh_root",
            now=CLASS_TIME,
        )
    chunk.assert_not_called()
    recover.assert_not_called()
    assert state.run("RUN1").status == "RUNNING"


def test_submit_worker_job_refuses_whatever_the_caller(tmp_path, closed):
    registry = tmp_path / "jobs_registry.json"
    with patch("video_encoding.watcher.mesh_tick.subprocess.run") as run:
        job = _submit_worker_job(_run_config(), "node001", tmp_path, registry, "t")
    run.assert_not_called()
    assert job is None
    assert not registry.exists()


def test_remediate_dead_job_refuses(tmp_path, closed):
    record = JobRecord("100", "exp", "tag", "2026-07-15T09:00:00Z", "node001")
    with patch("subprocess.run") as run:
        assert remediate_dead_job(record, repo_root=tmp_path) is None
    run.assert_not_called()


def _stub_gate(dest_dir: Path, exit_code: int) -> None:
    (dest_dir / "class_hours.py").write_text(f"raise SystemExit({exit_code})\n")


def test_register_mesh_run_refuses_when_gate_closed(tmp_path):
    scripts = tmp_path / "video_encoding" / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(REPO / "video_encoding/scripts/register_mesh_run.sh", scripts)
    conf = tmp_path / "video_encoding" / "conf" / "experiment"
    conf.mkdir(parents=True)
    (conf / "videomae_mesh_ssl.yaml").touch()
    (tmp_path / "runs" / "state").mkdir(parents=True)
    args = ["bash", str(scripts / "register_mesh_run.sh"), "R", "node055", "20", "1.0", "0.0", "1"]
    _stub_gate(scripts, 1)
    refused = subprocess.run(args, capture_output=True, text=True)
    configs = tmp_path / "runs" / "state" / "mesh_run_configs.json"
    assert refused.returncode != 0
    assert not configs.exists()
    _stub_gate(scripts, 0)  # control: the same call registers when open
    ok = subprocess.run(args, capture_output=True, text=True)
    assert ok.returncode == 0, ok.stderr
    assert json.loads(configs.read_text())[0]["run_id"] == "R"


def test_job_script_exits_before_work_when_gate_closed(tmp_path):
    # Mirror mesh_train.sbatch's layout; the worker it would run records a call.
    (tmp_path / "video_encoding" / "slurm").mkdir(parents=True)
    scripts = tmp_path / "video_encoding" / "scripts"
    scripts.mkdir(parents=True)
    job = tmp_path / "video_encoding" / "slurm" / "mesh_train.sbatch"
    shutil.copy2(REPO / "video_encoding/slurm/mesh_train.sbatch", job)
    marker = tmp_path / "worker_ran"
    (scripts / "run_mesh_worker.sh").write_text(f"touch {marker}\n")
    env = {"PATH": "/usr/bin:/bin", "SLURM_SUBMIT_DIR": str(tmp_path), "HOME": str(tmp_path)}
    _stub_gate(scripts, 1)
    r = subprocess.run(["bash", str(job)], env=env, capture_output=True, text=True)
    assert (r.returncode, marker.exists()) == (0, False)
    _stub_gate(scripts, 0)
    r = subprocess.run(["bash", str(job)], env=env, capture_output=True, text=True)
    assert (r.returncode, marker.exists()) == (0, True), r.stderr


def test_shim_passes_job_steps_but_gates_new_jobs(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shim = scripts / "slurm_class_hours_shim.sh"
    shutil.copy2(REPO / "video_encoding/scripts/slurm_class_hours_shim.sh", shim)
    _stub_gate(scripts, 1)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name in ("sbatch", "srun"):
        (bindir / name).symlink_to(shim)
    env = {"PATH": "/usr/bin:/bin"}
    new_job = subprocess.run([str(bindir / "sbatch"), "--version"], env=env, capture_output=True)
    assert new_job.returncode == 1  # refused before /usr/bin/sbatch is reached
    in_job = subprocess.run(
        [str(bindir / "sbatch"), "--version"],
        env={**env, "SLURM_JOB_ID": "7"},
        capture_output=True,
    )
    assert in_job.returncode == 1  # sbatch from inside a job is a new job
    step = subprocess.run(
        ["bash", "-c", f'SLURM_JOB_ID=7 "{bindir / "srun"}" --version; echo "rc=$?"'],
        env=env,
        capture_output=True,
        text=True,
    )
    # A job step skips the gate and execs the real srun: whatever that does on
    # this machine (126/127 if absent), it is not the gate's refusal code.
    assert "rc=1\n" not in step.stdout


# --- nothing can launch around the gate --------------------------------------

_SHELL_LAUNCH = re.compile(r"(^|[\s;|&(`])(sbatch|salloc)\s")


def _code_lines(text: str) -> str:
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def test_every_job_script_is_gated():
    scripts = sorted((REPO / "video_encoding").rglob("*.sbatch"))
    assert scripts
    for path in scripts:
        assert "class_hours.py" in _code_lines(path.read_text()), path


def test_every_launcher_is_gated():
    ungated = []
    for path in (REPO / "video_encoding").rglob("*"):
        if "tests" in path.parts or path.suffix not in (".sh", ".py"):
            continue
        text = path.read_text()
        code = _code_lines(text)
        launches = (
            _SHELL_LAUNCH.search(code)
            if path.suffix == ".sh"
            else re.search(r"""["']sbatch["']""", code)
        )
        if launches and "class_hours" not in code:
            ungated.append(str(path.relative_to(REPO)))
    assert ungated == []


def test_full_check_neither_resubmits_nor_escalates_in_class_hours(tmp_path):
    # The escalation agent may itself sbatch, so it is held too; the dead flag
    # recurs every tick and is handled once launching is allowed. Control: the
    # same dead job at 21:00 is resubmitted.
    from video_encoding.watcher.tick import run_full_check

    record = JobRecord("100", "exp_a", "tagA", "2026-07-15T09:00:00Z", "node001")
    results = tmp_path / "results"
    results.mkdir()
    calls = {}
    for label, when in (("class", CLASS_TIME), ("evening", _at(2026, 9, 21, 21).timestamp())):
        state_dir = tmp_path / label
        state_dir.mkdir()
        (state_dir / "squeue.json").write_text(json.dumps({"jobs": []}))
        from video_encoding.watcher.registry import append_job_record

        append_job_record(state_dir / "jobs_registry.json", record)
        with (
            patch("video_encoding.watcher.tick.collect_disk_usage_pct", return_value={}),
            patch("video_encoding.watcher.tick.remediate_dead_job", return_value=None) as rem,
            patch(
                "video_encoding.watcher.tick.run_escalation", return_value=tmp_path / "t.md"
            ) as esc,
        ):
            run_full_check(
                repo_root=tmp_path,
                state_dir=state_dir,
                results_root=results,
                now=when,
                state=WatcherState(),
            )
        calls[label] = (rem.call_count, esc.call_count)
    assert calls["class"] == (0, 0)
    assert calls["evening"] == (1, 1)


def test_pool_replacement_is_held_not_submitted(tmp_path, closed):
    from video_encoding.watcher.mesh_tick import submit_replacement

    state = WatcherState()
    with patch("video_encoding.watcher.mesh_tick.subprocess.run") as run:
        picked, note = submit_replacement(
            _run_config(), state.run("RUN1"), state, tmp_path, tmp_path, None, "node001"
        )
    run.assert_not_called()
    assert picked == []
    assert "held" in note
