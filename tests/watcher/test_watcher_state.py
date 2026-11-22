from __future__ import annotations

import json
from dataclasses import dataclass

from stern.watcher.registry import JobRecord, append_job_record
from stern.watcher.state import (
    FAST_INTERVAL_S,
    SLOW_INTERVAL_S,
    WatcherState,
    advance_backoff,
    clear_run_state,
    load_state,
    save_state,
    write_status_summary,
)


@dataclass
class _FakeRunConfig:
    run_id: str
    rounds: int


def _iso(epoch_s: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(epoch_s, tz=timezone.utc).isoformat()


def test_load_state_missing_file_bootstraps_immediate_full_check(tmp_path):
    state = load_state(tmp_path / "missing.json")
    assert state.next_full_check_at == 0.0
    assert state.current_interval_s == FAST_INTERVAL_S
    assert state.jobs == {}


def test_job_creates_default_entry_on_first_access():
    state = WatcherState()
    js = state.job("36901")
    assert js.slurm_job_id == "36901"
    assert js.status == "healthy"
    assert state.jobs["36901"] is js  # same object on repeat access


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "watcher_state.json"
    state = WatcherState(next_full_check_at=1000.0, current_interval_s=120.0)
    state.job("36901").auto_resubmit_count = 1
    state.job("36901").status = "escalated"
    state.escalation_day = "2026-07-15"
    state.escalation_count_today = 3

    save_state(path, state)
    loaded = load_state(path)

    assert loaded.next_full_check_at == 1000.0
    assert loaded.current_interval_s == 120.0
    assert loaded.jobs["36901"].auto_resubmit_count == 1
    assert loaded.jobs["36901"].status == "escalated"
    assert loaded.escalation_day == "2026-07-15"
    assert loaded.escalation_count_today == 3


def test_nan_escalations_in_flight_round_trips(tmp_path):
    path = tmp_path / "watcher_state.json"
    state = WatcherState()
    state.nan_escalations_in_flight["run1"] = 1234.0

    save_state(path, state)
    loaded = load_state(path)

    assert loaded.nan_escalations_in_flight == {"run1": 1234.0}


def test_watcher_state_roundtrips_last_tick_at(tmp_path):
    path = tmp_path / "watcher_state.json"
    state = WatcherState()
    state.last_tick_at = 12345.0
    save_state(path, state)
    loaded = load_state(path)
    assert loaded.last_tick_at == 12345.0


def test_load_state_defaults_last_tick_at_for_a_pre_existing_file(tmp_path):
    path = tmp_path / "watcher_state.json"
    path.write_text('{"jobs": {}, "runs": {}, "excluded_nodes": {}}')
    loaded = load_state(path)
    assert loaded.last_tick_at == 0.0


def test_advance_backoff_grows_toward_ceiling_when_no_action():
    state = WatcherState(current_interval_s=1000.0)
    advance_backoff(state, now=5000.0, action_taken=False)
    assert state.current_interval_s == min(1000.0 * 1.5, SLOW_INTERVAL_S)
    assert state.next_full_check_at == 5000.0 + state.current_interval_s


def test_advance_backoff_caps_at_slow_ceiling():
    state = WatcherState(current_interval_s=SLOW_INTERVAL_S)
    advance_backoff(state, now=0.0, action_taken=False)
    assert state.current_interval_s == SLOW_INTERVAL_S


def test_advance_backoff_resets_to_fast_floor_on_action():
    state = WatcherState(current_interval_s=SLOW_INTERVAL_S)
    advance_backoff(state, now=100.0, action_taken=True)
    assert state.current_interval_s == FAST_INTERVAL_S
    assert state.next_full_check_at == 100.0 + FAST_INTERVAL_S


def test_run_target_pool_size_roundtrips(tmp_path):
    path = tmp_path / "watcher_state.json"
    state = WatcherState()
    state.run("RUN1").target_pool_size = 16

    save_state(path, state)
    loaded = load_state(path)

    assert loaded.run("RUN1").target_pool_size == 16


def test_clear_run_state_removes_only_this_runs_crash_loop_exclusions():
    """0.5: 'wipe the dead list at job end' must not re-admit a node
    excluded for a different run or flagged by hand -- excluded_nodes is
    documented as never auto-reversed except by a human, so clearing must
    be scoped to exclusions this run itself caused."""
    state = WatcherState()
    state.excluded_nodes["node001"] = "crash-loop: missed 2 consecutive rounds of RUN1"
    state.excluded_nodes["node002"] = "crash-loop: missed 3 consecutive rounds of RUN2"
    state.excluded_nodes["node003"] = "manually flagged: broken GPU"

    clear_run_state(state, "RUN1")

    assert "node001" not in state.excluded_nodes
    assert (
        state.excluded_nodes["node002"]
        == "crash-loop: missed 3 consecutive rounds of RUN2"
    )
    assert state.excluded_nodes["node003"] == "manually flagged: broken GPU"


def test_clear_run_state_leaves_state_runs_untouched():
    """state.runs must survive DONE -- it is the ONLY source watch_training.py
    has for exempting a finished run's worker jobs from remediate_dead_job
    (mesh_run_ids = configured runs | state.runs); popping it here would
    reopen the exact bug fixed by the _is_mesh_job filter."""
    state = WatcherState()
    state.run("RUN1").status = "DONE"

    clear_run_state(state, "RUN1")

    assert "RUN1" in state.runs
    assert state.runs["RUN1"].status == "DONE"


def test_clear_run_state_is_a_noop_when_nothing_matches():
    state = WatcherState()
    state.excluded_nodes["node001"] = "manually flagged: broken GPU"

    clear_run_state(state, "RUN1")

    assert state.excluded_nodes == {"node001": "manually flagged: broken GPU"}


def test_write_status_summary_omits_per_node_job_ids(tmp_path):
    """Per-node identity (job id -> node -> health) must not appear at all --
    that list, at 16-32 workers/run across dozens of tracked runs, was most
    of mesh_status.json's size."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    state = WatcherState()
    state.run("RUN1").status = "RUNNING"
    state.run("RUN1").round = 3

    append_job_record(
        state_dir / "jobs_registry.json",
        JobRecord(
            slurm_job_id="100",
            experiment="exp",
            run_tag="RUN1",
            submitted_at="2026-08-31T00:00:00+00:00",
            nodelist="node014",
        ),
    )

    out_path = state_dir.parent / "status.json"
    write_status_summary(
        out_path, state, [_FakeRunConfig(run_id="RUN1", rounds=10)], state_dir=state_dir
    )
    payload = json.loads(out_path.read_text())

    entry = payload["runs"]["RUN1"]
    assert "job_ids" not in entry
    assert "chunk_timers" not in entry
    assert "100" not in json.dumps(entry)
    assert "node014" not in json.dumps(entry)


def test_write_status_summary_past_chunk_durations_cluster_by_pool_wide_wave(tmp_path):
    """A chunk relaunch submits every node in the pool within one tick, so
    those submissions must collapse into a single wave -- one duration per
    chunk, not one entry per node's slurm job id."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    state = WatcherState()
    state.run("RUN1").status = "RUNNING"

    registry_path = state_dir / "jobs_registry.json"
    wave_1_start = 1_000_000.0
    wave_2_start = wave_1_start + 3600.0  # well past the 300s wave-gap window
    # Two nodes per wave, submitted seconds apart -- must still cluster together.
    for job_id, ts in [
        ("1", wave_1_start),
        ("2", wave_1_start + 5.0),
        ("3", wave_2_start),
        ("4", wave_2_start + 5.0),
    ]:
        append_job_record(
            registry_path,
            JobRecord(
                slurm_job_id=job_id,
                experiment="exp",
                run_tag="RUN1",
                submitted_at=_iso(ts),
                nodelist=f"node{job_id}",
            ),
        )

    out_path = state_dir.parent / "status.json"
    write_status_summary(
        out_path, state, [_FakeRunConfig(run_id="RUN1", rounds=10)], state_dir=state_dir
    )
    entry = json.loads(out_path.read_text())["runs"]["RUN1"]

    assert entry["past_chunk_durations_h"] == [1.0]


def test_write_status_summary_round_pace_and_eta_from_rounds_jsonl(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    results_dir = tmp_path / "results" / "mesh"
    results_dir.mkdir(parents=True)

    base_ts = 1_000_000.0
    round_len = 3600.0  # 1h/round
    with (results_dir / "RUN1_rounds.jsonl").open("w") as fh:
        for i, ts in enumerate([base_ts, base_ts + round_len, base_ts + 2 * round_len]):
            fh.write(json.dumps({"round": i + 1, "ts": ts}) + "\n")

    state = WatcherState()
    state.run("RUN1").status = "RUNNING"
    state.run("RUN1").round = 3

    out_path = state_dir.parent / "status.json"
    write_status_summary(
        out_path, state, [_FakeRunConfig(run_id="RUN1", rounds=10)], state_dir=state_dir
    )
    entry = json.loads(out_path.read_text())["runs"]["RUN1"]

    assert entry["avg_round_duration_h"] == 1.0
    assert entry["recent_round_durations_h"] == [1.0, 1.0]
    # 10 total rounds, 3 done -> 7 remaining at 1h each.
    assert entry["eta_h"] == 7.0
    assert entry["eta_ts"] is not None


def test_write_status_summary_eta_is_none_with_fewer_than_two_rounds(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    state = WatcherState()
    state.run("RUN1").status = "RUNNING"

    out_path = state_dir.parent / "status.json"
    write_status_summary(
        out_path, state, [_FakeRunConfig(run_id="RUN1", rounds=10)], state_dir=state_dir
    )
    entry = json.loads(out_path.read_text())["runs"]["RUN1"]

    assert entry["avg_round_duration_h"] is None
    assert entry["eta_h"] is None
    assert entry["eta_ts"] is None
    assert entry["past_chunk_durations_h"] == []
