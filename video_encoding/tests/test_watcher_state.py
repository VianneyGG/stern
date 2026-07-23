from __future__ import annotations

from video_encoding.watcher.state import (
    FAST_INTERVAL_S,
    SLOW_INTERVAL_S,
    WatcherState,
    advance_backoff,
    load_state,
    save_state,
)


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


def test_run_target_pool_size_defaults_to_zero():
    state = WatcherState()
    assert state.run("RUN1").target_pool_size == 0


def test_run_target_pool_size_roundtrips(tmp_path):
    path = tmp_path / "watcher_state.json"
    state = WatcherState()
    state.run("RUN1").target_pool_size = 16

    save_state(path, state)
    loaded = load_state(path)

    assert loaded.run("RUN1").target_pool_size == 16
