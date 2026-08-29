from __future__ import annotations

from video_encoding.watcher.epoch_advance import (
    candidate_replacements,
    clear_missed_rounds,
    eligible_nodes,
    exclude_node,
    pending_nodes,
    pool_deficit,
    readmit_healthy_nodes,
    record_missed_round,
    select_idle_nodes,
)
from video_encoding.watcher.state import WatcherState, load_state, save_state


def test_pending_nodes_reads_checkpoint_filenames(tmp_path):
    (tmp_path / "RUN1_0_node001.pt").write_text("x")
    (tmp_path / "RUN1_0_node002.pt").write_text("x")
    (tmp_path / "RUN2_0_node089.pt").write_text("x")

    assert pending_nodes(tmp_path, "RUN1") == {"node001", "node002"}


def test_pending_nodes_scopes_to_round_num_when_given(tmp_path):
    (tmp_path / "RUN1_0_node001.pt").write_text("x")
    (tmp_path / "RUN1_1_node002.pt").write_text("x")

    assert pending_nodes(tmp_path, "RUN1", round_num=1) == {"node002"}


def test_one_missed_round_does_not_exclude():
    state = WatcherState()
    excluded = record_missed_round(state, "RUN1", "node001")
    assert excluded is False
    assert "node001" not in state.excluded_nodes


def test_two_consecutive_missed_rounds_excludes():
    state = WatcherState()
    record_missed_round(state, "RUN1", "node001")
    excluded = record_missed_round(state, "RUN1", "node001")
    assert excluded is True
    assert "node001" in state.excluded_nodes


def test_miss_then_checkin_then_miss_does_not_exclude():
    """The bug this fixes: nothing today resets missed_rounds on a
    successful check-in, so any node that hiccups twice ever (not
    consecutively) is permanently excluded."""
    state = WatcherState()
    record_missed_round(state, "RUN1", "node001")  # miss 1
    clear_missed_rounds(state.run("RUN1"), {"node001"})  # checks in
    excluded = record_missed_round(state, "RUN1", "node001")  # miss 2 (non-consecutive)
    assert excluded is False
    assert "node001" not in state.excluded_nodes


def test_clear_missed_rounds_only_touches_checked_in_nodes():
    state = WatcherState()
    run = state.run("RUN1")
    run.missed_rounds["node001"] = 1
    run.missed_rounds["node002"] = 1
    clear_missed_rounds(run, {"node001"})
    assert run.missed_rounds["node001"] == 0
    assert run.missed_rounds["node002"] == 1


def test_excluded_nodes_are_never_eligible():
    state = WatcherState()
    exclude_node(state, "node024", "quota: /Data at 97%")
    assert eligible_nodes(state, ["node001", "node024"]) == ["node001"]


def test_exclusion_records_the_reason_for_a_human():
    state = WatcherState()
    exclude_node(state, "node024", "quota: /Data at 97%")
    assert state.excluded_nodes["node024"] == "quota: /Data at 97%"


def test_watcher_state_roundtrips_runs_and_exclusions(tmp_path):
    path = tmp_path / "watcher_state.json"
    state = WatcherState()
    state.run("RUN1").round = 3
    exclude_node(state, "node024", "quota: /Data at 97%")

    save_state(path, state)
    loaded = load_state(path)

    assert loaded.excluded_nodes == {"node024": "quota: /Data at 97%"}
    assert loaded.run("RUN1").round == 3


def test_pool_deficit_zero_when_full():
    state = WatcherState()
    run = state.run("RUN1")
    run.target_pool_size = 3
    assert pool_deficit(run, ["a", "b", "c"], state) == 0


def test_pool_deficit_counts_excluded_nodes():
    state = WatcherState()
    run = state.run("RUN1")
    run.target_pool_size = 3
    exclude_node(state, "b", "crash-loop")
    assert pool_deficit(run, ["a", "b", "c"], state) == 1


def test_pool_deficit_never_negative():
    state = WatcherState()
    run = state.run("RUN1")
    run.target_pool_size = 2
    assert pool_deficit(run, ["a", "b", "c"], state) == 0


def test_candidate_replacements_excludes_in_use_and_excluded():
    tier_roster = ["a", "b", "c", "d"]
    result = candidate_replacements(tier_roster, in_use={"a", "b"}, excluded={"c"})
    assert result == ["d"]


def test_candidate_replacements_preserves_roster_order():
    tier_roster = ["a", "b", "c", "d"]
    result = candidate_replacements(tier_roster, in_use=set(), excluded=set())
    assert result == ["a", "b", "c", "d"]


def test_select_idle_nodes_returns_first_n_idle_in_candidate_order():
    assert select_idle_nodes(["a", "b", "c"], 1, idle={"b", "c"}) == ["b"]


def test_select_idle_nodes_partial_fill_when_not_enough_idle():
    assert select_idle_nodes(["a", "b", "c"], 3, idle={"c"}) == ["c"]


def test_select_idle_nodes_empty_when_none_idle():
    assert select_idle_nodes(["a", "b"], 2, idle=set()) == []


# --- transient exclusion / re-admission ---------------------------------


def _excluded_state(run_id: str = "RUN1") -> WatcherState:
    state = WatcherState()
    run = state.run(run_id)
    run.missed_rounds["node040"] = 2
    exclude_node(
        state, "node040", f"crash-loop: missed 2 consecutive rounds of {run_id}"
    )
    return state


def test_readmit_returns_a_recovered_node_and_clears_its_exclusion():
    """A node excluded for missing rounds was usually not broken: on
    2026-08-21 every one was up, reachable and still holding its shards --
    a foreign process held its GPU. Once that clears it must come back."""
    state = _excluded_state()
    assert readmit_healthy_nodes(state, "RUN1", {"node040"}) == ["node040"]
    assert state.excluded_nodes == {}
    assert state.run("RUN1").missed_rounds["node040"] == 0


def test_readmit_ignores_a_node_that_is_still_unhealthy():
    state = _excluded_state()
    assert readmit_healthy_nodes(state, "RUN1", set()) == []
    assert "node040" in state.excluded_nodes


def test_readmit_never_touches_an_exclusion_from_another_run():
    state = _excluded_state(run_id="RUN2")
    assert readmit_healthy_nodes(state, "RUN1", {"node040"}) == []
    assert "node040" in state.excluded_nodes


def test_readmit_never_touches_a_manual_exclusion():
    """exclude_node's contract: a human-set reason is never auto-reversed."""
    state = WatcherState()
    state.run("RUN1")
    exclude_node(state, "node069", "hardware: no shards seeded, do not use")
    assert readmit_healthy_nodes(state, "RUN1", {"node069"}) == []
    assert "node069" in state.excluded_nodes


def test_readmit_is_capped_so_a_genuine_crash_loop_cannot_flap():
    """A node that looks healthy but keeps dying takes the pool down with it
    if it is re-admitted every round. One second chance, mirroring tick.py's
    resubmit-once-then-escalate tolerance."""
    state = _excluded_state()
    assert readmit_healthy_nodes(state, "RUN1", {"node040"}) == ["node040"]

    state.run("RUN1").missed_rounds["node040"] = 2
    exclude_node(state, "node040", "crash-loop: missed 2 consecutive rounds of RUN1")
    assert readmit_healthy_nodes(state, "RUN1", {"node040"}) == []
    assert "node040" in state.excluded_nodes


def test_readmit_cap_is_configurable():
    state = _excluded_state()
    readmit_healthy_nodes(state, "RUN1", {"node040"})
    state.run("RUN1").missed_rounds["node040"] = 2
    exclude_node(state, "node040", "crash-loop: missed 2 consecutive rounds of RUN1")
    assert readmit_healthy_nodes(state, "RUN1", {"node040"}, max_readmissions=2) == [
        "node040"
    ]


def test_readmit_returns_several_nodes_in_a_stable_order():
    state = WatcherState()
    run = state.run("RUN1")
    for node in ("node046", "node040", "node045"):
        run.missed_rounds[node] = 2
        exclude_node(state, node, "crash-loop: missed 2 consecutive rounds of RUN1")
    recovered = readmit_healthy_nodes(state, "RUN1", {"node046", "node040", "node045"})
    assert recovered == ["node040", "node045", "node046"]
