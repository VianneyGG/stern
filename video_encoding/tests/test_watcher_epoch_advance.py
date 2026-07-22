from __future__ import annotations

from video_encoding.watcher.epoch_advance import (
    eligible_nodes,
    exclude_node,
    pending_nodes,
    record_missed_round,
)
from video_encoding.watcher.state import WatcherState, load_state, save_state


def test_pending_nodes_reads_checkpoint_filenames(tmp_path):
    (tmp_path / "RUN1_node001.pt").write_text("x")
    (tmp_path / "RUN1_node002.pt").write_text("x")
    (tmp_path / "RUN2_node089.pt").write_text("x")

    assert pending_nodes(tmp_path, "RUN1") == {"node001", "node002"}


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


def test_a_successful_round_resets_the_miss_counter():
    state = WatcherState()
    record_missed_round(state, "RUN1", "node001")
    state.run("RUN1").missed_rounds["node001"] = 0  # a successful round resets it

    excluded = record_missed_round(state, "RUN1", "node001")
    assert excluded is False
    assert "node001" not in state.excluded_nodes


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
