from __future__ import annotations

from video_encoding.mesh.pending_files import (
    parse_pending_filename,
    pending_filename,
    pending_files_for_round,
)


def test_pending_filename_round_trips_through_parse():
    name = pending_filename("pool-cars-phase3-smoke", 5, "node062")
    assert parse_pending_filename(
        name.removesuffix(".pt"), "pool-cars-phase3-smoke"
    ) == (
        5,
        "node062",
    )


def test_parse_rejects_a_different_run_ids_stem():
    assert parse_pending_filename("RUN2_5_node089", "RUN1") is None


def test_parse_rejects_the_old_two_part_format():
    """The pre-round-tagging filename format ({run_id}_{node}.pt) has no
    round segment -- must not be silently misparsed as if the node name
    were a round number, or vice versa."""
    assert parse_pending_filename("RUN1_node001", "RUN1") is None


def test_parse_hyphenated_run_id_does_not_swallow_the_round_segment():
    """Mirrors the existing hyphen-suffixed-sibling concern documented in
    run_outer_step.py's _load_worker_checkpoints -- 'pool-fish' must not
    match 'pool-fish-5r_...'. The `_` literal between run_id and the round
    segment is what keeps that scoping exact."""
    assert parse_pending_filename("pool-fish-5r_3_node003", "pool-fish") is None
    assert parse_pending_filename("pool-fish_3_node003", "pool-fish") == (
        3,
        "node003",
    )


def test_parse_rejects_a_non_numeric_round_segment():
    assert parse_pending_filename("RUN1_notaround_node001", "RUN1") is None


def test_pending_files_for_round_filters_out_other_rounds(tmp_path):
    (tmp_path / "RUN1_4_node001.pt").write_text("x")  # stale straggler
    (tmp_path / "RUN1_5_node001.pt").write_text("x")
    (tmp_path / "RUN1_5_node002.pt").write_text("x")
    (tmp_path / "RUN2_5_node089.pt").write_text("x")  # different run_id

    files = pending_files_for_round(tmp_path, "RUN1", 5)
    assert {f.name for f in files} == {"RUN1_5_node001.pt", "RUN1_5_node002.pt"}
