"""Detecting nodes that preemption has turned into silent freeloaders."""

from __future__ import annotations

import json

from stern.watcher.preemption import starving_nodes, trailing_abandons


def _rec(round_num: int, *, abandoned: bool) -> dict:
    return {"round": round_num, "abandoned": abandoned, "node": "node055"}


def test_trailing_abandons_counts_the_unbroken_tail():
    records = [
        _rec(1, abandoned=False),
        _rec(2, abandoned=True),
        _rec(3, abandoned=True),
    ]
    assert trailing_abandons(records) == 2


def test_trailing_abandons_is_zero_once_the_node_recovers():
    """node055 was contended for 23 rounds and then recovered -- a node that is
    contributing again must not stay flagged."""
    records = [
        _rec(1, abandoned=True),
        _rec(2, abandoned=True),
        _rec(3, abandoned=False),
    ]
    assert trailing_abandons(records) == 0


def test_trailing_abandons_uses_round_order_not_file_order():
    """A chunk relaunch appends from a fresh process."""
    records = [
        _rec(3, abandoned=False),
        _rec(1, abandoned=True),
        _rec(2, abandoned=True),
    ]
    assert trailing_abandons(records) == 0


def test_trailing_abandons_empty():
    assert trailing_abandons([]) == 0


def _write(tmp_path, run_id, node, records):
    d = tmp_path / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{node}.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")


def test_starving_nodes_reports_only_those_at_the_streak(tmp_path):
    _write(tmp_path, "RUN1", "node055", [_rec(i, abandoned=True) for i in range(1, 5)])
    _write(
        tmp_path, "RUN1", "node057", [_rec(1, abandoned=True), _rec(2, abandoned=False)]
    )
    _write(
        tmp_path, "RUN1", "node032", [_rec(1, abandoned=True), _rec(2, abandoned=True)]
    )

    assert starving_nodes(tmp_path, "RUN1") == {"node055": 4}


def test_starving_nodes_honours_a_lower_streak(tmp_path):
    _write(
        tmp_path, "RUN1", "node032", [_rec(1, abandoned=True), _rec(2, abandoned=True)]
    )
    assert starving_nodes(tmp_path, "RUN1", streak=2) == {"node032": 2}


def test_starving_nodes_missing_run_dir_is_empty(tmp_path):
    assert starving_nodes(tmp_path, "NOPE") == {}


def test_starving_nodes_survives_a_half_written_line(tmp_path):
    """The worker appends while the watcher reads -- a torn last line must
    not crash the tick."""
    d = tmp_path / "RUN1"
    d.mkdir(parents=True)
    good = "\n".join(json.dumps(_rec(i, abandoned=True)) for i in range(1, 5))
    (d / "node055.jsonl").write_text(good + '\n{"round": 5, "abando')
    assert starving_nodes(tmp_path, "RUN1") == {"node055": 4}


def test_starving_nodes_ignores_records_without_the_field(tmp_path):
    """Runs launched before preemption existed have no `abandoned` key."""
    _write(tmp_path, "RUN1", "node055", [{"round": i} for i in range(1, 6)])
    assert starving_nodes(tmp_path, "RUN1") == {}
