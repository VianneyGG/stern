"""Tests for `video_encoding/scripts/plot_node_scaling.py`.

Fixtures are hand-written JSONL in the real on-disk schema -- worker records
as `mesh_round._append_round_metrics` writes them, merge records as
`run_outer_step._append_merge_metrics` writes them -- so the reduction is
exercised against the format it will actually meet, not a mock.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from video_encoding.scripts import plot_node_scaling as pns

pytestmark = pytest.mark.system


def _write_worker(metrics_dir: Path, node: str, records: list[dict]) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / f"{node}.jsonl", "w") as f:
        for r in records:
            base = {
                "run_id": "cell",
                "node": node,
                "train_loss": 1.0,
                "val_loss": 0.5,
                "samp_s": 10.0,
                "n_shards": 3,
                "lr": 0.001,
                "n_samples": 100,
                "n_steps": 5,
                "ts": 0.0,
                "slurm_job_id": None,
            }
            base.update(r)
            f.write(json.dumps(base) + "\n")


def _write_merge(path: Path, per_round_contributors: dict[int, int]) -> None:
    """Merge records are 0-BASED on round; worker records are 1-based."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rnd, n in sorted(per_round_contributors.items()):
            f.write(
                json.dumps(
                    {
                        "run_id": "cell",
                        "round": rnd,
                        "n_contributors": n,
                        "contributors": [],
                        "alphas": {},
                        "timeout_forced": False,
                        "ts": 0.0,
                    }
                )
                + "\n"
            )


def test_load_cell_means_val_loss_across_nodes(tmp_path):
    d = tmp_path / "cell"
    _write_worker(d, "a", [{"round": 1, "val_loss": 0.4}])
    _write_worker(d, "b", [{"round": 1, "val_loss": 0.6}])

    points = pns.load_cell(d, expected_nodes=2)
    assert [p.val_loss for p in points] == [0.5]


def test_load_cell_accumulates_samples_across_nodes_and_rounds(tmp_path):
    d = tmp_path / "cell"
    for node in ("a", "b"):
        _write_worker(
            d,
            node,
            [{"round": 1, "n_samples": 100}, {"round": 2, "n_samples": 100}],
        )

    points = pns.load_cell(d, expected_nodes=2)
    assert [p.cum_samples for p in points] == [200, 400]


def test_wall_clock_is_relative_and_uses_the_median_ts(tmp_path):
    """One straggler must not stretch the time axis: rounds are
    time-budgeted, so a late node is a late node, not a longer round."""
    d = tmp_path / "cell"
    _write_worker(d, "a", [{"round": 1, "ts": 1000.0}, {"round": 2, "ts": 1100.0}])
    _write_worker(d, "b", [{"round": 1, "ts": 1000.0}, {"round": 2, "ts": 1110.0}])
    _write_worker(d, "c", [{"round": 1, "ts": 1000.0}, {"round": 2, "ts": 9999.0}])

    points = pns.load_cell(d, expected_nodes=3)
    assert points[0].wall_clock_s == 0.0
    assert points[1].wall_clock_s == 110.0  # median 1110.0, not the max


def test_load_cell_aligns_zero_based_merge_rounds_with_one_based_worker_rounds(
    tmp_path,
):
    """mesh_round writes `round_num + 1`; run_outer_step writes `run.round`
    unshifted. Reading them as the same index shifts every contributor
    count by one and would silently drop the wrong rounds."""
    d = tmp_path / "cell"
    _write_worker(d, "a", [{"round": 1}, {"round": 2}])
    merge = tmp_path / "cell_rounds.jsonl"
    _write_merge(merge, {0: 4, 1: 3})

    points = pns.load_cell(d, expected_nodes=4, merge_metrics=merge)
    assert [(p.round, p.n_contributors) for p in points] == [(1, 4), (2, 3)]


def test_load_cell_treats_a_missing_merge_file_as_full_strength(tmp_path):
    d = tmp_path / "cell"
    _write_worker(d, "a", [{"round": 1}])

    points = pns.load_cell(d, expected_nodes=4, merge_metrics=tmp_path / "absent.jsonl")
    assert points[0].n_contributors == 4


def test_drop_degraded_removes_short_rounds(tmp_path):
    points = [
        pns.RoundPoint(1, 0.5, 100, 0.0, 4),
        pns.RoundPoint(2, 0.4, 200, 100.0, 3),
        pns.RoundPoint(3, 0.3, 300, 200.0, 4),
    ]
    kept = pns.drop_degraded(points, expected_nodes=4)
    assert [p.round for p in kept] == [1, 3]


def test_time_to_loss_interpolates_between_rounds(tmp_path):
    """12-22 rounds is too coarse for a nearest-round answer."""
    points = [
        pns.RoundPoint(1, 0.6, 100, 100.0, 1),
        pns.RoundPoint(2, 0.4, 200, 200.0, 1),
    ]
    assert pns.time_to_loss(points, 0.5) == 150.0


def test_time_to_loss_returns_none_when_target_never_reached(tmp_path):
    points = [
        pns.RoundPoint(1, 0.6, 100, 100.0, 1),
        pns.RoundPoint(2, 0.55, 200, 200.0, 1),
    ]
    assert pns.time_to_loss(points, 0.3) is None


def test_efficiency_is_speedup_over_n():
    assert pns.efficiency(4, 400.0, 200.0) == 0.5


def test_deadline_truncates_the_series():
    points = [
        pns.RoundPoint(1, 0.6, 100, 0.0, 1),
        pns.RoundPoint(2, 0.5, 200, 3600.0, 1),
        pns.RoundPoint(3, 0.4, 300, 21600.1, 1),
    ]
    kept = pns.truncate_at(points, 6.0 * 3600)
    assert [p.round for p in kept] == [1, 2]


def test_main_writes_csv_and_pngs(tmp_path):
    metrics_root = tmp_path / "metrics"
    merge_root = tmp_path / "merge"
    for run_id, n_nodes in (("nsl-n01a", 1), ("nsl-n02", 2)):
        for i in range(n_nodes):
            _write_worker(
                metrics_root / run_id,
                f"node{i}",
                [
                    {"round": 1, "val_loss": 0.8, "ts": 0.0, "n_samples": 100},
                    {
                        "round": 2,
                        "val_loss": 0.4 / n_nodes,
                        "ts": 100.0,
                        "n_samples": 100,
                    },
                ],
            )
        _write_merge(merge_root / f"{run_id}_rounds.jsonl", {0: n_nodes, 1: n_nodes})

    out_csv = tmp_path / "out.csv"
    out_png = tmp_path / "curve.png"
    rc = subprocess.run(
        [
            sys.executable,
            "-m",
            "video_encoding.scripts.plot_node_scaling",
            "--metrics-root",
            str(metrics_root),
            "--merge-metrics-root",
            str(merge_root),
            "--target-loss",
            "0.5",
            "--out-csv",
            str(out_csv),
            "--out-png",
            str(out_png),
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )
    assert rc.returncode == 0, rc.stderr

    rows = out_csv.read_text().splitlines()
    assert len(rows) == 1 + 2  # header + one row per cell found
    for png in (
        out_png.with_name("curve_wallclock.png"),
        out_png.with_name("curve_samples.png"),
    ):
        assert png.stat().st_size > 0


def test_main_warns_when_the_target_is_above_a_cells_first_round(tmp_path):
    """t_loss == 0.0 means the cell started below the target, so every
    speedup against it is meaningless -- that must not read as plain 'n/a'."""
    metrics_root = tmp_path / "metrics"
    _write_worker(
        metrics_root / "nsl-n01a",
        "node0",
        [
            {"round": 1, "val_loss": 0.2, "ts": 0.0},
            {"round": 2, "val_loss": 0.1, "ts": 100.0},
        ],
    )

    rc = subprocess.run(
        [
            sys.executable,
            "-m",
            "video_encoding.scripts.plot_node_scaling",
            "--metrics-root",
            str(metrics_root),
            "--merge-metrics-root",
            str(tmp_path / "absent"),
            "--target-loss",
            "0.5",
            "--out-csv",
            str(tmp_path / "out.csv"),
            "--out-png",
            str(tmp_path / "c.png"),
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )
    assert rc.returncode == 0, rc.stderr
    assert "nsl-n01a" in rc.stderr
    assert "pick a lower target" in rc.stderr
