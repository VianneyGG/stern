from __future__ import annotations

from pathlib import Path

QUEUE_PATH = (
    Path(__file__).resolve().parents[2] / "runs" / "state" / "sweep_queue_cars.txt"
)

EXPECTED = [
    ("sw3-cars-lr060m090", 40, 0.6, 0.9, 8, "token", []),
    ("sw3-cars-lr080m090", 40, 0.8, 0.9, 8, "token", []),
    ("sw3-cars-accum128", 30, 0.7, 0.9, 8, "token", ["train.accum_steps=128"]),
    ("sw3-cars-lrmult15", 30, 0.7, 0.9, 8, "token", ["train.lr_mult=1.5"]),
    ("sw3-cars-wd010", 30, 0.7, 0.9, 8, "token", ["train.weight_decay=0.1"]),
]


def _parsed_lines() -> list[tuple]:
    lines = [ln.split("#", 1)[0].strip() for ln in QUEUE_PATH.read_text().splitlines()]
    lines = [ln for ln in lines if ln]
    parsed = []
    for ln in lines:
        fields = ln.split()
        run_id, rounds, outer_lr, outer_momentum, quorum_k, weighting = fields[:6]
        extra = fields[6:]
        parsed.append(
            (
                run_id,
                int(rounds),
                float(outer_lr),
                float(outer_momentum),
                int(quorum_k),
                weighting,
                extra,
            )
        )
    return parsed


def test_queue_file_exists():
    assert QUEUE_PATH.exists() is True


def test_queue_matches_the_design_doc_cells_in_priority_order():
    assert _parsed_lines() == EXPECTED


def test_nominal_total_hours_at_measured_round_cost():
    # 17 min/round, not the design doc's 16: recomputed from live login-node
    # metrics timestamps (see plan's "Corrected Assumptions"). This total
    # exceeds the 43h window -- cell 5 (weight_decay=0.1) is expected to be
    # cut short or dropped entirely; recompute the live cut point from actual
    # elapsed time once the sweep is running, don't trust this nominal figure
    # as a promise of what will complete.
    minutes_per_round = 17
    total_hours = sum(rounds for _, rounds, *_ in EXPECTED) * minutes_per_round / 60
    assert round(total_hours, 1) == 48.2
