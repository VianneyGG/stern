#!/usr/bin/env python3
"""Reduce a node-count scaling sweep to a speedup/efficiency table and curves.

Reads two roots that are NOT the same directory:

* worker per-round metrics, `<metrics-root>/<run_id>/<node>.jsonl`, written by
  `training/diloco/mesh_round._append_round_metrics` on the mesh share;
* merge per-round metrics, `<merge-metrics-root>/<run_id>_rounds.jsonl`,
  written by `scripts/run_outer_step` under the repo checkout.

A single root flag would silently find no merge records and score every
degraded round as full-strength.

The pure layer below is stdlib-only, like `aggregate_sweep.py`, so it runs on
a bare checkout; matplotlib is imported inside `main` only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

# Fixed by the sweep spec: run_id -> (node count, baseline run_id). Each cell
# is compared against the single-node cell from ITS OWN wave, because wave A's
# four cells share the fish subnet and wave B's do not.
CELLS: dict[str, tuple[int, str | None]] = {
    "nsl-n01a": (1, None),
    "nsl-n02": (2, "nsl-n01a"),
    "nsl-n04": (4, "nsl-n01a"),
    "nsl-n08": (8, "nsl-n01a"),
    "nsl-n16": (16, "nsl-n01b"),
    "nsl-n01b": (1, None),
}


@dataclass(frozen=True)
class RoundPoint:
    round: int  # 1-based, as the worker records number them
    val_loss: float  # mean across the cell's nodes
    cum_samples: int  # cumulative over rounds of the cell's node total
    wall_clock_s: float  # median ts of the round, minus the cell's first
    n_contributors: int


def load_cell(
    metrics_dir: Path, *, expected_nodes: int, merge_metrics: Path | None = None
) -> list[RoundPoint]:
    """Aggregate one cell's per-node JSONL into a per-round series."""
    by_round: dict[int, list[dict]] = {}
    for path in sorted(metrics_dir.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                by_round.setdefault(rec["round"], []).append(rec)

    contributors = _load_contributors(merge_metrics)

    points: list[RoundPoint] = []
    cum = 0
    first_ts: float | None = None
    for rnd in sorted(by_round):
        recs = by_round[rnd]
        cum += sum(int(r["n_samples"]) for r in recs)
        # Median, not max: rounds are time-budgeted, so one late node is a
        # late node rather than a longer round.
        ts = statistics.median(float(r["ts"]) for r in recs)
        if first_ts is None:
            first_ts = ts
        losses = [float(r["val_loss"]) for r in recs]
        finite = [v for v in losses if math.isfinite(v)]
        points.append(
            RoundPoint(
                round=rnd,
                val_loss=sum(finite) / len(finite) if finite else float("nan"),
                cum_samples=cum,
                wall_clock_s=ts - first_ts,
                n_contributors=contributors.get(rnd, expected_nodes),
            )
        )
    return points


def _load_contributors(merge_metrics: Path | None) -> dict[int, int]:
    """Merge-side contributor counts, keyed by the WORKER round number.

    run_outer_step writes `run.round` unshifted while mesh_round writes
    `round_num + 1`, so merge round 0 is worker round 1. Reading both as the
    same index shifts every count by one and drops the wrong rounds.

    A missing file means every round is treated as full strength -- losing
    the degraded-round filter is a smaller error than discarding the cell.
    """
    if merge_metrics is None or not merge_metrics.exists():
        return {}
    counts: dict[int, int] = {}
    for line in merge_metrics.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            counts[int(rec["round"]) + 1] = int(rec["n_contributors"])
    return counts


def drop_degraded(
    points: list[RoundPoint], *, expected_nodes: int
) -> list[RoundPoint]:
    """Drop rounds the cell ran below full strength: with no spare nodes a
    dead worker permanently shrinks a cell, and those rounds do not measure
    the node count the cell is labelled with."""
    return [p for p in points if p.n_contributors >= expected_nodes]


def truncate_at(points: list[RoundPoint], deadline_s: float) -> list[RoundPoint]:
    return [p for p in points if p.wall_clock_s <= deadline_s]


def time_to_loss(points: list[RoundPoint], target: float) -> float | None:
    """Wall-clock at which the series first reaches `target`, linearly
    interpolated between the bracketing rounds; None if never reached.

    12-22 rounds is far too coarse for a nearest-round answer.
    """
    prev: RoundPoint | None = None
    for p in points:
        if math.isnan(p.val_loss):
            continue
        if p.val_loss <= target:
            if prev is None or prev.val_loss == p.val_loss:
                return p.wall_clock_s
            frac = (prev.val_loss - target) / (prev.val_loss - p.val_loss)
            return prev.wall_clock_s + frac * (p.wall_clock_s - prev.wall_clock_s)
        prev = p
    return None


def efficiency(n: int, t_baseline: float, t_cell: float) -> float:
    """Speedup over node count: 1.0 is linear scaling."""
    return (t_baseline / t_cell) / n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-root", type=Path, default=Path.home() / "stern_mesh" / "metrics")
    parser.add_argument("--merge-metrics-root", type=Path, default=Path("runs/results/mesh"))
    parser.add_argument("--deadline-h", type=float, default=6.0)
    parser.add_argument(
        "--target-loss",
        type=float,
        required=True,
        help="loss level speedup is measured at; pick one every cell reached",
    )
    parser.add_argument("--out-csv", type=Path, default=Path("node_scaling.csv"))
    parser.add_argument("--out-png", type=Path, default=Path("node_scaling.png"))
    args = parser.parse_args()

    deadline_s = args.deadline_h * 3600.0

    series: dict[str, list[RoundPoint]] = {}
    dropped: dict[str, int] = {}
    for run_id, (n_nodes, _) in CELLS.items():
        cell_dir = args.metrics_root / run_id
        if not cell_dir.is_dir():
            continue
        points = load_cell(
            cell_dir,
            expected_nodes=n_nodes,
            merge_metrics=args.merge_metrics_root / f"{run_id}_rounds.jsonl",
        )
        points = truncate_at(points, deadline_s)
        kept = drop_degraded(points, expected_nodes=n_nodes)
        dropped[run_id] = len(points) - len(kept)
        series[run_id] = kept

    if not series:
        raise SystemExit(f"plot_node_scaling: no cell metrics under {args.metrics_root}")

    ttl = {rid: time_to_loss(pts, args.target_loss) for rid, pts in series.items()}

    # A cell already at the target in its first round gives t=0: the target
    # is above where that cell started, so every speedup computed against it
    # is meaningless. Say so rather than letting it fall through to "n/a".
    too_high = [rid for rid, t in ttl.items() if t == 0.0]
    if too_high:
        print(
            f"plot_node_scaling: WARNING target loss {args.target_loss} is at or "
            f"above the first recorded round of {', '.join(sorted(too_high))} -- "
            "pick a lower target that every cell actually trained down to",
            file=sys.stderr,
        )

    rows = []
    for run_id, points in series.items():
        n_nodes, baseline_id = CELLS[run_id]
        t_cell = ttl[run_id]
        t_base = ttl.get(baseline_id) if baseline_id else None
        speedup = t_base / t_cell if (t_base and t_cell) else None
        rows.append(
            {
                "run_id": run_id,
                "n": n_nodes,
                "rounds_kept": len(points),
                "rounds_dropped": dropped[run_id],
                "final_val_loss": points[-1].val_loss if points else "",
                "time_to_loss_s": t_cell if t_cell is not None else "",
                "speedup": speedup if speedup is not None else "",
                "efficiency": efficiency(n_nodes, t_base, t_cell) if speedup else "",
            }
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    header = f"{'run_id':<10} {'n':>3} {'kept':>5} {'drop':>5} {'final':>8} {'t_loss':>10} {'speedup':>8} {'eff':>6}"
    print(header)
    print("-" * len(header))
    for r in rows:
        def _fmt(key, spec):
            return format(r[key], spec) if r[key] != "" else "n/a"

        print(
            f"{r['run_id']:<10} {r['n']:>3} {r['rounds_kept']:>5} {r['rounds_dropped']:>5} "
            f"{_fmt('final_val_loss', '8.4f'):>8} {_fmt('time_to_loss_s', '10.1f'):>10} "
            f"{_fmt('speedup', '8.2f'):>8} {_fmt('efficiency', '6.2f'):>6}"
        )

    # The two single-node cells differ only in which wave they ran in, so
    # their gap IS the wave-contention effect -- never fold it into a cell's
    # speedup, it belongs on its own line.
    if ttl.get("nsl-n01a") and ttl.get("nsl-n01b"):
        print(
            f"\ncontention effect (nsl-n01a vs nsl-n01b, same cell, different wave): "
            f"{ttl['nsl-n01a'] / ttl['nsl-n01b']:.3f}x"
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for suffix, xs, xlabel in (
        ("wallclock", lambda pts: [p.wall_clock_s for p in pts], "wall-clock (s)"),
        ("samples", lambda pts: [p.cum_samples for p in pts], "cumulative samples"),
    ):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for run_id, points in series.items():
            ax.plot(xs(points), [p.val_loss for p in points], marker="o", label=run_id)
        ax.axhline(args.target_loss, linestyle="--", linewidth=0.8, color="grey")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("val loss (mean across nodes, pre-merge)")
        ax.legend()
        fig.tight_layout()
        out = args.out_png.with_name(f"{args.out_png.stem}_{suffix}.png")
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"wrote {out}")

    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
