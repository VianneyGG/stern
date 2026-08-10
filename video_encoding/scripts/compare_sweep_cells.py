"""Reduce cars-pool DiLoCo sweep cell metrics to a baseline-relative
comparison table. See docs/superpowers/plans/2026-08-10-cars-mesh-hp-sweep.md
and its "Corrected Assumptions" for why the baseline is pool-cars-full-50r
(a cold start), not pool-cars-full-100r (a continuation).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def mean_val_loss_at_round(metrics_dir: Path, run_id: str, round_num: int) -> float:
    losses = []
    run_dir = Path(metrics_dir) / run_id
    for path in sorted(run_dir.glob("*.jsonl")) if run_dir.is_dir() else []:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if int(rec["round"]) == round_num:
                losses.append(float(rec["val_loss"]))
    if not losses:
        raise ValueError(f"{run_id}: no node reported round {round_num}")
    return sum(losses) / len(losses)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--baseline-run-id", required=True)
    parser.add_argument("--baseline-round", type=int, required=True)
    parser.add_argument(
        "cells",
        nargs="+",
        help="run_id:round pairs, e.g. sw3-cars-lr060m090:40",
    )
    args = parser.parse_args()

    baseline = mean_val_loss_at_round(
        args.metrics_dir, args.baseline_run_id, args.baseline_round
    )
    print(f"baseline {args.baseline_run_id}@{args.baseline_round}: {baseline:.4f}")
    for cell in args.cells:
        run_id, round_str = cell.rsplit(":", 1)
        loss = mean_val_loss_at_round(args.metrics_dir, run_id, int(round_str))
        print(f"{run_id}@{round_str}: {loss:.4f}  (delta {loss - baseline:+.4f})")


if __name__ == "__main__":
    main()
