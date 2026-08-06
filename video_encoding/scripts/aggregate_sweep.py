"""Aggregate SSL sweep result JSONs into a throughput/quality table.

Reads every `runs/results/*.json` (WHN schema; written by
`training/tb_logger.write_result_json`), keys rows by `run_tag`, and reports:

- a tidy CSV + aligned console table of config x metrics,
- `scaling_efficiency` for Phase-1 cells that have an N=1 baseline at the same
  local batch: efficiency = samples/s(N) / (N * samples/s(N=1)),
- the Pareto-best throughput cell and the best-converging quality cell.

Stdlib only — no pandas, so it runs on a bare checkout.

Usage:
  uv run python video_encoding/scripts/aggregate_sweep.py \
      [--results-dir runs/results] [--out sweep.csv]
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path

# Metric columns pulled from each result's `metrics` dict, in display order.
COLUMNS = [
    "run_tag",
    "world_size",
    "local_batch",
    "accum_steps",
    "effective_batch",
    "precision",
    "grad_compress",
    "num_workers",
    "samples_per_s_global",
    "samples_per_s_per_gpu",
    "peak_vram_gb",
    "step_time_ms_mean",
    "recon_loss",
    "nan_steps",
]


def load_rows(results_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(glob.glob(str(results_dir / "*.json"))):
        try:
            payload = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError):
            continue
        metrics = payload.get("metrics") or {}
        if "samples_per_s_global" not in metrics:
            continue  # not a sweep run (e.g. a probe result)
        row = {c: metrics.get(c) for c in COLUMNS}
        row["run_tag"] = payload.get("run_tag") or payload.get("experiment")
        row["_file"] = Path(path).name
        rows.append(row)
    return rows


def add_scaling_efficiency(rows: list[dict]) -> None:
    """Fill `scaling_efficiency` using the N=1 cell at the same local batch."""
    baseline: dict[int, float] = {}  # local_batch -> samples/s at world_size 1
    for r in rows:
        if r.get("world_size") == 1 and r.get("samples_per_s_global"):
            baseline[r["local_batch"]] = r["samples_per_s_global"]
    for r in rows:
        n = r.get("world_size")
        b = r.get("local_batch")
        g = r.get("samples_per_s_global")
        base = baseline.get(b)
        if n and g and base and n > 0:
            r["scaling_efficiency"] = round(g / (n * base), 3)
        else:
            r["scaling_efficiency"] = None


def fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.2f}"
    return "" if v is None else str(v)


def print_table(rows: list[dict], cols: list[str]) -> None:
    widths = {c: max(len(c), *(len(fmt(r.get(c))) for r in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in sorted(rows, key=lambda x: str(x.get("run_tag"))):
        print("  ".join(fmt(r.get(c)).ljust(widths[c]) for c in cols))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="runs/results", type=Path)
    ap.add_argument("--out", default="sweep.csv", type=Path)
    args = ap.parse_args()

    rows = load_rows(args.results_dir)
    if not rows:
        print(f"No sweep result JSONs found under {args.results_dir}")
        return
    add_scaling_efficiency(rows)

    cols = COLUMNS + ["scaling_efficiency"]
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print_table(rows, cols)
    print()

    best_tp = max(rows, key=lambda r: r.get("samples_per_s_global") or 0)
    print(
        f"Best throughput : {best_tp['run_tag']} "
        f"= {fmt(best_tp['samples_per_s_global'])} samp/s "
        f"(N={best_tp['world_size']}, batch={best_tp['local_batch']}, "
        f"accum={best_tp['accum_steps']}, peak={fmt(best_tp['peak_vram_gb'])}GB)"
    )
    # math.isfinite excludes both None and NaN -- a NaN recon_loss (all steps
    # in an epoch were non-finite, see `nan_steps`) must never win "best": NaN
    # comparisons are always False, so an unfiltered min() over a NaN-mixed
    # list is order-dependent and can silently pick the NaN row.
    quality = [
        r
        for r in rows
        if isinstance(r.get("recon_loss"), (int, float))
        and math.isfinite(r["recon_loss"])
    ]
    if quality:
        best_q = min(quality, key=lambda r: r["recon_loss"])
        print(
            f"Best recon loss : {best_q['run_tag']} "
            f"= {fmt(best_q['recon_loss'])} (eff_batch={best_q['effective_batch']})"
        )
    print(f"\nWrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
