"""Does temporal order carry VA signal at all?

One frozen encoder, one clip set, one head architecture, three input conditions:
normal, temporally shuffled, and frozen-frame. A separate head is trained per
condition, because a head trained on dynamic features and scored on static ones
is simply out of distribution.

The shuffled-vs-normal comparison is the one that matters. Absolute CCC on
VEATIC is confounded by a large domain gap (VoxCeleb2 talking heads to
Hollywood clips) and by the lack of a face-cropping stage, so a low number alone
is uninterpretable. The paired comparison controls for both.

Run once per (checkpoint, condition) through the training entrypoint, then aggregate:

    ~/.local/bin/uv run --no-sync python -m stern.evaluation.probes.temporal_conditions \\
        --pred-dir runs/results/probe_veatic --out runs/results/probe_gate.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from stern.evaluation.ccc import concordance_correlation_coefficient


def paired_bootstrap_ccc_delta(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    target: np.ndarray,
    *,
    n_resamples: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """95% CI on CCC(pred_a) - CCC(pred_b), by paired bootstrap over windows.

    Windows are resampled as pairs, because both conditions are scored on the
    same windows. Resampling them independently would widen the interval.
    """
    rng = np.random.default_rng(seed)
    n = len(target)
    delta = concordance_correlation_coefficient(
        pred_a, target
    ) - concordance_correlation_coefficient(pred_b, target)
    deltas = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        deltas[i] = concordance_correlation_coefficient(
            pred_a[idx], target[idx]
        ) - concordance_correlation_coefficient(pred_b[idx], target[idx])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(delta), float(lo), float(hi)


def gate_verdict(lo: float, hi: float) -> str:
    """Classify a CI on a CCC difference.

    open: temporal order carries VA signal, so the motion work is justified.
    closed: no measurable temporal contribution on VEATIC. Do not generalise.
    broken: shuffling raised CCC, which should not happen.
    """
    if lo > 0.0:
        return "open"
    if hi < 0.0:
        return "broken"
    return "closed"


CONDITIONS = ("normal", "shuffled", "frozen")


def prediction_path(pred_dir: Path, checkpoint: str, condition: str) -> Path:
    """Where one (checkpoint, condition) run's predictions live."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}: expected {CONDITIONS}")
    return pred_dir / f"{checkpoint}__{condition}.npz"


def parse_prediction_path(path: Path) -> tuple[str, str]:
    """Inverse of prediction_path.

    The separator is a double underscore because checkpoint names contain single
    ones. Splitting `random_init__normal` on '_' would give ('random', 'init__normal').
    """
    checkpoint, _, condition = path.stem.partition("__")
    if condition not in CONDITIONS:
        raise ValueError(f"cannot parse {path.name!r} into (checkpoint, condition)")
    return checkpoint, condition


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pred-dir",
        type=Path,
        required=True,
        help="directory of <checkpoint>__<condition>.npz files with 'pred' and 'target'",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-resamples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    loaded: dict[tuple[str, str], np.lib.npyio.NpzFile] = {}
    for path in sorted(args.pred_dir.glob("*.npz")):
        loaded[parse_prediction_path(path)] = np.load(path)

    report: dict[str, dict] = {}
    for checkpoint in sorted({c for c, _ in loaded}):
        normal = loaded.get((checkpoint, "normal"))
        if normal is None:
            continue
        entry: dict = {
            "ccc": {
                cond: float(
                    concordance_correlation_coefficient(
                        loaded[(checkpoint, cond)]["pred"],
                        loaded[(checkpoint, cond)]["target"],
                    )
                )
                for c, cond in loaded
                if c == checkpoint
            }
        }
        for cond in ("shuffled", "frozen"):
            other = loaded.get((checkpoint, cond))
            if other is None:
                continue
            delta, lo, hi = paired_bootstrap_ccc_delta(
                normal["pred"],
                other["pred"],
                normal["target"],
                n_resamples=args.n_resamples,
                seed=args.seed,
            )
            entry[f"normal_minus_{cond}"] = {"delta": delta, "lo": lo, "hi": hi}
            if cond == "shuffled":
                entry["gate"] = gate_verdict(lo, hi)
        report[checkpoint] = entry
        print(f"{checkpoint:>12} {entry.get('gate', 'n/a'):>7} {entry['ccc']}")

    args.out.write_text(json.dumps(report, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
