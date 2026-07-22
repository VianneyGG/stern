"""Merge CLI wrapping the DiLoCo outer optimizer (`video_encoding/mesh/outer_step.py`)
with a persistent momentum buffer.

Runs as a subprocess from the watcher's epoch-advance logic (mirroring
`remediate.py`'s existing shell-out pattern): reads whatever worker
checkpoints have landed in `pending/`, computes the alpha-weighted
pseudo-gradient, takes one Nesterov outer-optimizer step, and atomically
overwrites `master.pt` and the momentum buffer. Deliberately does *not*
delete the pending checkpoints or advance any round counter -- that is the
caller's job (Task 11), so this module stays a pure, retryable merge step:
a crash or non-zero exit here just means the next cron tick tries again
against the same pending/ contents.

Weighting is `data` (FedAvg sample-count proportional) by default and
intentionally never falls back silently: a worker checkpoint missing
`n_samples` is a contract violation from the training side (Task 12's
`pretrain.py` follow-up is supposed to always write it) and must fail loud
rather than let some default weight quietly bias the merge.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

from video_encoding.mesh.outer_step import (
    StateDict,
    average_deltas,
    data_weights,
    nesterov_outer_step,
    pseudo_gradient,
)

WEIGHTING_CHOICES = ("data", "inverse_loss")


def _load_worker_checkpoints(pending_dir: Path) -> dict[str, dict[str, Any]]:
    """Load every `*.pt` in `pending_dir`, keyed by filename stem.

    Validates the two keys every downstream computation needs regardless of
    weighting scheme (`model_state_dict`, `n_samples`) up front, so a
    malformed checkpoint fails before any merge math runs rather than
    surfacing as a confusing KeyError deep in `average_deltas`.
    """
    files = sorted(pending_dir.glob("*.pt"))
    if not files:
        raise ValueError(f"merge_round: no pending checkpoints found in {pending_dir}")

    checkpoints: dict[str, dict[str, Any]] = {}
    for f in files:
        payload = torch.load(f, map_location="cpu", weights_only=False)
        if "n_samples" not in payload:
            raise ValueError(
                f"merge_round: checkpoint '{f.name}' is missing required key "
                "'n_samples' -- refusing to default a merge weight silently"
            )
        if "model_state_dict" not in payload:
            raise ValueError(
                f"merge_round: checkpoint '{f.name}' is missing required key "
                "'model_state_dict'"
            )
        checkpoints[f.stem] = payload
    return checkpoints


def _inverse_loss_weights(checkpoints: dict[str, dict[str, Any]]) -> dict[str, float]:
    inv: dict[str, float] = {}
    for name, ckpt in checkpoints.items():
        if "val_loss" not in ckpt:
            raise ValueError(
                f"merge_round: checkpoint for '{name}' is missing required key "
                "'val_loss', needed for weighting='inverse_loss'"
            )
        val_loss = float(ckpt["val_loss"])
        if val_loss <= 0:
            raise ValueError(
                f"merge_round: non-positive val_loss for '{name}' ({val_loss}) "
                "-- cannot compute an inverse-loss weight"
            )
        inv[name] = 1.0 / val_loss
    total = sum(inv.values())
    return {name: v / total for name, v in inv.items()}


def _compute_alphas(
    checkpoints: dict[str, dict[str, Any]], weighting: str
) -> dict[str, float]:
    if weighting == "data":
        return data_weights(
            {name: int(ckpt["n_samples"]) for name, ckpt in checkpoints.items()}
        )
    if weighting == "inverse_loss":
        return _inverse_loss_weights(checkpoints)
    raise ValueError(
        f"merge_round: unknown weighting scheme {weighting!r}, expected one of "
        f"{WEIGHTING_CHOICES}"
    )


def _atomic_save(path: Path, payload: dict[str, Any]) -> None:
    """tmp + rename, matching `video_encoding/watcher/registry.py` /
    `video_encoding/training/checkpoint.py:save()` -- the watcher can be
    killed mid-write, and a rename is atomic on the shared NFS mount."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def merge_round(
    pending_dir: Path,
    master_path: Path,
    momentum_path: Path,
    *,
    lr: float,
    momentum: float,
    weighting: str = "data",
) -> dict:
    """Run one DiLoCo outer-optimizer merge round.

    Reads every pending worker checkpoint, computes alpha weights per
    `weighting`, averages the master-minus-worker pseudo-gradients, and
    takes one Nesterov outer step -- persisting the resulting master weights
    and momentum buffer atomically. Returns
    `{"n_workers": int, "alphas": dict[str, float]}`.
    """
    pending_dir = Path(pending_dir)
    master_path = Path(master_path)
    momentum_path = Path(momentum_path)

    checkpoints = _load_worker_checkpoints(pending_dir)
    alphas = _compute_alphas(checkpoints, weighting)

    master_payload = torch.load(master_path, map_location="cpu", weights_only=False)
    master_sd: StateDict = master_payload["model_state_dict"]

    deltas = {
        name: pseudo_gradient(master_sd, ckpt["model_state_dict"])
        for name, ckpt in checkpoints.items()
    }
    delta = average_deltas(deltas, alphas)

    momentum_buf: StateDict | None = None
    if momentum_path.exists():
        momentum_payload = torch.load(
            momentum_path, map_location="cpu", weights_only=False
        )
        momentum_buf = momentum_payload["momentum_buf"]
    else:
        # Expected only on round 1 -- warn loudly rather than silently
        # degrading to plain averaging, so a real bug elsewhere (e.g. the
        # buffer failing to write) doesn't masquerade as "DiLoCo underperforms".
        print(
            f"WARNING: merge_round: no momentum buffer found at {momentum_path} "
            "-- proceeding without momentum (expected only on round 1; this "
            "round degrades to plain weighted averaging)",
            file=sys.stderr,
        )

    new_master, new_buf = nesterov_outer_step(
        master_sd, delta, momentum_buf, lr=lr, momentum=momentum
    )

    _atomic_save(master_path, {"model_state_dict": new_master})
    _atomic_save(momentum_path, {"momentum_buf": new_buf})

    return {"n_workers": len(checkpoints), "alphas": alphas}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "DiLoCo outer-optimizer merge step: averages pending worker "
            "checkpoints' pseudo-gradients into master.pt with persistent "
            "Nesterov momentum."
        )
    )
    parser.add_argument("--pending-dir", type=Path, required=True)
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--momentum", type=Path, required=True)
    parser.add_argument("--outer-lr", type=float, default=0.7)
    parser.add_argument("--outer-momentum", type=float, default=0.9)
    parser.add_argument("--weighting", choices=WEIGHTING_CHOICES, default="data")
    args = parser.parse_args(argv)

    try:
        result = merge_round(
            args.pending_dir,
            args.master,
            args.momentum,
            lr=args.outer_lr,
            momentum=args.outer_momentum,
            weighting=args.weighting,
        )
    except ValueError as exc:
        print(f"run_outer_step: {exc}", file=sys.stderr)
        return 1

    print(
        f"run_outer_step: merged {result['n_workers']} worker checkpoint(s) "
        f"(weighting={args.weighting}, lr={args.outer_lr}, momentum={args.outer_momentum})"
    )
    for name, alpha in sorted(result["alphas"].items()):
        print(f"  alpha[{name}] = {alpha:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
