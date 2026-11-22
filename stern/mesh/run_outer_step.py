"""Merge CLI wrapping the DiLoCo outer optimizer (`stern/mesh/outer_step.py`)
with a persistent momentum buffer.

Runs as a subprocess from the watcher's epoch-advance logic. It reads the
worker checkpoints in `pending/`, computes the alpha-weighted pseudo-gradient,
takes one Nesterov outer step, and atomically overwrites `master.pt` and the
momentum buffer.

It does not delete pending checkpoints or advance any round counter. The
caller does that, so this module is a retryable merge step: a crash or
non-zero exit just means the next tick retries against the same `pending/`.
There is one hard limit. Once `master.pt` is rewritten the round is applied,
and a later failure (including a caller SIGKILL on timeout, which cannot be
caught here) cannot undo it. `merge_round` therefore detects an already-applied
round (`master.pt` stamped with that round) and reports a no-op success instead
of retrying deltas that are now stale.

Weighting is `data` (FedAvg sample-count proportional) by default and never
falls back silently. A checkpoint missing `n_samples` is a contract violation
from the training side and must fail loudly rather than bias the merge.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Collection
from pathlib import Path
from typing import Any

import torch

from stern.mesh.outer_step import (
    StateDict,
    average_deltas,
    data_weights,
    filter_stale_deltas,
    nesterov_outer_step,
    pseudo_gradient,
    to_fp32_state_dict,
)
from stern.mesh.pending_files import (
    parse_pending_filename,
    pending_files_for_round,
)
from stern.mesh.quantize import quantize_with_error_feedback
from stern.mesh.quorum import NodeStats, token_weights
from stern.training.tb_logger import make_writer, write_mesh_round_scalars

WEIGHTING_CHOICES = ("data", "inverse_loss", "token")

# Dtype that master.pt is written in. fp32 is the default. fp16 halves the NFS
# broadcast but doubles the quantization noise in pseudo_gradient, and the
# trade depends on the model. Measure ||delta||/||w|| per tensor on a real
# round before switching (see _quantize_broadcast_with_error_feedback).
BROADCAST_DTYPE_CHOICES = ("fp32", "fp16")


def _load_worker_checkpoints(
    pending_dir: Path,
    run_id: str,
    *,
    round_num: int | None = None,
    eligible: Collection[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Load this run's pending checkpoints, keyed by node name.

    `pending_dir` is one flat directory shared by every concurrent run, and
    finished runs leave files behind. The glob is therefore always scoped to
    `run_id`, so sibling and dead runs are never averaged in.

    Filenames are `{run_id}_{round}_{node}.pt` (see `pending_files.py`). With
    `round_num` set, only that round's files are read. A straggler that
    finishes a prior round late must not be averaged into a different round.
    `round_num=None` keeps every matching file.

    `eligible=None` keeps every node. Otherwise a node not in `eligible` is
    dropped with a log line, not an error. A quarantined node may keep
    training and writing checkpoints for a few rounds.

    Each checkpoint must carry `n_samples` and exactly one of
    `model_state_dict` (full weights, nothing to diff against) or
    `delta_state_dict` (already `master - worker`). These are validated up
    front so a malformed checkpoint fails before any merge math runs.
    """
    if round_num is not None:
        files = sorted(pending_files_for_round(pending_dir, run_id, round_num))
    else:
        files = sorted(pending_dir.glob(f"{run_id}_*.pt"))
    if not files:
        raise ValueError(
            f"merge_round: no pending checkpoints found for run_id={run_id!r} "
            f"round_num={round_num!r} in {pending_dir}"
        )

    checkpoints: dict[str, dict[str, Any]] = {}
    for f in files:
        parsed = parse_pending_filename(f.stem, run_id)
        if parsed is None:
            continue  # not this run_id's pattern (or predates round-tagging)
        _, node = parsed
        if eligible is not None and node not in eligible:
            print(
                f"run_outer_step: dropped non-roster checkpoint '{f.name}' "
                f"(node={node!r} not eligible for run_id={run_id!r})",
                file=sys.stderr,
            )
            continue
        payload = torch.load(f, map_location="cpu", weights_only=False)
        if "n_samples" not in payload:
            raise ValueError(
                f"merge_round: checkpoint '{f.name}' is missing required key "
                "'n_samples' -- refusing to default a merge weight silently"
            )
        has_model = "model_state_dict" in payload
        has_delta = "delta_state_dict" in payload
        if has_model == has_delta:
            raise ValueError(
                f"merge_round: checkpoint '{f.name}' must have exactly one of "
                "'model_state_dict' or 'delta_state_dict', found "
                f"{'both' if has_model else 'neither'}"
            )
        payload = dict(payload)
        state_key = "model_state_dict" if has_model else "delta_state_dict"
        payload[state_key] = to_fp32_state_dict(payload[state_key])
        checkpoints[node] = payload
    if not checkpoints:
        raise ValueError(
            f"merge_round: no pending checkpoints found for run_id={run_id!r} "
            f"round_num={round_num!r} in {pending_dir} (found files, but none "
            "parsed as this run_id's pending-file pattern)"
        )
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
        if not math.isfinite(val_loss) or val_loss <= 0:
            raise ValueError(
                f"merge_round: non-finite val_loss for '{name}' ({val_loss}) "
                "-- cannot compute an inverse-loss weight"
            )
        inv[name] = 1.0 / val_loss
    total = sum(inv.values())
    return {name: v / total for name, v in inv.items()}


def _token_weights(checkpoints: dict[str, dict[str, Any]]) -> dict[str, float]:
    stats: dict[str, NodeStats] = {}
    for name, ckpt in checkpoints.items():
        if "n_steps" not in ckpt:
            raise ValueError(
                f"merge_round: checkpoint for '{name}' is missing required key "
                "'n_steps', needed for weighting='token'"
            )
        stats[name] = NodeStats(
            c_tokens=int(ckpt["n_samples"]), c_steps=int(ckpt["n_steps"])
        )
    return token_weights(stats)


def _compute_alphas(
    checkpoints: dict[str, dict[str, Any]], weighting: str
) -> dict[str, float]:
    if weighting == "data":
        return data_weights(
            {name: int(ckpt["n_samples"]) for name, ckpt in checkpoints.items()}
        )
    if weighting == "inverse_loss":
        return _inverse_loss_weights(checkpoints)
    if weighting == "token":
        return _token_weights(checkpoints)
    raise ValueError(
        f"merge_round: unknown weighting scheme {weighting!r}, expected one of "
        f"{WEIGHTING_CHOICES}"
    )


def _atomic_save(path: Path, payload: dict[str, Any]) -> None:
    """tmp + rename, matching `stern/watcher/registry.py` /
    `stern/training/checkpoint.py:save()` -- the watcher can be
    killed mid-write, and a rename is atomic on the shared NFS mount."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _quantize_broadcast_with_error_feedback(
    master_sd: StateDict, residual_path: Path | None
) -> StateDict:
    """Cast `master_sd` to fp16 for the NFS broadcast, carrying the rounding residual.

    Every worker reads master.pt each round, so this is the largest NFS term:
    the fp32 master fanned out to every node. fp16 rather than bf16, because
    the weights sit roughly in [-1, 1]. bf16's extra exponent range is unused
    there, and it gives up three mantissa bits.

    Thin wrapper around `quantize.quantize_with_error_feedback`, which the
    worker's pseudo-gradient push also uses.
    """
    return quantize_with_error_feedback(master_sd, residual_path)


def _append_merge_metrics(
    metrics_out: Path,
    *,
    run_id: str,
    round_num: int | None,
    result: dict,
    timeout_forced: bool,
    quorum_relaxed: bool = False,
    effective_quorum_k: int | None = None,
    misses_charged: int = 0,
    misses_excused_job_absent: int = 0,
    misses_correlated: int = 0,
    exclusions_suspended_no_replacement: int = 0,
    requeues_seen: int = 0,
) -> None:
    """Append one JSONL record for this merge round to `metrics_out`.

    Contributors and alpha weights otherwise exist only in this process's
    stdout, which is unusable across a sweep. A failed write is logged and
    swallowed, since metrics must never fail a merge that succeeded.

    `misses_excused_job_absent` and `misses_correlated` stay 0. Every miss is
    counted as `missed_round_count`, with no blame-aware split yet. `requeues_seen`
    stays 0 because the worker's requeue journal is not aggregated on the watcher
    side. These are reported as 0 rather than guessed.
    """
    record = {
        "run_id": run_id,
        "round": round_num,
        "n_contributors": result["n_workers"],
        "contributors": sorted(result["alphas"]),
        "alphas": result["alphas"],
        "timeout_forced": timeout_forced,
        "quorum_relaxed": quorum_relaxed,
        "effective_quorum_k": effective_quorum_k,
        "misses_charged": misses_charged,
        "misses_excused_job_absent": misses_excused_job_absent,
        "misses_correlated": misses_correlated,
        "exclusions_suspended_no_replacement": exclusions_suspended_no_replacement,
        "requeues_seen": requeues_seen,
        "ts": time.time(),
    }
    try:
        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_out, "a") as f:
            f.write(json.dumps(record) + "\n")
    except OSError as exc:
        print(
            f"run_outer_step: WARNING failed to write metrics to {metrics_out} "
            f"(run_id={run_id} round={round_num}): {exc}",
            file=sys.stderr,
        )


def _write_tb_scalars(
    *,
    tb_dir: Path,
    pending_dir: Path,
    run_id: str,
    round_num: int,
    n_workers: int,
    timeout_forced: bool,
    alphas: dict[str, float],
) -> None:
    """Write this merge's contributor scalars, then replay backlogged per-round scalars.

    Never raises: diagnostics must never fail a merge that succeeded.
    `mesh_shared` is derived from `pending_dir` (always
    `mesh_shared/checkpoints/pending`, see `watcher/mesh_tick.py::_do_merging`),
    so callers have no separate flag to keep in sync.
    """
    mesh_shared = pending_dir.parent.parent
    last_round_path = tb_dir.parent / "tb_last_round.txt"
    try:
        last_round = (
            int(last_round_path.read_text().strip()) if last_round_path.exists() else 0
        )
        writer = make_writer(tb_dir, rank=0)
        if writer is None:
            return
        writer.add_scalar("merge/n_contributors", n_workers, round_num)
        writer.add_scalar(
            "merge/timeout_forced", 1.0 if timeout_forced else 0.0, round_num
        )
        for node, alpha in sorted(alphas.items()):
            writer.add_scalar(f"merge/alpha/{node}", alpha, round_num)
        new_last_round = write_mesh_round_scalars(
            writer, mesh_shared, run_id, last_round
        )
        writer.flush()
        writer.close()
        if new_last_round != last_round:
            last_round_path.write_text(str(new_last_round))
    except Exception as exc:  # noqa: BLE001 -- must never fail the merge
        print(
            f"run_outer_step: WARNING failed to write TB scalars to {tb_dir}: {exc}",
            file=sys.stderr,
        )


def filter_nonfinite(
    checkpoints: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Split off checkpoints carrying a nan or inf tensor.

    A node that trains zero samples (it was assigned a shard it does not hold)
    produces nan weights while its val_loss stays healthy, since val shards
    resolve locally on every node. Averaging is not robust to this: one nan
    makes the merged tensor nan regardless of its alpha. So these checkpoints
    must be dropped before weighting.

    Returns `(kept, dropped_names)`. The caller decides whether an empty `kept`
    is fatal.
    """
    kept: dict[str, dict[str, Any]] = {}
    dropped: list[str] = []
    for name, ckpt in checkpoints.items():
        state = ckpt.get("delta_state_dict") or ckpt.get("model_state_dict") or {}
        if all(torch.isfinite(t).all() for t in state.values() if torch.is_tensor(t)):
            kept[name] = ckpt
        else:
            dropped.append(name)
    return kept, dropped


def merge_round(
    pending_dir: Path,
    master_path: Path,
    momentum_path: Path,
    *,
    run_id: str,
    lr: float,
    momentum: float,
    weighting: str = "data",
    round_num: int | None = None,
    allow_missing_momentum: bool = False,
    broadcast_dtype: str = "fp32",
    residual_path: Path | None = None,
    eligible: Collection[str] | None = None,
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

    if broadcast_dtype not in BROADCAST_DTYPE_CHOICES:
        raise ValueError(
            f"merge_round: unknown broadcast_dtype {broadcast_dtype!r}, expected "
            f"one of {BROADCAST_DTYPE_CHOICES}"
        )

    master_payload = torch.load(master_path, map_location="cpu", weights_only=False)
    # Upcast for the same reason worker checkpoints are upcast below: with
    # broadcast_dtype="fp16" master.pt is itself stored fp16, and every
    # pseudo-gradient/momentum operation below must run in fp32 regardless
    # of what dtype the file happened to be written in.
    master_sd: StateDict = to_fp32_state_dict(master_payload["model_state_dict"])
    current_master_version = master_payload.get("master_version")

    if round_num is not None and current_master_version == round_num:
        # merge_round is the only writer of master.pt, and it stamps the round
        # it merged. Reaching here means an earlier attempt completed that
        # write but failed to report success (the caller may SIGKILL on
        # timeout while diagnostics still run). Re-merging is wrong: the deltas
        # were computed against the master this round replaced, so they are all
        # stale. Report a no-op success so the caller clears pending/ and
        # advances. Only a failure between the master and momentum writes loses
        # anything, and then only that round's momentum update.
        print(
            f"run_outer_step: WARNING master.pt is already stamped "
            f"master_version={round_num} (run_id={run_id!r}) -- this round was "
            "already merged and a retry would drop every pending delta as "
            "stale; treating as a no-op success. Check that the momentum "
            f"buffer at {momentum_path} carries that merge's mtime: if the "
            "interrupted attempt died between the two writes, this round's "
            "momentum update was lost.",
            file=sys.stderr,
        )
        return {"n_workers": 0, "alphas": {}, "already_merged": True}

    checkpoints = _load_worker_checkpoints(
        pending_dir, run_id, round_num=round_num, eligible=eligible
    )
    checkpoints, nonfinite = filter_nonfinite(checkpoints)
    if nonfinite:
        print(
            f"run_outer_step: WARNING dropped {len(nonfinite)} checkpoint(s) with "
            f"non-finite weights (run_id={run_id!r}): {sorted(nonfinite)}",
            file=sys.stderr,
        )
    if not checkpoints:
        raise ValueError(
            f"merge_round: every worker checkpoint for run_id={run_id!r} has "
            f"non-finite weights ({sorted(nonfinite)}) -- refusing to merge. A "
            "worker training zero samples reports train_loss=nan while its "
            "val_loss stays healthy, so re-run python -m stern.mesh.validate_seeding before "
            "restarting this run."
        )
    checkpoints, stale = filter_stale_deltas(checkpoints, current_master_version)
    if stale:
        print(
            f"run_outer_step: WARNING dropped {len(stale)} checkpoint(s) pushed "
            f"against a stale master (run_id={run_id!r}): {sorted(stale)}",
            file=sys.stderr,
        )
    if not checkpoints:
        raise ValueError(
            f"merge_round: every worker checkpoint for run_id={run_id!r} "
            f"round_num={round_num!r} was pushed against a stale master "
            f"({sorted(stale)}); master.pt is at master_version="
            f"{current_master_version!r} -- refusing to merge nothing"
        )
    alphas = _compute_alphas(checkpoints, weighting)

    deltas = {
        name: ckpt["delta_state_dict"]
        if "delta_state_dict" in ckpt
        else pseudo_gradient(master_sd, ckpt["model_state_dict"])
        for name, ckpt in checkpoints.items()
    }
    delta = average_deltas(deltas, alphas)

    momentum_buf: StateDict | None = None
    if momentum_path.exists():
        momentum_payload = torch.load(
            momentum_path, map_location="cpu", weights_only=False
        )
        momentum_buf = momentum_payload["momentum_buf"]
    elif round_num is not None and round_num > 0 and not allow_missing_momentum:
        # The buffer lives on watcher-local disk, which a sweep can clear
        # mid-run. Continuing silently would degrade Nesterov to plain
        # averaging for the rest of the run, showing up only as "DiLoCo
        # underperforms". Past round 0 this is fatal unless the caller opts out.
        raise ValueError(
            f"merge_round: no momentum buffer at {momentum_path} but round_num="
            f"{round_num} > 0 -- refusing to silently degrade to plain weighted "
            "averaging; pass allow_missing_momentum=True to override"
        )
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

    if broadcast_dtype == "fp16":
        broadcast_sd = _quantize_broadcast_with_error_feedback(
            new_master, residual_path
        )
    else:
        broadcast_sd = new_master

    _atomic_save(
        master_path,
        {
            "model_state_dict": broadcast_sd,
            "master_version": round_num,
            # Carried from the bootstrap so every merged (and captured) master
            # still names the seed its run was initialised from.
            "init_seed": master_payload.get("init_seed"),
        },
    )
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
    parser.add_argument(
        "--run-id",
        required=True,
        help=(
            "Only merge pending/{run-id}_*.pt. Required, not defaulted: "
            "pending/ is shared by every concurrent run."
        ),
    )
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--momentum", type=Path, required=True)
    parser.add_argument("--outer-lr", type=float, default=0.7)
    parser.add_argument("--outer-momentum", type=float, default=0.9)
    parser.add_argument("--weighting", choices=WEIGHTING_CHOICES, default="data")
    parser.add_argument(
        "--round",
        dest="round_num",
        type=int,
        default=None,
        help="Round being merged; past round 0 a missing momentum buffer is fatal.",
    )
    parser.add_argument("--allow-missing-momentum", action="store_true")
    parser.add_argument(
        "--broadcast-dtype", choices=BROADCAST_DTYPE_CHOICES, default="fp32"
    )
    parser.add_argument(
        "--residual",
        type=Path,
        default=None,
        help="Watcher-local error-feedback buffer, used only with --broadcast-dtype fp16.",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=None,
        help=(
            "Append one JSONL merge record to this path. Omit to skip "
            "(back-compat, no file created)."
        ),
    )
    parser.add_argument(
        "--timeout-forced",
        action="store_true",
        help=(
            "Set by the watcher's forced-merge branch when this round's "
            "checkpoints were merged after ROUND_TIMEOUT_S rather than a "
            "full check-in -- recorded verbatim into --metrics-out."
        ),
    )
    parser.add_argument(
        "--tb-dir",
        type=Path,
        default=None,
        help=(
            "Write this round's TensorBoard scalars here. Omit to skip "
            "(back-compat, no file created)."
        ),
    )
    parser.add_argument(
        "--eligible-nodes",
        type=str,
        default=None,
        help=(
            "Comma-separated roster to accept checkpoints from; drop "
            "(log, don't raise) any others."
        ),
    )
    parser.add_argument(
        "--quorum-relaxed",
        action="store_true",
        help=(
            "Set by the watcher when this round's quorum floor was relaxed "
            "below quorum_k for unreplaceable-missing nodes "
            "-- recorded verbatim into --metrics-out."
        ),
    )
    parser.add_argument("--effective-quorum-k", type=int, default=None)
    parser.add_argument("--misses-charged", type=int, default=0)
    parser.add_argument("--misses-excused-job-absent", type=int, default=0)
    parser.add_argument("--misses-correlated", type=int, default=0)
    parser.add_argument("--exclusions-suspended-no-replacement", type=int, default=0)
    args = parser.parse_args(argv)

    eligible = set(args.eligible_nodes.split(",")) if args.eligible_nodes else None
    try:
        result = merge_round(
            args.pending_dir,
            args.master,
            args.momentum,
            run_id=args.run_id,
            lr=args.outer_lr,
            momentum=args.outer_momentum,
            weighting=args.weighting,
            round_num=args.round_num,
            allow_missing_momentum=args.allow_missing_momentum,
            broadcast_dtype=args.broadcast_dtype,
            residual_path=args.residual,
            eligible=eligible,
        )
    except ValueError as exc:
        print(f"run_outer_step: {exc}", file=sys.stderr)
        return 1

    if result.get("already_merged"):
        print(
            f"run_outer_step: round {args.round_num} of {args.run_id} was "
            "already merged into master.pt -- nothing to do"
        )
        return 0

    if args.metrics_out is not None:
        _append_merge_metrics(
            args.metrics_out,
            run_id=args.run_id,
            round_num=args.round_num,
            result=result,
            timeout_forced=args.timeout_forced,
            quorum_relaxed=args.quorum_relaxed,
            effective_quorum_k=args.effective_quorum_k,
            misses_charged=args.misses_charged,
            misses_excused_job_absent=args.misses_excused_job_absent,
            misses_correlated=args.misses_correlated,
            exclusions_suspended_no_replacement=args.exclusions_suspended_no_replacement,
        )

    if args.tb_dir is not None:
        _write_tb_scalars(
            tb_dir=args.tb_dir,
            pending_dir=args.pending_dir,
            run_id=args.run_id,
            round_num=args.round_num,
            n_workers=result["n_workers"],
            timeout_forced=args.timeout_forced,
            alphas=result["alphas"],
        )

    print(
        f"run_outer_step: merged {result['n_workers']} worker checkpoint(s) "
        f"(weighting={args.weighting}, lr={args.outer_lr}, momentum={args.outer_momentum})"
    )
    for name, alpha in sorted(result["alphas"].items()):
        print(f"  alpha[{name}] = {alpha:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
