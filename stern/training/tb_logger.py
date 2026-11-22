"""TensorBoard writer + WHN-schema result JSON, as plain functions.

No wrapper class: `make_writer` returns a `SummaryWriter` or `None` (call
sites guard with `if writer:`), and `write_result_json` is a standalone
function. Schema parity with the What_Happens_Next repo's result JSONs is
deliberate: both projects share the same cluster, so cross-project isoFLOP /
cost-per-sample comparisons stay meaningful even though the JSONs now live
under this repo's own `runs/results/`, not a shared `.claude/results/` path.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from stern.training.ddp import _get


def result_json_filename(experiment_name: str, run_tag: str | None) -> str:
    """Stable filename for one experiment+run_tag's result JSON.

    Shared by `write_result_json`'s overwrite path and the watcher's stall
    detector (stern/watcher/results.py) -- both must agree on this
    exact name or the watcher will never find a running job's live file.
    """
    suffix = f"-{run_tag}" if run_tag else ""
    return f"{experiment_name}{suffix}.json"


def make_writer(log_dir: Path | str | None, *, rank: int = 0):
    """Return a `SummaryWriter` on rank 0 with a `log_dir`, else `None`."""
    if rank != 0 or log_dir is None:
        return None
    from torch.utils.tensorboard import SummaryWriter

    return SummaryWriter(log_dir=str(log_dir))


def write_result_json(
    metrics: dict[str, Any],
    cfg,
    *,
    results_root: Path | str,
    rank: int = 0,
    overwrite: bool = False,
) -> Path | None:
    """Write one result JSON per run. Call only on rank 0.

    Schema-compatible with WHN's per-run JSONs so `runs/results/` stays
    cross-project comparable. Merges: caller-provided `metrics`, resolved
    cfg snapshot, and environment/reproducibility fields.

    `overwrite=True` writes to a stable, timestamp-free filename keyed on
    experiment+run_tag, so a long run can emit its *current* state every epoch
    and the aggregator sees exactly one up-to-date row per cell mid-run (the
    default timestamped form only lands once, at loop exit — useless for a
    walltime-bounded run that never reaches its epoch horizon).
    """
    if rank != 0:
        return None

    from omegaconf import OmegaConf

    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    experiment_name = _get(cfg, "experiment.name", default="unknown")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_tag = _get(cfg, "run_tag", default=None)

    payload: dict[str, Any] = {
        "timestamp": timestamp,
        "experiment": experiment_name,
        "run_tag": run_tag,
        "env": _env_snapshot(),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "metrics": metrics,
    }

    stable_name = result_json_filename(experiment_name, run_tag)
    if overwrite:
        # Stable name -> re-emitting each epoch overwrites in place instead of
        # littering results_root with one file per epoch.
        out_path = results_root / stable_name
    else:
        out_path = results_root / f"{timestamp}-{stable_name}"
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"Wrote result JSON -> {out_path}", flush=True)
    return out_path


def _env_snapshot() -> dict[str, Any]:
    """Reproducibility fields captured at result-write time."""
    return {
        "git_sha": _git_sha(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
        "hostname": socket.gethostname(),
    }


def write_mesh_round_scalars(
    writer,
    mesh_shared: Path,
    run_id: str,
    last_round: int,
    *,
    final: bool = False,
) -> int:
    """Replay not-yet-flushed mesh rounds from every node's JSONL onto `writer`.

    Nodes are discovered by globbing `mesh_shared/metrics/<run_id>/*.jsonl`
    (no fixed roster) -- run_outer_step.py's merge only ever sees the
    contributors of the round it just merged, but a node that missed
    quorum still has its own JSONL and should still show up here.

    Each node's `_append_round_metrics` (training/diloco/mesh_round.py)
    record uses a 1-based `round`; scalars are written at 0-based step
    `round - 1` to match run_outer_step.py's own `--round`.

    The *newest* round observed across all nodes is held back one call,
    never flushed immediately, unless `final=True`: a worker only appends
    its JSONL record for round R after the merge for round R has already
    run (it's blocked in a post-push barrier the merge itself releases),
    so a fast node's round-R record can land before a straggler's. Holding
    the newest round back gives every node one more merge cycle (~one
    round) to catch up before that round's scalars (including the pool
    mean) are finalized. Once flushed a round is never revisited, so a
    node that never reports for a round simply isn't counted in that
    round's pool mean -- this is monitoring, not the source of truth (the
    JSONL itself, on NFS, is unaffected and still has every record).
    `final=True` is for a run that just reached DONE: there is no future
    merge cycle left to wait for, so every remaining round (including the
    one that would otherwise still be held back) is flushed as-is.

    Returns the highest round number flushed this call, or `last_round`
    unchanged if nothing new was flushed. The caller persists this as the
    next call's `last_round`.
    """
    metrics_dir = mesh_shared / "metrics" / run_id
    per_round: dict[int, dict[str, dict[str, Any]]] = {}
    for jsonl_path in sorted(metrics_dir.glob("*.jsonl")):
        node = jsonl_path.stem
        try:
            lines = jsonl_path.read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            round_ = record.get("round")
            if not isinstance(round_, int) or round_ <= last_round:
                continue
            per_round.setdefault(round_, {})[node] = record

    if not per_round:
        return last_round

    if final:
        rounds_to_flush = sorted(per_round)
    else:
        newest_round = max(per_round)
        rounds_to_flush = sorted(r for r in per_round if r < newest_round)

    new_last_round = last_round
    for round_ in rounds_to_flush:
        step = round_ - 1
        train_losses: list[float] = []
        val_losses: list[float] = []
        samp_s_total = 0.0
        pool_job_id: float | None = None
        for node, record in sorted(per_round[round_].items()):
            writer.add_scalar(f"loss/train/{node}", record["train_loss"], step)
            writer.add_scalar(f"loss/val/{node}", record["val_loss"], step)
            writer.add_scalar(
                f"throughput/samples_per_sec/{node}", record["samp_s"], step
            )
            n_shards = record.get("n_shards")
            if n_shards is not None:
                writer.add_scalar(f"data/n_shards/{node}", n_shards, step)
            lr = record.get("lr")
            if lr is not None:
                writer.add_scalar(f"config/lr/{node}", lr, step)
            train_losses.append(record["train_loss"])
            val_losses.append(record["val_loss"])
            samp_s_total += record["samp_s"]
            job_id = record.get("slurm_job_id")
            if job_id is not None and str(job_id).isdigit():
                writer.add_scalar(f"slurm_job_id/{node}", float(job_id), step)
                if pool_job_id is None:
                    pool_job_id = float(job_id)
        if pool_job_id is not None:
            writer.add_scalar("slurm_job_id/pool", pool_job_id, step)
        if train_losses:
            writer.add_scalar(
                "loss/train/pool_mean",
                sum(train_losses) / len(train_losses),
                step,
            )
        if val_losses:
            writer.add_scalar(
                "loss/val/pool_mean",
                sum(val_losses) / len(val_losses),
                step,
            )
        writer.add_scalar("throughput/samples_per_sec/pool_sum", samp_s_total, step)
        new_last_round = round_

    return new_last_round


def flush_final_tb_rounds(tb_dir: Path, mesh_shared: Path, run_id: str) -> None:
    """Best-effort: flush every round still held back once a run reaches DONE.

    write_mesh_round_scalars's straggler holdback (see its docstring) always
    leaves a run's last round or two unflushed while it's still live -- there
    is no next merge cycle coming once the run is DONE, so this is the one
    call site that passes final=True. Mirrors run_outer_step.py's
    _write_tb_scalars bookkeeping (same tb_last_round.txt convention) but
    carries no merge-specific scalars, since no merge runs at DONE.

    Never raises -- called from the watcher's DONE transition
    (stern/watcher/mesh_tick.py), where a TB write failure must
    never block a run from finishing.
    """
    last_round_path = tb_dir.parent / "tb_last_round.txt"
    try:
        last_round = (
            int(last_round_path.read_text().strip()) if last_round_path.exists() else 0
        )
        writer = make_writer(tb_dir, rank=0)
        if writer is None:
            return
        new_last_round = write_mesh_round_scalars(
            writer, mesh_shared, run_id, last_round, final=True
        )
        writer.flush()
        writer.close()
        if new_last_round != last_round:
            last_round_path.write_text(str(new_last_round))
    except Exception as exc:  # noqa: BLE001 -- must never fail the DONE transition
        print(
            f"flush_final_tb_rounds: WARNING failed to write TB scalars to {tb_dir}: {exc}",
            file=sys.stderr,
        )


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        ).strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None
