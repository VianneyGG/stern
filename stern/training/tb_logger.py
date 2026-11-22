"""TensorBoard writer and result-JSON output for training runs, as plain functions.

No wrapper class: `make_writer` returns a `SummaryWriter` or `None` (call
sites guard with `if writer:`). Result JSONs match the schema of the sibling
What_Happens_Next project, so cross-project comparisons on the shared cluster
still line up. The files live under this repo's `runs/results/`.
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
    """Stable filename for one experiment and run_tag's result JSON.

    The watcher's stall detector (stern/watcher/results.py) looks up a live
    run's file by this name, so both sides must use this function.
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
    """Write one result JSON for a run. Call only on rank 0.

    The payload holds the caller's `metrics`, the resolved config, and
    environment and reproducibility fields.

    `overwrite=True` writes a stable, timestamp-free filename keyed on
    experiment and run_tag. A walltime-bounded run can then re-emit its
    current state every epoch and the aggregator sees one row per cell. The
    timestamped form only lands at loop exit, which such a run may never reach.
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
    """Replay mesh rounds not yet flushed to `writer`, read from each node's JSONL.

    Nodes are discovered by globbing `mesh_shared/metrics/<run_id>/*.jsonl`.
    A node that missed quorum still has a JSONL and still shows up here.
    Records use a 1-based `round`; scalars go at step `round - 1`.

    The newest round seen is held back until the next call, unless
    `final=True`. A node appends its round-R record only after the merge for
    round R, so a straggler's record can land after a fast node's. Holding
    the newest round back gives stragglers one more merge cycle. A round
    that is flushed is never revisited. This is monitoring only; the JSONL
    on NFS remains the source of truth.

    Returns the highest round flushed, or `last_round` if nothing new was
    flushed. The caller stores the result as the next `last_round`.
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

    Once a run is DONE no merge cycle follows, so this is the only caller that
    passes `final=True`. It keeps the `tb_last_round.txt` bookkeeping that
    run_outer_step.py uses, but writes no merge-specific scalars.

    Never raises. It runs from the watcher's DONE transition
    (stern/watcher/mesh_tick.py), and a TB write failure must not block the run.
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
