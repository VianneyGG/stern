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
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from video_encoding.training.ddp import _get


def result_json_filename(experiment_name: str, run_tag: str | None) -> str:
    """Stable filename for one experiment+run_tag's result JSON.

    Shared by `write_result_json`'s overwrite path and the watcher's stall
    detector (video_encoding/watcher/results.py) -- both must agree on this
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
