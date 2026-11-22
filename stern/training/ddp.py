"""DDP setup/teardown tuned for the SallesInfo cluster.

Cluster context (T1/T2/T3 findings, docs/superpowers/specs/
2026-06-30-cluster-tier1-findings.md):

- 1 GPU per node (SLURM whole-node allocation, no GPU GRES). `LOCAL_RANK`
  is always 0.
- 1GbE fabric; NCCL busBW saturates around 0.115 GB/s for messages ≥16 MB
  (T3). Bandwidth, not latency, is the bottleneck.
- Uses `env://` init, with SLURM env vars mapped in when torchrun didn't
  set them (mirrors the WHN train.py pattern).

Two knobs that matter for this fabric:

- `bucket_cap_mb=25` (vs. torch's 25 MB default, WHN uses 200). Smaller
  buckets pipeline the allreduce with compute better when total gradient
  volume is small (LoRA runs move <20 MB/step per T5 proxy) and the pipe
  is thin.
- `grad_compress='fp16'` (opt-in). fp16 compression hook halves NCCL
  bytes at negligible convergence cost for anything larger than a handful
  of layers. Off by default because it can degrade tiny models; opt in
  for LoRA + full-encoder runs.
"""

from __future__ import annotations

import os
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_ddp(cfg) -> tuple[int, int, int, torch.device]:
    """Initialize DDP from SLURM env vars.

    Returns `(rank, world_size, local_rank, device)`. Safe to call
    unconditionally: single-process runs (`world_size == 1`) skip
    `init_process_group` and return `(0, 1, 0, cuda:0 or cpu)`.
    """
    _slurm_to_torch_env()

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    _apply_nccl_env(cfg)

    if world_size > 1:
        timeout_s = int(_get(cfg, "train.ddp.nccl_timeout_sec", default=300))
        backend = _get(cfg, "train.ddp.backend", default="nccl")
        dist.init_process_group(backend=backend, timeout=timedelta(seconds=timeout_s))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    return rank, world_size, local_rank, device


def wrap_ddp(
    model: torch.nn.Module,
    cfg,
    *,
    local_rank: int,
) -> torch.nn.Module:
    """Wrap `model` in `DistributedDataParallel` with cluster-tuned settings.

    No-op when `world_size == 1`. `static_graph` is disabled whenever the
    training loop uses `no_sync()` for gradient accumulation (they're
    incompatible) or when unused parameters need finding.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return model

    accum_steps = int(_get(cfg, "train.accum_steps", default=1))
    find_unused = bool(_get(cfg, "train.ddp.find_unused_parameters", default=False))
    bucket_cap_mb = int(_get(cfg, "train.ddp.bucket_cap_mb", default=25))
    static_graph = accum_steps <= 1 and not find_unused

    ddp_model = DDP(
        model,
        device_ids=[local_rank] if torch.cuda.is_available() else None,
        output_device=local_rank if torch.cuda.is_available() else None,
        static_graph=static_graph,
        gradient_as_bucket_view=True,
        find_unused_parameters=find_unused,
        bucket_cap_mb=bucket_cap_mb,
    )

    grad_compress = str(_get(cfg, "train.ddp.grad_compress", default="")).lower()
    if grad_compress == "fp16":
        # Halves NCCL bytes at the cost of a small numerical error. The
        # highest-value knob on our 1GbE fabric — enable for any run that
        # spends >5% of step time in allreduce.
        from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import (
            fp16_compress_hook,
        )

        ddp_model.register_comm_hook(None, fp16_compress_hook)

    return ddp_model


def teardown_ddp() -> None:
    """Destroy the process group if one exists. Idempotent."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Mean-reduce a tensor across all ranks in place.

    No-op when DDP isn't initialised (returns the input unchanged) so the
    same call site works in both single- and multi-node runs.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return tensor


def _slurm_to_torch_env() -> None:
    """Map SLURM per-task vars → PyTorch env:// vars.

    Only fills gaps: if torchrun already injected RANK/WORLD_SIZE/LOCAL_RANK
    this is a no-op. MASTER_ADDR / MASTER_PORT must be exported by the
    batch script before `srun` (see slurm/train.sbatch).
    """
    if "RANK" in os.environ or "SLURM_PROCID" not in os.environ:
        return
    os.environ["RANK"] = os.environ["SLURM_PROCID"]
    os.environ["WORLD_SIZE"] = os.environ.get("SLURM_NTASKS", "1")
    os.environ["LOCAL_RANK"] = os.environ.get("SLURM_LOCALID", "0")


def _apply_nccl_env(cfg) -> None:
    nccl_env = _get(cfg, "train.ddp.nccl_env", default=None) or {}
    for key, value in nccl_env.items():
        os.environ.setdefault(str(key), str(value))


def _get(cfg, dotted: str, *, default: Any) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        try:
            node = node[part] if isinstance(node, dict) else getattr(node, part)
        except (KeyError, AttributeError):
            return default
    return node
