#!/usr/bin/env python
"""Recipe C synthetic benchmark — real V-JEPA2 + real LoRA + real DDP, synthetic inputs.

SYNTHETIC DATA BENCHMARK, NOT A TRAINING RUN.  Measures per-step time, peak
VRAM, and DDP gradient payload for Recipe C (LoRA-adapted V-JEPA2 + supervised
VA head), fed torch.randn tensors shaped like real video input instead of real
video crops.  The real bottleneck (compute + VRAM + NCCL comm) is what T5 needs
for the cluster scheduling decision.

See docs/superpowers/specs/2026-07-01-t5-real-recipe-c-benchmark-design.md

Usage (single-node wiring check, from repo root):
    uv run python video_encoding/scripts/bench_recipe_c.py

Usage (multi-node via sbatch):
    sbatch video_encoding/slurm/t5_recipe_c.sbatch
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModel

CHECKPOINT = "facebook/vjepa2-vitl-fpc16-256-ssv2"
WARMUP = 3
ITERS = 15


def _ccc_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Lin's CCC, torch-differentiable.  Biased estimators for training stability."""
    mean_p = pred.mean()
    mean_t = target.mean()
    diff_p = pred - mean_p
    diff_t = target - mean_t
    cov = (diff_p * diff_t).mean()
    var_p = (diff_p**2).mean()
    var_t = (diff_t**2).mean()
    bias_sq = (mean_p - mean_t) ** 2
    return (2.0 * cov) / (var_p + var_t + bias_sq + 1e-8)


def ccc_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - mean(CCC_valence, CCC_arousal).

    Benchmark default only — the real combination strategy (mean vs joint vs
    weighted) is a modelling decision deferred to video_encoding/probe/train.py with real
    data in hand.
    """
    ccc_v = _ccc_torch(pred[:, 0], target[:, 0])
    ccc_a = _ccc_torch(pred[:, 1], target[:, 1])
    return 1.0 - (ccc_v + ccc_a) / 2.0


class VAHead(nn.Module):
    """Inline VA head: mean-pool last_hidden_state → Linear(embed_dim, 2) → tanh.

    Bounds output to [-1, 1] matching the harmonised VA range (preprocess.yaml:
    va_range).  Not promoted to video_encoding/probe/ until real data exists.
    """

    def __init__(self, embed_dim: int = 1024) -> None:
        super().__init__()
        self.proj = nn.Linear(embed_dim, 2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pooled = hidden_states.float().mean(dim=1)  # (B, embed_dim)
        return torch.tanh(self.proj(pooled))  # (B, 2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Local batch size per GPU (T1.7 RTX3090 fwd_bwd ceiling is 4)",
    )
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument(
        "--results-path", type=Path, default=Path(".claude/results/t5_recipe_c.jsonl")
    )
    args = parser.parse_args()

    rank = int(os.environ.get("SLURM_PROCID", 0))
    world = int(os.environ.get("SLURM_NTASKS", 1))
    if torch.cuda.is_available():
        local_gpu = rank % torch.cuda.device_count()
        device = f"cuda:{local_gpu}"
        nccl_backend = "nccl"
        ddp_device_ids = [local_gpu]
    else:
        local_gpu = None
        device = "cpu"
        nccl_backend = "gloo"
        ddp_device_ids = None
        print(
            "WARNING: no CUDA device, running on CPU (wiring check only — metrics invalid)"
        )

    # Model download: happens before DDP init so slow downloads don't hold up
    # the rendezvous barrier on faster nodes.  HF_HOME set to /tmp per node
    # by the sbatch, keeping checkpoint data off NFS.
    model = AutoModel.from_pretrained(CHECKPOINT)
    cfg = model.config
    frames = getattr(cfg, "frames_per_clip", getattr(cfg, "num_frames", 16))
    size = getattr(cfg, "crop_size", getattr(cfg, "image_size", 256))

    # LoRA on query+value — confirmed by introspection of the loaded model:
    # encoder.layer.N.attention.{query,key,value,proj}, embed_dim=1024.
    # NOT q_proj/v_proj, which belong to an unrelated pooler head.
    lora_cfg = LoraConfig(r=args.lora_rank, target_modules=["query", "value"])
    model = get_peft_model(model, lora_cfg)
    # peft already sets requires_grad=False on all non-adapter params.

    head = VAHead(embed_dim=1024)
    model = model.to(device).train()
    head = head.to(device).train()

    if world > 1:
        dist.init_process_group(
            backend=nccl_backend,
            init_method="env://",
            rank=rank,
            world_size=world,
        )
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=ddp_device_ids
        )
        head = torch.nn.parallel.DistributedDataParallel(
            head, device_ids=ddp_device_ids
        )

    all_params = list(model.parameters()) + list(head.parameters())
    trainable_params = sum(p.numel() for p in all_params if p.requires_grad)
    frozen_params = sum(p.numel() for p in all_params if not p.requires_grad)

    optimizer = torch.optim.AdamW([p for p in all_params if p.requires_grad], lr=1e-4)

    # Synthetic tensors shaped like real V-JEPA2 video crops (B, T, C, H, W)
    x = torch.randn(args.batch_size, frames, 3, size, size, device=device)
    targets = torch.rand(args.batch_size, 2, device=device) * 2 - 1

    use_cuda = device != "cpu"
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    step_times: list[float] = []
    loss = torch.tensor(0.0)
    for i in range(WARMUP + ITERS):
        t0 = time.perf_counter()
        optimizer.zero_grad()
        out = model(pixel_values_videos=x)
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        pred = head(h)
        loss = ccc_loss(pred, targets)
        loss.backward()
        optimizer.step()
        if use_cuda:
            torch.cuda.synchronize(device)
        if i >= WARMUP:
            step_times.append(time.perf_counter() - t0)

    peak_vram_gb = (
        torch.cuda.max_memory_allocated(device) / (1024**3) if use_cuda else 0.0
    )
    # DDP communicates gradients only for trainable (LoRA + head) params
    grad_bytes = sum(
        p.grad.numel() * p.grad.element_size()
        for p in all_params
        if p.requires_grad and p.grad is not None
    )

    result = {
        "rank": rank,
        "world_size": world,
        "batch_size": args.batch_size,
        "lora_rank": args.lora_rank,
        "frames": frames,
        "size": size,
        "step_time_s": round(sum(step_times) / len(step_times), 4),
        "peak_vram_gb": round(peak_vram_gb, 3),
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "grad_bytes": grad_bytes,
        "loss": float(loss.detach().cpu()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint": CHECKPOINT,
    }
    print(json.dumps(result))

    if rank == 0:
        args.results_path.parent.mkdir(parents=True, exist_ok=True)
        with args.results_path.open("a") as f:
            f.write(json.dumps(result) + "\n")

    if world > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
