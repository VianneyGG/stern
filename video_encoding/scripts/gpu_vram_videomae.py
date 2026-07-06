#!/usr/bin/env python
"""T5 Phase 3: fresh VRAM / step-time ramp for VideoMAEv2 + LoRA + the MAE
decoder (Recipe B's actual training shape), on one GPU node.

Does NOT reuse the master-branch tier1-findings numbers -- those were
measured for V-JEPA2 ViT-L (326M params), a much larger model than
VideoMAEv2-Base. Uses one real downloaded TalkingCelebs-mini clip (not
synthetic tensors) resampled to fill each batch, exercising the real
frame_decoder -> encoder+LoRA -> ssl_adapt-shaped forward/backward path.

Usage (from repo root, on a GPU node):
    PYTHONPATH=. uv run python video_encoding/scripts/gpu_vram_videomae.py \\
        --clip-path /Data/$USER/talkingcelebs_mini/raw/train/id00000/ZNYmK19-d0U/000.mp4 \\
        --results-path .claude/results/deep_T3_videomae_vram.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import VideoMAEForPreTraining

from video_encoding.data.frame_decoder import decode_frames
from video_encoding.encoders.lora import attach_lora
from video_encoding.ssl import tube_mask

CHECKPOINT = "OpenGVLab/VideoMAEv2-Base"
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
WARMUP = 3
ITERS = 10


def build_model(lora_rank: int, device: str):
    model = VideoMAEForPreTraining.from_pretrained(CHECKPOINT)
    model.videomae = attach_lora(model.videomae, rank=lora_rank)
    return model.to(device)


def trainable_and_frozen_counts(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, frozen


def run_batch_size(model, base_frame: torch.Tensor, batch_size: int, device: str) -> dict:
    num_frames = model.config.num_frames
    tubelet_size = model.config.tubelet_size
    num_tubelets = num_frames // tubelet_size
    total_patches = model.videomae.base_model.embeddings.num_patches
    num_patches_per_frame = total_patches // num_tubelets

    pixel_values = base_frame.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1).to(device)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    step_times = []
    for i in range(WARMUP + ITERS):
        bool_masked_pos = torch.stack(
            [tube_mask(num_patches_per_frame, num_tubelets, 0.90) for _ in range(batch_size)]
        ).to(device)

        start = time.perf_counter()
        optimizer.zero_grad()
        outputs = model(pixel_values=pixel_values, bool_masked_pos=bool_masked_pos)
        outputs.loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        if i >= WARMUP:
            step_times.append(elapsed)

    peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
    trainable, frozen = trainable_and_frozen_counts(model)
    grad_bytes = sum(
        p.grad.numel() * p.grad.element_size()
        for p in model.parameters()
        if p.requires_grad and p.grad is not None
    )
    return {
        "batch_size": batch_size,
        "peak_vram_gb": round(peak_vram_gb, 3),
        "step_time_s": round(sum(step_times) / len(step_times), 4),
        "trainable_params": trainable,
        "frozen_params": frozen,
        "grad_bytes": grad_bytes,
        "loss": float(outputs.loss.detach().cpu()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-path", type=Path, required=True)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--results-path", type=Path, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA device available, running on CPU (slow, for wiring-check only)")

    model = build_model(args.lora_rank, device)
    model.train()

    frames = decode_frames(args.clip_path, num_frames=model.config.num_frames)
    from transformers import VideoMAEImageProcessor
    processor = VideoMAEImageProcessor.from_pretrained(CHECKPOINT)
    base_frame = processor(frames, return_tensors="pt")["pixel_values"][0]

    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    with args.results_path.open("a") as f:
        for batch_size in BATCH_SIZES:
            try:
                result = run_batch_size(model, base_frame, batch_size, device)
            except torch.cuda.OutOfMemoryError:
                print(f"batch_size={batch_size}: OOM, stopping ramp")
                break
            result["timestamp"] = datetime.now(timezone.utc).isoformat()
            result["checkpoint"] = CHECKPOINT
            result["lora_rank"] = args.lora_rank
            print(json.dumps(result))
            f.write(json.dumps(result) + "\n")
            f.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
