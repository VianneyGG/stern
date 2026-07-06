"""SSL objective: VideoMAEv2 masked-pixel reconstruction with the training
machinery needed for real multi-node runs on the SallesInfo cluster
(docs/superpowers/specs/2026-07-01-training-pipeline-design.md §4.6).

Design shape mirrors What_Happens_Next's `src/train.py`: one public entry
point (`ssl_adapt`) plus a small set of private helpers, no class
hierarchy. The public signature keeps the original SSL-specific kwargs
(``lora_rank``/``mask_ratio``/``epochs``/``lr``) and layers the new
runtime knobs (DDP rank, AMP, accum, checkpointing, scheduler) on top —
callers that only pass the originals still get a working single-process
run, which is what the smoke test in ``tests/test_ssl_smoke.py`` depends
on.

LoRA is applied to the `videomae` (encoder) submodule only. The decoder
(`decoder`, `encoder_to_decoder`, `mask_token` -- confirmed real submodule
names by inspecting transformers.VideoMAEForPreTraining.__init__) stays
fully trainable: it's a from-scratch head that has to learn reconstruction,
so freezing it would give it nothing to learn with. Only the encoder's base
weights are frozen; its LoRA params and the whole decoder receive gradients.

Tube masking: HF's `bool_masked_pos` (True = hidden from the encoder, see
VideoMAEEmbeddings.forward's `embeddings[~bool_masked_pos]`) is generated
once per spatial position and repeated identically across every temporal
tubelet, matching VideoMAE's original tube-masking scheme (not independent
per-tubelet masks).
"""

from __future__ import annotations

import contextlib
import math
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from transformers import VideoMAEForPreTraining

from video_encoding.encoders.lora import attach_lora
from video_encoding.training import checkpoint as ckpt_io
from video_encoding.training import flops as flops_mod
from video_encoding.training.ddp import all_reduce_mean, wrap_ddp
from video_encoding.training.logger import make_writer, write_result_json


def tube_mask(
    num_patches_per_frame: int,
    num_tubelets: int,
    mask_ratio: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Boolean mask of shape (num_tubelets * num_patches_per_frame,), True=masked.

    One spatial pattern is sampled and tiled identically across every
    tubelet (a "tube" through time), not resampled per tubelet.
    """
    num_masked = round(num_patches_per_frame * mask_ratio)
    spatial_mask = torch.zeros(num_patches_per_frame, dtype=torch.bool)
    masked_indices = torch.randperm(num_patches_per_frame, generator=generator)[
        :num_masked
    ]
    spatial_mask[masked_indices] = True
    return spatial_mask.repeat(num_tubelets)


@dataclass
class _MaskMeta:
    """Cached mask-geometry values so we don't recompute them per step."""

    num_patches_per_frame: int
    num_tubelets: int
    input_shape: tuple[int, ...]  # (1, T, C, H, W) for FLOPs sizing


def ssl_adapt(
    checkpoint: str,
    train_dataloader,
    *,
    val_dataloader=None,
    lora_rank: int = 8,
    mask_ratio: float = 0.90,
    epochs: int = 10,
    lr: float = 1e-4,
    # Runtime knobs (safe defaults keep the smoke-test path single-process).
    accum_steps: int = 1,
    use_amp: bool = False,
    grad_clip: float = 0.0,
    warmup_epochs: int = 0,
    checkpoint_path: Path | str | None = None,
    resume: bool = True,
    save_latest: bool = True,
    results_root: Path | str | None = None,
    log_dir: Path | None = None,
    rank: int = 0,
    world_size: int = 1,
    local_rank: int = 0,
    device: torch.device | str | None = None,
    cfg=None,
    seed: int = 0,
) -> nn.Module:
    """Adapt `checkpoint` via LoRA + masked reconstruction.

    Returns the LoRA-wrapped VideoMAEForPreTraining model with DDP/compile
    wrappers stripped (`checkpoint.unwrap`), the same on every rank.
    """
    device = _resolve_device(device, local_rank)
    torch.manual_seed(seed + rank)

    writer = make_writer(log_dir, rank=rank)
    if rank == 0:
        print(
            f"ssl_adapt start: checkpoint={checkpoint} rank={rank}/{world_size}",
            flush=True,
        )

    model, mask_meta = _build_model(checkpoint, lora_rank, device)
    model = wrap_ddp(model, cfg, local_rank=local_rank) if cfg is not None else model

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    scheduler = _cosine_lr(
        optimizer, epochs=epochs, warmup=warmup_epochs, start_epoch=0
    )
    scaler = (
        torch.amp.GradScaler("cuda") if (use_amp and device.type == "cuda") else None
    )

    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
    latest_path = _latest_sibling(checkpoint_path)
    start_epoch, best_metric, disk_best = _maybe_resume(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        best_path=checkpoint_path,
        latest_path=latest_path,
        resume=resume,
    )
    if start_epoch > 0:
        scheduler = _cosine_lr(
            optimizer, epochs=epochs, warmup=warmup_epochs, start_epoch=start_epoch
        )
        if rank == 0:
            print(f"resumed from epoch={start_epoch} best={best_metric}", flush=True)

    stop_flag = [False]
    prev_handlers = _install_stop_handlers(stop_flag, rank)

    is_ssl = True
    best_metric = best_metric if best_metric is not None else float("inf")
    step_cursor = 0

    try:
        for epoch in range(start_epoch, epochs):
            _set_sampler_epoch(train_dataloader, epoch)
            t_epoch = time.monotonic()
            train_loss, step_cursor = _train_epoch(
                model=model,
                loader=train_dataloader,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                mask_meta=mask_meta,
                mask_ratio=mask_ratio,
                accum_steps=accum_steps,
                grad_clip=grad_clip,
                writer=writer,
                step_offset=step_cursor,
            )
            val_loss = (
                _validate_epoch(
                    model=model,
                    loader=val_dataloader,
                    device=device,
                    mask_meta=mask_meta,
                    mask_ratio=mask_ratio,
                )
                if val_dataloader is not None
                else train_loss
            )

            train_loss, val_loss, stop_now = _reduce_epoch(
                train_loss, val_loss, stop_flag[0], device
            )
            stop_flag[0] = stop_now
            scheduler.step()

            improved = val_loss < best_metric
            if improved:
                best_metric = val_loss

            if writer is not None:
                for tag, value in {
                    "ssl/train_loss": train_loss,
                    "ssl/val_loss": val_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch_s": time.monotonic() - t_epoch,
                }.items():
                    writer.add_scalar(tag, value, epoch)
            if rank == 0:
                print(
                    f"epoch {epoch + 1}/{epochs}  train_loss={train_loss:.4f}  "
                    f"val_loss={val_loss:.4f}  best={best_metric:.4f}",
                    flush=True,
                )

            if rank == 0 and checkpoint_path is not None:
                payload = _payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_metric=best_metric,
                    val_metric=val_loss,
                    cfg=cfg,
                )
                if improved:
                    _, disk_best = ckpt_io.save_best(
                        checkpoint_path,
                        payload,
                        best_metric,
                        is_ssl=is_ssl,
                        cached_disk_best=disk_best,
                    )
                if save_latest and latest_path is not None:
                    ckpt_io.save(latest_path, payload)

            if stop_flag[0]:
                if rank == 0:
                    print(
                        "Stop signal honoured — exiting after checkpoint write.",
                        flush=True,
                    )
                break
    finally:
        _restore_handlers(prev_handlers)
        if rank == 0 and results_root is not None and cfg is not None:
            _write_final_json(
                cfg=cfg,
                results_root=Path(results_root),
                model=model,
                mask_meta=mask_meta,
                device=device,
                best_metric=best_metric,
                epochs_trained=epochs,
            )
        if writer is not None:
            writer.close()

    return ckpt_io.unwrap(model)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _resolve_device(device, local_rank: int) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def _build_model(
    checkpoint: str, lora_rank: int, device: torch.device
) -> tuple[nn.Module, _MaskMeta]:
    model = VideoMAEForPreTraining.from_pretrained(checkpoint)
    model.videomae = attach_lora(model.videomae, rank=lora_rank)
    model = model.to(device)

    num_frames = model.config.num_frames
    tubelet_size = model.config.tubelet_size
    num_tubelets = num_frames // tubelet_size
    total_patches = model.videomae.base_model.embeddings.num_patches
    num_patches_per_frame = total_patches // num_tubelets

    image_size = int(getattr(model.config, "image_size", 224))
    mask_meta = _MaskMeta(
        num_patches_per_frame=num_patches_per_frame,
        num_tubelets=num_tubelets,
        input_shape=(1, num_frames, 3, image_size, image_size),
    )
    return model, mask_meta


def _train_epoch(
    *,
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    mask_meta: _MaskMeta,
    mask_ratio: float,
    accum_steps: int,
    grad_clip: float,
    writer,
    step_offset: int,
) -> tuple[float, int]:
    """Run one training epoch. Returns `(avg_loss, next_step_offset)`.

    `step_offset` is a monotonic TB-step counter threaded across epochs by
    the caller -- computing it from `epoch * len(loader)` breaks for any
    loader without a static `__len__` (e.g. an IterableDataset), silently
    aliasing every epoch's scalar events onto the same step indices.
    """
    model.train()
    running_loss = 0.0
    running_count = 0
    optimizer.zero_grad(set_to_none=True)

    try:
        num_steps = len(loader)
    except TypeError:
        num_steps = None

    global_step = step_offset
    for step, pixel_values in enumerate(loader):
        pixel_values = pixel_values.to(device, non_blocking=True)
        batch_size = pixel_values.shape[0]
        bool_masked_pos = _sample_mask(mask_meta, mask_ratio, batch_size, device)

        is_boundary = (step + 1) % accum_steps == 0 or (
            num_steps is not None and step + 1 == num_steps
        )
        # Skip DDP allreduce on non-boundary micro-steps. `no_sync` requires
        # a DDP-wrapped model; single-process runs fall through to nullcontext.
        no_sync_ctx = (
            model.no_sync()
            if not is_boundary and hasattr(model, "no_sync")
            else contextlib.nullcontext()
        )
        with no_sync_ctx:
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    outputs = model(
                        pixel_values=pixel_values, bool_masked_pos=bool_masked_pos
                    )
                    loss = outputs.loss / accum_steps
                scaler.scale(loss).backward()
            else:
                outputs = model(
                    pixel_values=pixel_values, bool_masked_pos=bool_masked_pos
                )
                loss = outputs.loss / accum_steps
                loss.backward()

        running_loss += float(loss.item()) * accum_steps * batch_size
        running_count += batch_size

        if is_boundary:
            if scaler is not None:
                if grad_clip > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if writer is not None:
            writer.add_scalar("ssl/recon_loss", float(outputs.loss.item()), global_step)
        global_step += 1

    return running_loss / max(running_count, 1), global_step


@torch.no_grad()
def _validate_epoch(
    *,
    model: nn.Module,
    loader,
    device: torch.device,
    mask_meta: _MaskMeta,
    mask_ratio: float,
) -> float:
    model.eval()
    running_loss = 0.0
    running_count = 0
    # Deterministic mask across ranks/epochs so val loss is comparable.
    gen = torch.Generator().manual_seed(0)
    for pixel_values in loader:
        pixel_values = pixel_values.to(device, non_blocking=True)
        batch_size = pixel_values.shape[0]
        bool_masked_pos = _sample_mask(
            mask_meta, mask_ratio, batch_size, device, generator=gen
        )
        outputs = model(pixel_values=pixel_values, bool_masked_pos=bool_masked_pos)
        running_loss += float(outputs.loss.item()) * batch_size
        running_count += batch_size
    return running_loss / max(running_count, 1)


def _sample_mask(
    mask_meta: _MaskMeta,
    mask_ratio: float,
    batch_size: int,
    device: torch.device,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    return torch.stack(
        [
            tube_mask(
                mask_meta.num_patches_per_frame,
                mask_meta.num_tubelets,
                mask_ratio,
                generator=generator,
            )
            for _ in range(batch_size)
        ]
    ).to(device)


def _cosine_lr(
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int,
    warmup: int,
    start_epoch: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    def _lambda(epoch: int) -> float:
        if warmup > 0 and epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(epochs - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, _lambda, last_epoch=max(start_epoch - 1, -1)
    )


def _maybe_resume(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    best_path: Path | None,
    latest_path: Path | None,
    resume: bool,
) -> tuple[int, float | None, float | None]:
    """Resume from the newest-touched checkpoint. Prefers `_latest` (per-epoch
    snapshot, up to 1 epoch stale) over `best` (only updated on improvement).

    `disk_best` is peeked whenever `best_path` exists, independent of
    `resume` -- `resume=False` must still seed `checkpoint.save_best`'s
    anti-regression guard, or a forced restart pointed at a checkpoint dir
    that already holds a better checkpoint would silently overwrite it
    with a worse one on the very first epoch.

    Returns `(start_epoch, best_metric, cached_disk_best)`.
    """
    disk_best = ckpt_io.peek_best(best_path) if best_path is not None else None
    if not resume or best_path is None:
        return 0, None, disk_best
    source = latest_path if (latest_path and latest_path.exists()) else best_path
    if not source.exists():
        return 0, None, disk_best
    start_epoch, best_metric = ckpt_io.load(
        source,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        expected_model_name="videomae_ssl",
        expected_optimizer_name="adam",
    )
    return start_epoch, best_metric, disk_best


def _payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_metric: float,
    val_metric: float,
    cfg,
) -> dict[str, Any]:
    from omegaconf import OmegaConf

    raw = ckpt_io.unwrap(model)

    return {
        "model_state_dict": raw.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "best_val_metric": float(best_metric),
        "val_metric": float(val_metric),
        "model_name": "videomae_ssl",
        "optimizer_name": "adam",
        "config": OmegaConf.to_container(cfg, resolve=True)
        if cfg is not None
        else None,
    }


def _reduce_epoch(
    train_loss: float, val_loss: float, stop_flag: bool, device: torch.device
) -> tuple[float, float, bool]:
    """Average losses + OR the stop flag across ranks in one call.

    Piggybacking the stop bit prevents one rank breaking out of the loop
    while others block on the next collective (a classic DDP deadlock).
    """
    if not (dist.is_available() and dist.is_initialized()):
        return train_loss, val_loss, stop_flag
    tensor = torch.tensor(
        [train_loss, val_loss, float(stop_flag)], dtype=torch.float32, device=device
    )
    all_reduce_mean(tensor)
    reduced_train = float(tensor[0])
    reduced_val = float(tensor[1])
    reduced_stop = float(tensor[2]) > 0.0
    return reduced_train, reduced_val, reduced_stop


def _set_sampler_epoch(dataloader, epoch: int) -> None:
    sampler = getattr(dataloader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def _latest_sibling(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path.with_name(path.stem + "_latest" + path.suffix)


def _install_stop_handlers(stop_flag: list[bool], rank: int):
    """Register SIGTERM/SIGUSR1 → set stop flag.

    SLURM sends SIGTERM on scancel or node failure; `#SBATCH --signal=USR1@120`
    (see slurm/train.sbatch) sends SIGUSR1 two minutes before walltime, giving
    us time to checkpoint cleanly instead of losing an epoch.
    """

    def _handler(sig, _frame):
        stop_flag[0] = True
        if rank == 0:
            print(f"Signal {sig} received — will stop after this epoch.", flush=True)

    return {
        signal.SIGTERM: signal.signal(signal.SIGTERM, _handler),
        signal.SIGUSR1: signal.signal(signal.SIGUSR1, _handler),
    }


def _restore_handlers(prev: dict) -> None:
    for sig, handler in prev.items():
        with contextlib.suppress(Exception):
            signal.signal(sig, handler)


def _write_final_json(
    *,
    cfg,
    results_root: Path,
    model: nn.Module,
    mask_meta: _MaskMeta,
    device: torch.device,
    best_metric: float,
    epochs_trained: int,
) -> None:
    # Measure just the encoder — the pretraining decoder's kwargs mean
    # FlopCounterMode would need a bespoke wrapper otherwise, and the
    # encoder cost is what matters for isoFLOP comparisons.
    raw = ckpt_io.unwrap(model)
    encoder = getattr(raw, "videomae", raw)
    cost = flops_mod.measure_cost(
        encoder, sample_shape=mask_meta.input_shape, device=device
    )
    metrics = {
        "best_val_recon_loss": float(best_metric),
        "metric": "recon_loss",
        "epochs_trained": int(epochs_trained),
    }
    write_result_json(metrics=metrics, cfg=cfg, results_root=results_root, flops=cost)
