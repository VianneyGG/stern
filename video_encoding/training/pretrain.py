"""Hydra entrypoint: VideoMAE self-supervised pretraining on unlabelled video.

Invoked by scripts/run_experiment.sh as:
  `python -m video_encoding.training.pretrain experiment=videomae_ssl_pretrain_voxmini [overrides...]`

Training loop mirrors training/train.py but:
- Consumes unlabelled video (VoxCeleb2 via build_ssl_loader)
- Loss = reconstruction MSE (not CCC)
- Saves checkpoints with is_ssl=True (lower loss = better)
- Produces a frozen encoder checkpoint for Stage 2 (probe training)
"""

from __future__ import annotations

import contextlib
import functools
import os
import time
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from torch.utils.data import DataLoader, DistributedSampler

from video_encoding.data.loaders import build_ssl_loader
from video_encoding.data.ssl_windows import SSLWindowDataset
from video_encoding.models.videomaev2.masking import (
    MotionGuidedTubeMasking,
    RandomTubeMasking,
)
from video_encoding.models.videomaev2.mae import VideoMAEForPreTraining
from video_encoding.training import checkpoint as ckpt_io
from video_encoding.training.ddp.ddp import (
    all_reduce_mean,
    setup_ddp,
    teardown_ddp,
    wrap_ddp,
)
from video_encoding.training.loop_utils import set_sampler_epoch
from video_encoding.training.tb_logger import make_writer, write_result_json


def run_pretrain_stage(
    cfg: DictConfig,
    *,
    rank: int,
    world_size: int,
    local_rank: int,
    device: torch.device,
) -> dict:
    """Main pretraining loop (mirrors training/train.py structure).

    Args:
        cfg: Hydra config with experiment, dataset, train, model fields
        rank: global rank in DDP
        world_size: total ranks
        local_rank: rank on this node
        device: torch device
    """
    # Isolate each sweep cell under its run_tag: every cell shares the same
    # experiment.name, so without this they would clobber / cross-resume from
    # each other's best.pt / latest.pt (and share a TensorBoard dir).
    run_tag = cfg.get("run_tag", None)
    ckpt_name = f"{cfg.experiment.name}/{run_tag}" if run_tag else cfg.experiment.name
    checkpoint_root = Path(cfg.train.checkpoint_root) / ckpt_name
    if rank == 0:
        checkpoint_root.mkdir(parents=True, exist_ok=True)

    # Build SSL loader (unlabelled, grouped by identity)
    ssl_dataset_cfg = build_ssl_loader(cfg.dataset, cfg.preprocess.window)
    dataset = SSLWindowDataset(
        ssl_dataset_cfg,
        num_frames=int(cfg.model.num_frames),
        image_size=int(cfg.model.image_size),
        overlap=float(cfg.preprocess.window.overlap),
    )
    train_loader = _make_ssl_loader(
        dataset,
        batch_size=int(cfg.train.batch_size),
        seed=int(cfg.train.seed),
        rank=rank,
        world_size=world_size,
        num_workers=int(cfg.train.get("num_workers", 4)),
    )

    # Build the RAW (unwrapped) model + optimizer + scheduler and resume BEFORE
    # wrap_ddp. This cluster has no shared filesystem: checkpoints live on
    # node-local /Data, so only the physical node holding rank 0 ever has the
    # file. Resuming after wrap_ddp with every rank reading its own local disk
    # made ranks disagree on start_epoch (rank 0 sees the checkpoint, others
    # silently don't) -- which desyncs the training-loop range across ranks
    # and deadlocks the next DDP collective. Fix: only rank 0 reads the file;
    # wrap_ddp's constructor-time parameter broadcast then propagates the
    # resumed weights to every rank, and _maybe_resume_pretrain explicitly
    # broadcasts (start_epoch, best_loss, optimizer/scheduler state) so all
    # ranks agree before entering the loop.
    model = VideoMAEForPreTraining(cfg.model).to(device)
    amp = str(cfg.train.get("amp", "off")).lower()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=_scaled_lr(cfg, world_size),
        betas=(0.9, 0.95),
        weight_decay=float(cfg.train.get("weight_decay", 0.0)),
    )
    scheduler = _build_scheduler(optimizer, cfg)

    best_path = checkpoint_root / "best.pt"
    latest_path = checkpoint_root / "latest.pt"
    disk_best = ckpt_io.peek_best(best_path) if rank == 0 else None
    start_epoch, best_loss = _maybe_resume_pretrain(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        best_path=best_path,
        latest_path=latest_path,
        resume=bool(cfg.train.get("resume", True)),
        rank=rank,
    )

    model = wrap_ddp(model, cfg, local_rank=local_rank)
    # GradScaler only for fp16 (bf16 has fp32 dynamic range, needs no scaling).
    scaler = (
        torch.amp.GradScaler("cuda")
        if amp == "fp16" and device.type == "cuda"
        else None
    )
    writer = make_writer(checkpoint_root / "tb", rank=rank)
    result: dict = {}

    if bool(cfg.model.get("motion_mask", False)):
        masker = MotionGuidedTubeMasking(mask_ratio=float(cfg.model.mask_ratio))
    else:
        masker = RandomTubeMasking(mask_ratio=float(cfg.model.mask_ratio))

    max_steps = int(cfg.train.get("max_steps", 0))
    accum_steps = int(cfg.train.get("accum_steps", 1))
    local_batch = int(cfg.train.batch_size)
    last_bench: dict = {}

    for epoch in range(start_epoch, int(cfg.train.epochs)):
        set_sampler_epoch(train_loader, epoch)
        stats = _train_epoch_ssl(
            model,
            train_loader,
            optimizer,
            device,
            masker,
            accum_steps=accum_steps,
            grad_clip=float(cfg.train.get("grad_clip", 0.0)),
            max_steps=max_steps,
            amp=amp,
            scaler=scaler,
        )
        train_loss = stats["loss"]

        # Aggregate throughput across ranks — this is a collective, so EVERY
        # rank must reach it (keep it out of the rank==0 guard below).
        per_gpu = torch.tensor([stats["samples_per_s_per_gpu"]], device=device)
        per_gpu = all_reduce_mean(per_gpu)
        samples_per_s_per_gpu = float(per_gpu.item())
        samples_per_s_global = samples_per_s_per_gpu * world_size

        last_bench = {
            "recon_loss": float(train_loss),
            "world_size": world_size,
            "local_batch": local_batch,
            "accum_steps": accum_steps,
            "effective_batch": world_size * local_batch * accum_steps,
            "precision": str(cfg.train.get("amp", "off")),
            "grad_compress": str(cfg.train.get("ddp", {}).get("grad_compress", "")),
            "num_workers": int(cfg.train.get("num_workers", 4)),
            "samples_per_s_global": samples_per_s_global,
            "samples_per_s_per_gpu": samples_per_s_per_gpu,
            "peak_vram_gb": stats["peak_vram_gb"],
            "step_time_ms_mean": stats["step_time_ms_mean"],
            "nan_steps": stats["nan_steps"],
        }

        if writer is not None:
            writer.add_scalar("train/recon_loss", train_loss, epoch)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)
            writer.add_scalar("perf/samples_per_s_global", samples_per_s_global, epoch)
            writer.add_scalar("perf/peak_vram_gb", stats["peak_vram_gb"], epoch)
            # SummaryWriter buffers on a ~120s timer + flushes on close(); a
            # crash (NCCL timeout, walltime kill) before that timer loses the
            # whole epoch. Flush now so every completed epoch is durable.
            writer.flush()

        if rank == 0:
            print(
                f"epoch {epoch + 1}/{cfg.train.epochs}  train_loss={train_loss:.4f}  "
                f"best={best_loss:.4f}  {samples_per_s_global:.1f} samp/s  "
                f"peak={stats['peak_vram_gb']:.1f}GB",
                flush=True,
            )

            improved = train_loss < best_loss
            if improved:
                best_loss = train_loss

            payload = _payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_loss=best_loss,
            )
            if improved:
                _, disk_best = ckpt_io.save_best(
                    best_path,
                    payload,
                    best_loss,
                    is_ssl=True,
                    cached_disk_best=disk_best,
                )
            if bool(cfg.train.get("save_latest", True)):
                ckpt_io.save(latest_path, payload)

            # Emit the current result each epoch (overwrite=True -> one stable
            # file per run_tag) so the aggregator has live data on a run that
            # is walltime-bounded and may never reach cfg.train.epochs.
            write_result_json(
                last_bench,
                cfg,
                results_root=cfg.train.results_root,
                overwrite=True,
            )

        # Deterministic + local on every rank — step outside the rank==0 guard.
        scheduler.step()

    result = last_bench
    if writer is not None:
        writer.close()
    if rank == 0:
        write_result_json(
            result, cfg, results_root=cfg.train.results_root, overwrite=True
        )

    return {"stage": "pretrain", **result}


def _train_epoch_ssl(
    model,
    loader,
    optimizer,
    device,
    masker: "RandomTubeMasking | MotionGuidedTubeMasking",
    *,
    accum_steps: int = 1,
    grad_clip: float = 0.0,
    max_steps: int = 0,
    timing_warmup_steps: int = 10,
    amp: str = "off",
    scaler: "torch.amp.GradScaler | None" = None,
    progress_desc: str | None = None,
) -> dict:
    """Train one epoch (or `max_steps` micro-steps) of SSL pretraining.

    Mirrors training/train.py's `_train_epoch` but applies masking, computes
    reconstruction loss, and measures throughput/VRAM for the scaling sweep.
    Returns a dict
    with `loss`, `samples_per_s_per_gpu`, `step_time_ms_mean`, `peak_vram_gb`,
    `steps`, `samples`.

    Timing notes (this feeds a systems benchmark, so accuracy matters):
    - The loss is accumulated on-device and read once at the end; a per-step
      `.item()` would force a CUDA sync every step and hide compute/comm
      overlap, biasing multi-node throughput downward.
    - Only steps after `timing_warmup_steps` are timed (cuDNN autotune +
      allocator growth make the first steps unrepresentative), with a single
      `cuda.synchronize()` at each end of the timed window.

    `progress_desc`: when set, wraps the loader in a tqdm progress bar with
    this label (used by the mesh round loop to show "round N" progress on an
    interactive terminal). Purely cosmetic -- it only wraps the iterator, so
    every numeric field in the returned dict is identical with or without it.
    Default `None` preserves the exact existing behavior/output for every
    call site that doesn't pass it.
    """
    model.train()
    # Unwrap once. `model` may be bare (world_size=1 / mesh), DDP-wrapped, or
    # torch.compile'd -- ckpt_io.unwrap strips both wrappers, so `core` is
    # always the real module rather than an OptimizedModule we'd be reaching
    # through by attribute proxy.
    core = ckpt_io.unwrap(model)
    token_grid = core.encoder.token_grid  # (T', H', W')
    is_cuda = device.type == "cuda"
    autocast_enabled = amp in ("bf16", "fp16")
    amp_dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    loss_sum = torch.zeros((), device=device)
    count = 0
    optimizer.zero_grad(set_to_none=True)
    try:
        num_steps = len(loader)
    except TypeError:
        num_steps = None
    if max_steps > 0:
        num_steps = max_steps if num_steps is None else min(num_steps, max_steps)

    t_start: float | None = None
    timed_samples = 0
    timed_steps = 0
    executed_steps = 0
    nan_steps = 0

    iterable = loader
    if progress_desc is not None:
        from tqdm import tqdm

        iterable = tqdm(loader, desc=progress_desc, total=num_steps)

    for step, pixel_values in enumerate(iterable):
        if step == timing_warmup_steps:
            if is_cuda:
                torch.cuda.synchronize(device)
            t_start = time.perf_counter()

        pixel_values = pixel_values.to(device)
        batch_size = pixel_values.shape[0]

        # Sample tube mask for this step
        keep_indices, mask_indices = masker(pixel_values, token_grid, device)

        is_boundary = (step + 1) % accum_steps == 0 or (
            num_steps is not None and step + 1 == num_steps
        )
        no_sync_ctx = (
            model.no_sync()
            if not is_boundary and hasattr(model, "no_sync")
            else contextlib.nullcontext()
        )
        with no_sync_ctx:
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled
            ):
                pred, target = model(pixel_values, keep_indices, mask_indices)
                loss = core.compute_loss(pred, target) / accum_steps
            # bf16/fp16 forward on a freshly-initialized deep ViT occasionally
            # produces a non-finite loss on an isolated batch (observed on real
            # data; not reproducible in fp32). Skip backward entirely rather
            # than let it through: backward() would broadcast NaN into every
            # parameter's .grad, permanently corrupting the model for all
            # later steps -- one bad batch would end the whole run. Reported
            # via `nan_steps` so a genuinely unhealthy run stays visible.
            if not torch.isfinite(loss):
                nan_steps += 1
            elif scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if torch.isfinite(loss):
            loss_sum += loss.detach() * (accum_steps * batch_size)
            count += batch_size
        executed_steps += 1
        if t_start is not None:
            timed_samples += batch_size
            timed_steps += 1

        if is_boundary:
            # fp16 path: unscale before clipping (clip operates on true grads),
            # then let the scaler own the step + scale update.
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

        if max_steps > 0 and executed_steps >= max_steps:
            break

    if is_cuda:
        torch.cuda.synchronize(device)
    elapsed = (time.perf_counter() - t_start) if t_start is not None else 0.0

    samples_per_s = timed_samples / elapsed if elapsed > 0 else 0.0
    step_time_ms = (elapsed / timed_steps * 1000.0) if timed_steps > 0 else 0.0
    peak_vram_gb = torch.cuda.max_memory_allocated(device) / 1e9 if is_cuda else 0.0
    # count==0 with nan_steps>0 means every step this epoch was non-finite;
    # report NaN explicitly rather than a misleadingly perfect 0.0/1.
    loss_val = float(loss_sum.item()) / count if count > 0 else float("nan")
    return {
        "loss": loss_val,
        "samples_per_s_per_gpu": samples_per_s,
        "step_time_ms_mean": step_time_ms,
        "peak_vram_gb": peak_vram_gb,
        "steps": executed_steps,
        "samples": count,
        "nan_steps": nan_steps,
    }


def _payload(*, model, optimizer, scheduler, epoch: int, best_loss: float) -> dict:
    payload = ckpt_io.base_payload(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        best_val_metric=best_loss,
        model_name="videomae_for_pretraining",
        optimizer_name="adamw",
    )
    payload["scheduler_state_dict"] = (
        scheduler.state_dict() if scheduler is not None else None
    )
    return payload


def _scaled_lr(cfg, world_size: int, *, local_batch: int | None = None) -> float:
    """Linear-scaling-rule LR: base_lr * effective_batch/256 * lr_mult.

    Falls back to the flat `train.learning_rate` when `base_lr` is unset, so
    throughput-only Phase-1 cells that don't care about convergence still run.

    `local_batch` overrides `cfg.train.batch_size` as the per-step sample
    count fed into `eff_batch`. Every existing (non-mesh) caller omits it, so
    `eff_batch` still reads `cfg.train.batch_size` for them -- behavior is
    unchanged. The mesh round loop passes its real per-step batch
    (`_MESH_BATCH_SIZE`, currently 1) explicitly: `cfg.train.batch_size` (16
    in ssl.yaml) is never the mesh path's actual per-optimizer-step sample
    count, so scaling off it would silently mis-tune the LR for every mesh
    run regardless of `accum_steps`.
    """
    base_lr = cfg.train.get("base_lr", None)
    if base_lr is None:
        return float(cfg.train.learning_rate)
    batch = int(cfg.train.batch_size) if local_batch is None else int(local_batch)
    eff_batch = world_size * batch * int(cfg.train.get("accum_steps", 1))
    return float(base_lr) * eff_batch / 256.0 * float(cfg.train.get("lr_mult", 1.0))


def _build_scheduler(optimizer, cfg):
    """Cosine decay with optional linear warmup (per-epoch stepping)."""
    from torch.optim.lr_scheduler import (
        CosineAnnealingLR,
        LinearLR,
        SequentialLR,
    )

    epochs = max(int(cfg.train.epochs), 1)
    warmup = int(cfg.train.get("warmup_epochs", 0))
    if warmup <= 0:
        return CosineAnnealingLR(optimizer, T_max=epochs)
    warmup_sched = LinearLR(optimizer, start_factor=1e-2, total_iters=warmup)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=max(epochs - warmup, 1))
    return SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[warmup])


def _maybe_resume_pretrain(
    *,
    model,
    optimizer,
    scheduler,
    device: torch.device,
    best_path: Path,
    latest_path: Path,
    resume: bool,
    rank: int,
) -> tuple[int, float]:
    """Resume from checkpoint. Returns (start_epoch, best_loss).

    Only rank 0 reads the checkpoint file: this cluster has no shared
    filesystem, so a node-local checkpoint only exists on whichever physical
    node held rank 0 when it was written -- every other rank's local read
    would silently miss it. Model weights reach the other ranks for free via
    wrap_ddp's constructor-time broadcast (called right after this returns);
    (start_epoch, best_loss, optimizer/scheduler state) are broadcast here
    explicitly so every rank agrees on the training-loop range before it
    starts (a mismatch here desyncs ranks and deadlocks the first DDP
    collective in the loop body).
    """
    if not resume:
        return 0, float("inf")

    if rank == 0:
        source = (
            latest_path
            if latest_path.exists()
            else (best_path if best_path.exists() else None)
        )
        if source is not None:
            start_epoch, best_metric = ckpt_io.load(
                source,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                expected_model_name="videomae_for_pretraining",
                expected_optimizer_name="adamw",
            )
            best_loss = best_metric if best_metric is not None else float("inf")
            print(
                f"resumed from {source} at epoch={start_epoch} best={best_loss}",
                flush=True,
            )
        else:
            start_epoch, best_loss = 0, float("inf")
    else:
        start_epoch, best_loss = 0, float("inf")

    if dist.is_available() and dist.is_initialized():
        # Broadcast the scalar decision plus optimizer/scheduler state
        # (arbitrary Python objects incl. tensors -- pickled under the hood,
        # a one-time cost only paid at resume, not per step).
        payload = [
            (start_epoch, best_loss, optimizer.state_dict(), scheduler.state_dict())
            if rank == 0
            else None
        ]
        dist.broadcast_object_list(payload, src=0)
        start_epoch, best_loss, opt_state, sched_state = payload[0]
        if rank != 0:
            optimizer.load_state_dict(opt_state)
            scheduler.load_state_dict(sched_state)

    return start_epoch, best_loss


def _make_ssl_loader(
    dataset,
    batch_size: int,
    seed: int,
    rank: int,
    world_size: int,
    num_workers: int = 4,
) -> DataLoader:
    """Create a distributed SSL data loader."""
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def _select_stage(
    cfg: DictConfig,
    *,
    rank: int,
    world_size: int,
    local_rank: int,
    device: torch.device,
) -> functools.partial:
    """Return a zero-arg-callable (functools.partial) for whichever training
    stage this process should run.

    Mesh dispatch is gated on the MESH_RUN_ID env var (set by
    run_mesh_worker.sh before this process starts), not a Hydra config
    field: every non-mesh SLURM entrypoint (train.sbatch) must stay
    byte-for-byte unchanged, and an env-var gate keeps this dispatch purely
    additive. Returns a `functools.partial` (not a lambda) so callers/tests
    can inspect `.func` to see which stage was selected without running it.
    """
    if os.environ.get("MESH_RUN_ID"):
        from video_encoding.training.diloco.mesh_round import run_mesh_pretrain_stage

        return functools.partial(run_mesh_pretrain_stage, cfg, device=device)
    return functools.partial(
        run_pretrain_stage,
        cfg,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
    )


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> dict:
    """Hydra entrypoint."""
    rank, world_size, local_rank, device = setup_ddp(cfg)
    try:
        stage = _select_stage(
            cfg, rank=rank, world_size=world_size, local_rank=local_rank, device=device
        )
        return stage()
    finally:
        teardown_ddp()


if __name__ == "__main__":
    main()
