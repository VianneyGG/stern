"""Hydra entrypoint: end-to-end training of a from-scratch VideoMAE + VA head.

Invoked by scripts/run_experiment.sh as
`python -m video_encoding.training.train experiment=<name> [overrides...]`.

Decode fixed 16-frame windows, feed them through a `VideoMAEEncoder` + a 2-D
VA regression head, and optimise CCC end-to-end. No LoRA, no cached
embeddings (those "recipes" were deleted -- they came from a V-JEPA/MARLIN-
oriented spec that does not fit training VideoMAE itself). Optionally loads
a Stage 1 SSL-pretrained encoder (`train.pretrain_encoder_path`) and
optionally freezes it (`train.freeze_backbone`) for a frozen-encoder probe,
per `docs/superpowers/specs/2026-07-15-videomae-config-model-upgrade-design.md`.

CI tests inject a lightweight fake backbone (see tests/test_train_stage.py)
because a full ViT-B forward per step is slow; the real encoder has its own
CPU-sized unit test in tests/test_videomae_encoder.py.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.distributed import DistributedSampler

from eval.ccc import ccc_loss, concordance_correlation_coefficient
from video_encoding.data.loaders import build_probe_loader
from video_encoding.data.probe_windows import VAWindowDataset
from video_encoding.models import VideoMAEVARegressor
from video_encoding.training import checkpoint as ckpt_io
from video_encoding.training.ddp import setup_ddp, teardown_ddp, wrap_ddp
from video_encoding.training.loop_utils import set_sampler_epoch
from video_encoding.training.tb_logger import make_writer, write_result_json


def run_train_stage(
    cfg: DictConfig,
    *,
    rank: int,
    world_size: int,
    local_rank: int,
    device: torch.device,
) -> dict:
    checkpoint_root = Path(cfg.train.checkpoint_root) / cfg.experiment.name
    if rank == 0:
        checkpoint_root.mkdir(parents=True, exist_ok=True)

    labelled = build_probe_loader(cfg.dataset, cfg.preprocess.window)
    dataset = VAWindowDataset(
        labelled,
        num_frames=int(cfg.model.num_frames),
        image_size=int(cfg.model.image_size),
        overlap=float(cfg.preprocess.window.overlap),
        va_range=tuple(cfg.dataset.va_range),
    )
    train_loader, val_loader = _split_loaders(
        dataset,
        val_frac=0.1,
        batch_size=int(cfg.train.batch_size),
        seed=int(cfg.train.seed),
        rank=rank,
        world_size=world_size,
    )

    model = VideoMAEVARegressor(cfg.model).to(device)

    pretrain_encoder_path = cfg.train.get("pretrain_encoder_path", None)
    if pretrain_encoder_path:
        ckpt = ckpt_io.load_checkpoint_file(pretrain_encoder_path)
        state = ckpt.get("model_state_dict", ckpt)
        encoder_state = {
            k[len("encoder.") :]: v
            for k, v in state.items()
            if k.startswith("encoder.")
        }
        missing, unexpected = ckpt_io.load_encoder_state(model.backbone, encoder_state)
        if rank == 0:
            print(
                f"loaded pretrained encoder from {pretrain_encoder_path} "
                f"(missing={len(missing)}, unexpected={len(unexpected)})",
                flush=True,
            )
    if bool(cfg.train.get("freeze_backbone", False)):
        for p in model.backbone.parameters():
            p.requires_grad = False

    model = wrap_ddp(model, cfg, local_rank=local_rank)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=float(cfg.train.learning_rate))
    writer = make_writer(checkpoint_root / "tb", rank=rank)

    best_path = checkpoint_root / "best.pt"
    latest_path = checkpoint_root / "latest.pt"
    disk_best = ckpt_io.peek_best(best_path)
    start_epoch, best_ccc = _maybe_resume(
        model=model,
        optimizer=optimizer,
        device=device,
        best_path=best_path,
        latest_path=latest_path,
        resume=bool(cfg.train.get("resume", True)),
        rank=rank,
    )
    result: dict = {}
    for epoch in range(start_epoch, int(cfg.train.epochs)):
        set_sampler_epoch(train_loader, epoch)
        train_loss = _train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            accum_steps=int(cfg.train.get("accum_steps", 1)),
            grad_clip=float(cfg.train.get("grad_clip", 0.0)),
        )
        val = _validate(model, val_loader, device) if val_loader else {}
        mean_ccc = val.get("mean_ccc", -train_loss)  # fall back to -loss if no val

        if writer is not None:
            writer.add_scalar("train/loss", train_loss, epoch)
            for k, v in val.items():
                writer.add_scalar(f"val/{k}", v, epoch)
        if rank == 0:
            print(
                f"epoch {epoch + 1}/{cfg.train.epochs}  train_loss={train_loss:.4f}  "
                f"val_ccc={mean_ccc:.4f}  best={best_ccc:.4f}",
                flush=True,
            )

        if rank == 0:
            improved = mean_ccc > best_ccc
            if improved:
                best_ccc = mean_ccc
                result = val
            payload = _payload(
                model=model, optimizer=optimizer, epoch=epoch, best_ccc=best_ccc
            )
            if improved:
                _, disk_best = ckpt_io.save_best(
                    best_path,
                    payload,
                    best_ccc,
                    is_ssl=False,
                    cached_disk_best=disk_best,
                )
            if bool(cfg.train.get("save_latest", True)):
                ckpt_io.save(latest_path, payload)

            # Emit the current result each epoch (overwrite=True -> one stable
            # file per run_tag), mirroring pretrain.py's heartbeat -- gives the
            # SLURM training watcher's stall detector a live mtime signal for
            # probe-stage runs too, not just SSL pretraining.
            write_result_json(
                val or result,
                cfg,
                results_root=cfg.train.results_root,
                overwrite=True,
            )

    if writer is not None:
        writer.close()
    if rank == 0:
        write_result_json(
            result, cfg, results_root=cfg.train.results_root, overwrite=True
        )
    return {"stage": "train", **result}


def _payload(*, model, optimizer, epoch: int, best_ccc: float) -> dict:
    return ckpt_io.base_payload(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        best_val_metric=best_ccc,
        model_name="videomae_va_regressor",
        optimizer_name="adam",
    )


def _maybe_resume(
    *,
    model,
    optimizer,
    device: torch.device,
    best_path: Path,
    latest_path: Path,
    resume: bool,
    rank: int,
) -> tuple[int, float]:
    """Resume from `latest_path` (up to 1 epoch stale) if present, else
    `best_path`. Returns `(start_epoch, best_ccc)`; `(0, -inf)` on a fresh
    start or when `resume=False`."""
    if not resume:
        return 0, -float("inf")
    source = (
        latest_path
        if latest_path.exists()
        else (best_path if best_path.exists() else None)
    )
    if source is None:
        return 0, -float("inf")
    start_epoch, best_metric = ckpt_io.load(
        source,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        device=device,
        expected_model_name="videomae_va_regressor",
        expected_optimizer_name="adam",
    )
    if rank == 0:
        print(
            f"resumed from {source} at epoch={start_epoch} best={best_metric}",
            flush=True,
        )
    return start_epoch, best_metric if best_metric is not None else -float("inf")


def _train_epoch(
    model, loader, optimizer, device, *, accum_steps: int = 1, grad_clip: float = 0.0
) -> float:
    """Trains one epoch with `accum_steps` micro-batches per optimizer step.

    Also used (in spirit) by training/pretrain.py's `_train_epoch_ssl`:
    `no_sync()` skips DDP's allreduce on non-boundary micro-steps, and
    `grad_clip` (when >0) clips the accumulated gradient right before
    `optimizer.step()`.
    """
    model.train()
    total, count = 0.0, 0
    optimizer.zero_grad(set_to_none=True)
    try:
        num_steps = len(loader)
    except TypeError:
        num_steps = None

    for step, (pixel_values, labels) in enumerate(loader):
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)
        batch_size = pixel_values.shape[0]

        is_boundary = (step + 1) % accum_steps == 0 or (
            num_steps is not None and step + 1 == num_steps
        )
        no_sync_ctx = (
            model.no_sync()
            if not is_boundary and hasattr(model, "no_sync")
            else contextlib.nullcontext()
        )
        with no_sync_ctx:
            pred = model(pixel_values)
            loss = ccc_loss(pred, labels) / accum_steps
            loss.backward()

        total += float(loss.item()) * accum_steps * batch_size
        count += batch_size

        if is_boundary:
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    return total / max(count, 1)


@torch.no_grad()
def _validate(model, loader, device) -> dict:
    model.eval()
    preds, targets = [], []
    for pixel_values, labels in loader:
        pred = model(pixel_values.to(device)).cpu().numpy()
        preds.append(pred)
        targets.append(labels.numpy())
    pred = np.concatenate(preds)
    target = np.concatenate(targets)
    ccc_v = concordance_correlation_coefficient(pred[:, 0], target[:, 0])
    ccc_a = concordance_correlation_coefficient(pred[:, 1], target[:, 1])
    return {"val_ccc_v": ccc_v, "val_ccc_a": ccc_a, "mean_ccc": (ccc_v + ccc_a) / 2}


def _split_loaders(
    dataset: Dataset,
    *,
    val_frac: float,
    batch_size: int,
    seed: int,
    rank: int,
    world_size: int,
) -> tuple[DataLoader, DataLoader | None]:
    """Deterministic 90/10 split; DistributedSampler on the train side when
    multi-node. `drop_last=True` on train because `ccc_loss` needs >1 sample
    per batch to estimate variance."""
    n_total = len(dataset)
    n_val = max(1, int(round(n_total * val_frac))) if n_total >= 10 else 0
    gen = torch.Generator().manual_seed(seed)
    if n_val > 0:
        train_ds, val_ds = random_split(
            dataset, [n_total - n_val, n_val], generator=gen
        )
    else:
        train_ds, val_ds = dataset, None

    sampler = (
        DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        if world_size > 1
        else None
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=True,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        if val_ds is not None
        else None
    )
    return train_loader, val_loader


def run_training(cfg: DictConfig) -> dict:
    torch.manual_seed(int(cfg.train.seed))
    rank, world_size, local_rank, device = setup_ddp(cfg)
    try:
        return run_train_stage(
            cfg, rank=rank, world_size=world_size, local_rank=local_rank, device=device
        )
    finally:
        teardown_ddp()


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> dict:
    return run_training(cfg)


if __name__ == "__main__":
    main()
