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
import math
from dataclasses import replace
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.distributed import DistributedSampler

from eval.ccc import ccc_loss, concordance_correlation_coefficient
from video_encoding.data.loaders import (
    LabelledDataset,
    build_probe_loader,
    identity_disjoint_split,
)
from video_encoding.data.probe_windows import VAWindowDataset, make_temporal_transform
from video_encoding.models import VideoMAEVARegressor
from video_encoding.models.levjepa.encoder import LeVJEPAEncoder
from video_encoding.models.levjepa_hf import HFLeVJEPAVARegressor
from video_encoding.training import checkpoint as ckpt_io
from video_encoding.training.ddp.ddp import setup_ddp, teardown_ddp, wrap_ddp
from video_encoding.training.loop_utils import set_sampler_epoch
from video_encoding.training.tb_logger import make_writer, write_result_json


def _temporal_transform(cfg: DictConfig):
    """Build the probe's temporal-condition transform from config.

    Defaults to "normal" (no transform), so an existing probe config is
    unchanged.
    """
    return make_temporal_transform(
        str(cfg.train.get("temporal_condition", "normal")), int(cfg.train.seed)
    )


def _extract_encoder_state(state: dict, *, encoder_kind: str) -> dict:
    """Isolate the encoder's own state-dict keys from a Stage-1 checkpoint.

    VideoMAE's checkpoint wraps `VideoMAEForPreTraining`, whose `encoder`
    submodule keys are saved under an `"encoder."` prefix -- strip it.
    LeVJEPA has two checkpoint shapes: the single-node path
    (`pretrain_levjepa.py`'s `_payload` -> `checkpoint.base_payload(model=
    encoder, ...)`) saves the encoder directly with no prefix, but the
    DiLoCo mesh path (`mesh_round.py`'s `_LeVJEPAPretrainModule`) wraps
    encoder+projector as submodules, so its checkpoint keys are
    `"encoder."`/`"projector."`-prefixed. Detect which shape this is by
    whether any key already starts with `"encoder."` -- stripping
    unconditionally would match zero keys and silently load nothing for the
    single-node shape.
    """
    if encoder_kind == "levjepa":
        if any(k.startswith("encoder.") for k in state):
            return {
                k[len("encoder.") :]: v
                for k, v in state.items()
                if k.startswith("encoder.")
            }
        return dict(state)
    return {
        k[len("encoder.") :]: v for k, v in state.items() if k.startswith("encoder.")
    }


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
    loader_kwargs = dict(
        batch_size=int(cfg.train.batch_size),
        rank=rank,
        world_size=world_size,
        num_workers=int(cfg.train.get("num_workers", 0)),
    )
    if str(cfg.train.get("split", "random")) == "identity":
        # Identity-disjoint and seed-independent, so held-out windows share no
        # clip or actor with training and seed-paired runs score on one set.
        fit_ids, scored_ids = identity_disjoint_split(
            labelled.records, float(cfg.train.get("split_fraction", 0.2))
        )
        train_loader, val_loader = _make_loaders(
            _window_dataset(cfg, labelled, fit_ids),
            _window_dataset(cfg, labelled, scored_ids),
            **loader_kwargs,
        )
    else:
        train_loader, val_loader = _split_loaders(
            _window_dataset(cfg, labelled, None),
            val_frac=0.1,
            seed=int(cfg.train.seed),
            **loader_kwargs,
        )

    encoder_kind = str(cfg.model.get("encoder", "videomae"))
    if encoder_kind == "levjepa_hf":
        model = HFLeVJEPAVARegressor(cfg.model).to(device)
    elif encoder_kind == "levjepa":
        # LeVJEPAEncoder deliberately has no patch_embed/pos_embed attrs, so
        # VideoMAEVARegressor._backbone_tokens's hasattr check routes to its
        # self.backbone(pixel_values=...) fallback -- a drop-in Stage-2
        # backbone with zero changes to the regressor itself.
        model = VideoMAEVARegressor(cfg.model, backbone=LeVJEPAEncoder(cfg.model)).to(
            device
        )
    else:
        model = VideoMAEVARegressor(cfg.model).to(device)
    model_name = {
        "levjepa_hf": "levjepa_hf_va_regressor",
        "levjepa": "levjepa_va_regressor",
    }.get(encoder_kind, "videomae_va_regressor")

    pretrain_encoder_path = cfg.train.get("pretrain_encoder_path", None)
    if pretrain_encoder_path:
        ckpt = ckpt_io.load_checkpoint_file(pretrain_encoder_path)
        state = ckpt.get("model_state_dict", ckpt)
        encoder_state = _extract_encoder_state(state, encoder_kind=encoder_kind)
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
    optimizer, optimizer_name = _build_optimizer(model, cfg.train)
    scheduler = _build_scheduler(
        optimizer, cfg.train, steps_per_epoch=len(train_loader)
    )
    amp_bf16 = bool(cfg.train.get("amp_bf16", False))
    writer = make_writer(checkpoint_root / "tb", rank=rank)

    best_path = checkpoint_root / "best.pt"
    latest_path = checkpoint_root / "latest.pt"
    disk_best = ckpt_io.peek_best(best_path)
    start_epoch, best_ccc = _maybe_resume(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        model_name=model_name,
        optimizer_name=optimizer_name,
        device=device,
        best_path=best_path,
        latest_path=latest_path,
        resume=bool(cfg.train.get("resume", True)),
        rank=rank,
    )
    result: dict = {}
    final: dict = {}
    epochs_run = 0
    for epoch in range(start_epoch, int(cfg.train.epochs)):
        epochs_run += 1
        set_sampler_epoch(train_loader, epoch)
        train_loss = _train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            accum_steps=int(cfg.train.get("accum_steps", 1)),
            grad_clip=float(cfg.train.get("grad_clip", 0.0)),
            scheduler=scheduler,
            amp_bf16=amp_bf16,
        )
        val = (
            _validate(model, val_loader, device, amp_bf16=amp_bf16)
            if val_loader
            else {}
        )
        final = val
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
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_ccc=best_ccc,
                model_name=model_name,
                optimizer_name=optimizer_name,
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
    return {"stage": "train", **result, "final": final, "epochs_run": epochs_run}


def _build_optimizer(model, train_cfg) -> tuple[torch.optim.Optimizer, str]:
    """`adam` (the default, unchanged) or `adamw` with the `head.*` params at
    `learning_rate * head_lr_mult` -- a fine-tuned pretrained backbone wants a
    far smaller step than a freshly initialised head."""
    name = str(train_cfg.get("optimizer", "adam"))
    lr = float(train_cfg.learning_rate)
    named = [
        (n, p) for n, p in ckpt_io.unwrap(model).named_parameters() if p.requires_grad
    ]
    if name == "adam":
        return torch.optim.Adam([p for _, p in named], lr=lr), name
    if name != "adamw":
        raise ValueError(f"train.optimizer must be adam or adamw, got {name!r}")
    head_lr = lr * float(train_cfg.get("head_lr_mult", 1.0))
    groups = [
        {"params": [p for n, p in named if not n.startswith("head.")], "lr": lr},
        {"params": [p for n, p in named if n.startswith("head.")], "lr": head_lr},
    ]
    groups = [g for g in groups if g["params"]]
    wd = float(train_cfg.get("weight_decay", 0.0))
    return torch.optim.AdamW(groups, weight_decay=wd), name


def _build_scheduler(optimizer, train_cfg, *, steps_per_epoch: int):
    """Per-optimizer-step linear warmup then cosine to 0, or None when
    `warmup_epochs` is unset (the default: constant LR, unchanged)."""
    warmup_epochs = float(train_cfg.get("warmup_epochs", 0))
    if warmup_epochs <= 0:
        return None
    steps = steps_per_epoch // int(train_cfg.get("accum_steps", 1))
    total = max(1, steps * int(train_cfg.epochs))
    warmup = max(1, int(steps * warmup_epochs))

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _payload(
    *,
    model,
    optimizer,
    epoch: int,
    best_ccc: float,
    scheduler=None,
    model_name: str = "videomae_va_regressor",
    optimizer_name: str = "adam",
) -> dict:
    payload = ckpt_io.base_payload(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        best_val_metric=best_ccc,
        model_name=model_name,
        optimizer_name=optimizer_name,
    )
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    return payload


def _maybe_resume(
    *,
    model,
    optimizer,
    scheduler,
    model_name: str,
    optimizer_name: str,
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
        scheduler=scheduler,
        device=device,
        expected_model_name=model_name,
        expected_optimizer_name=optimizer_name,
    )
    if rank == 0:
        print(
            f"resumed from {source} at epoch={start_epoch} best={best_metric}",
            flush=True,
        )
    return start_epoch, best_metric if best_metric is not None else -float("inf")


def _train_epoch(
    model,
    loader,
    optimizer,
    device,
    *,
    accum_steps: int = 1,
    grad_clip: float = 0.0,
    scheduler=None,
    amp_bf16: bool = False,
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
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp_bf16):
                pred = model(pixel_values)
            loss = ccc_loss(pred.float(), labels) / accum_steps
            loss.backward()

        total += float(loss.item()) * accum_steps * batch_size
        count += batch_size

        if is_boundary:
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

    return total / max(count, 1)


@torch.no_grad()
def _validate(model, loader, device, *, amp_bf16: bool = False) -> dict:
    model.eval()
    preds, targets = [], []
    for pixel_values, labels in loader:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp_bf16):
            pred = model(pixel_values.to(device))
        pred = pred.float().cpu().numpy()
        preds.append(pred)
        targets.append(labels.numpy())
    pred = np.concatenate(preds)
    target = np.concatenate(targets)
    ccc_v = concordance_correlation_coefficient(pred[:, 0], target[:, 0])
    ccc_a = concordance_correlation_coefficient(pred[:, 1], target[:, 1])
    return {"val_ccc_v": ccc_v, "val_ccc_a": ccc_a, "mean_ccc": (ccc_v + ccc_a) / 2}


def _window_dataset(
    cfg: DictConfig, labelled: LabelledDataset, identity_ids: set[str] | None
) -> VAWindowDataset:
    """Windows over `labelled`, restricted to `identity_ids` when given."""
    if identity_ids is not None:
        labelled = replace(
            labelled,
            records=[r for r in labelled.records if r.identity_id in identity_ids],
        )
    return VAWindowDataset(
        labelled,
        num_frames=int(cfg.model.num_frames),
        image_size=int(cfg.model.image_size),
        overlap=float(cfg.preprocess.window.overlap),
        va_range=tuple(cfg.dataset.va_range),
        frame_stride=int(cfg.model.get("frame_stride", 1)),
        transform=_temporal_transform(cfg),
    )


def _split_loaders(
    dataset: Dataset,
    *,
    val_frac: float,
    batch_size: int,
    seed: int,
    rank: int,
    world_size: int,
    num_workers: int = 0,
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
    return _make_loaders(
        train_ds,
        val_ds,
        batch_size=batch_size,
        rank=rank,
        world_size=world_size,
        num_workers=num_workers,
    )


def _make_loaders(
    train_ds: Dataset,
    val_ds: Dataset | None,
    *,
    batch_size: int,
    rank: int,
    world_size: int,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader | None]:
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
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
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
