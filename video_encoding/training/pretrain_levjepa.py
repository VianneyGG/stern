"""Hydra entrypoint: LeVJEPA collapse-free self-supervised video pretraining.

Single-node only (no DDP wrapping) per the design's rollout order -- mesh
wiring is a separate, later step in training/diloco/mesh_round.py.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from video_encoding.data.loaders import build_ssl_loader
from video_encoding.data.ssl_windows import SSLWindowDataset
from video_encoding.models.levjepa.encoder import LeVJEPAEncoder
from video_encoding.models.levjepa.losses import SigregOptions, levjepa_loss
from video_encoding.models.levjepa.projector import Projector
from video_encoding.models.levjepa.views import build_views, derive_seed
from video_encoding.training import checkpoint as ckpt_io
from video_encoding.training.loop_utils import WeightEMA
from video_encoding.training.pretrain import _build_scheduler
from video_encoding.training.tb_logger import make_writer, write_result_json


class LeVJEPAViewDataset(Dataset):
    """Wraps a raw-frames dataset (e.g. `SSLWindowDataset(raw_frames=True)`),
    building the global/local views in `__getitem__` -- so DataLoader workers
    do the resize/crop/photometric-jitter work in parallel, instead of it
    running serially on the main process inside the training step.
    """

    def __init__(
        self, raw_dataset: Dataset, *, num_local: int, global_size: int, local_size: int
    ):
        self._raw = raw_dataset
        self._num_local = num_local
        self._global_size = global_size
        self._local_size = local_size

    def __len__(self) -> int:
        return len(self._raw)

    def __getitem__(self, idx: int):
        clip = self._raw[idx]  # (T, H, W, 3) uint8
        return build_views(
            clip,
            num_local=self._num_local,
            global_size=self._global_size,
            local_size=self._local_size,
        )


class LeVJEPAViewIterableDataset(torch.utils.data.IterableDataset):
    """Iterable-dataset counterpart to `LeVJEPAViewDataset`, for wrapping a
    streaming raw-frames source (e.g. `MeshShardIterableDataset(...,
    raw_frames=True)`) that has no `__len__`/`__getitem__` -- only
    `__iter__`. Builds the same `(global_view, local_views, crop_boxes)`
    tuple per clip; DataLoader's default collation batches it the same way.
    """

    def __init__(
        self,
        raw_dataset: torch.utils.data.IterableDataset,
        *,
        num_local: int,
        global_size: int,
        local_size: int,
        view_seed: int | None = None,
    ):
        self._raw = raw_dataset
        self._num_local = num_local
        self._global_size = global_size
        self._local_size = local_size
        # Opt-in (train.rng_split): views come from a private generator
        # re-seeded from (view_seed, round) at every __iter__, so they do not
        # depend on how much global RNG the model consumed, and a relaunched
        # worker resumes a round with that round's views. None = global RNG.
        self._view_seed = view_seed
        self._round = 0

    def set_round(self, round_num: int) -> None:
        """Passthrough to the wrapped dataset, if it supports round-keyed
        shuffling (MeshShardIterableDataset does) -- mesh_round.py's
        train_one_round calls this unconditionally before each round."""
        self._round = round_num
        set_round = getattr(self._raw, "set_round", None)
        if set_round is not None:
            set_round(round_num)

    def __iter__(self):
        generator = None
        if self._view_seed is not None:
            generator = torch.Generator().manual_seed(
                derive_seed(self._view_seed, "views", self._round)
            )
        for clip in self._raw:
            yield build_views(
                clip,
                num_local=self._num_local,
                global_size=self._global_size,
                local_size=self._local_size,
                generator=generator,
            )


def _build_levjepa_loader(cfg: DictConfig, rank: int, world_size: int) -> DataLoader:
    ssl_dataset_cfg = build_ssl_loader(cfg.dataset, cfg.preprocess.window)
    raw_dataset = SSLWindowDataset(
        ssl_dataset_cfg,
        num_frames=int(cfg.model.num_frames),
        image_size=int(cfg.model.image_size),
        overlap=float(cfg.preprocess.window.overlap),
        frame_stride=int(cfg.preprocess.window.get("frame_stride", 1)),
        raw_frames=True,
    )
    dataset = LeVJEPAViewDataset(
        raw_dataset,
        num_local=int(cfg.model.get("num_local_views", 4)),
        global_size=int(cfg.model.image_size),
        local_size=int(cfg.model.get("local_image_size", 96)),
    )
    num_workers = int(cfg.train.get("num_workers", 4))
    return DataLoader(
        dataset,
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=True,  # a 1-clip tail batch would trip _levjepa_step's guard
    )


def _levjepa_step(
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    encoder: LeVJEPAEncoder,
    projector: Projector,
    device: torch.device,
    *,
    sigreg_lambda: float,
    sigreg_directions: int,
    sigreg_opts: SigregOptions | None = None,
    sigreg_key: tuple[int, ...] = (),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run one LeVJEPA forward pass over a batch already split into views.

    `batch` is `(global_batch, local_batch, crop_boxes)` as produced by
    `LeVJEPAViewDataset`/`build_views`: `global_batch` is `(B, 3, T, H, W)`,
    `local_batch` is `(B, V, 3, T, h, w)`, `crop_boxes` is `(B, V, 4)`.
    """
    z_global, z_locals = _levjepa_embed(batch, encoder, projector, device)
    total, components = levjepa_loss(
        z_global,
        z_locals,
        lam=sigreg_lambda,
        num_directions=sigreg_directions,
        opts=sigreg_opts,
        key=sigreg_key,
    )
    components["z_global_norm"] = z_global.norm(dim=-1).mean().detach()
    components["z_locals_norm"] = z_locals.norm(dim=-1).mean().detach()
    return total, components


def _levjepa_embed(
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    encoder: LeVJEPAEncoder,
    projector: Projector,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The forward half of `_levjepa_step`: (z_global (B, K), z_locals (B, V, K))."""
    global_batch, local_batch, crop_boxes = batch
    global_batch = global_batch.to(device)
    local_batch = local_batch.to(device)
    crop_boxes = crop_boxes.to(device)
    B, V = local_batch.shape[0], local_batch.shape[1]
    if B < 2:
        # SIGReg is a statistic over the batch (minimised at z=0 for a single
        # sample) and the projector's BatchNorm would only see one clip's views.
        raise ValueError(
            f"LeVJEPA needs at least 2 clips per batch, got {B}; set "
            "mesh.micro_batch_size / train.batch_size accordingly"
        )
    local_flat = local_batch.reshape(B * V, *local_batch.shape[2:])
    crop_boxes_flat = crop_boxes.reshape(B * V, 4)

    cls_global = encoder.forward_cls(global_batch)  # (B, D)
    cls_locals_flat = encoder.forward_cls(
        local_flat, crop_box=crop_boxes_flat
    )  # (B*V, D)

    # Project both branches through one shared BatchNorm forward: computing
    # global/local batch statistics separately (two projector() calls) would
    # inject a systematic global<->local offset that fights the invariance
    # objective the projector output feeds into.
    all_cls = torch.cat([cls_global, cls_locals_flat], dim=0)
    all_z = projector(all_cls)
    z_global, z_locals_flat = all_z.split([B, B * V], dim=0)
    z_locals = z_locals_flat.reshape(B, V, -1)
    return z_global, z_locals


def _payload(
    encoder: LeVJEPAEncoder,
    projector: Projector,
    optimizer: torch.optim.Optimizer,
    ema: WeightEMA,
    epoch: int,
    total_loss: float,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> dict:
    payload = ckpt_io.base_payload(
        model=encoder,
        optimizer=optimizer,
        epoch=epoch,
        best_val_metric=total_loss,
        model_name="levjepa_encoder",
        optimizer_name="adamw",
    )
    payload["projector_state_dict"] = projector.state_dict()
    payload["ema_state_dict"] = ema.state_dict()
    payload["scheduler_state_dict"] = (
        scheduler.state_dict() if scheduler is not None else None
    )
    return payload


def run_levjepa_pretrain_stage(
    cfg: DictConfig,
    *,
    rank: int,
    world_size: int,
    local_rank: int,
    device: torch.device,
) -> dict:
    """LeVJEPA pretraining loop. Mirrors `pretrain.run_pretrain_stage`'s
    signature and return-dict shape."""
    run_tag = cfg.get("run_tag", None)
    ckpt_name = f"{cfg.experiment.name}/{run_tag}" if run_tag else cfg.experiment.name
    checkpoint_root = Path(cfg.train.checkpoint_root) / ckpt_name
    if rank == 0:
        checkpoint_root.mkdir(parents=True, exist_ok=True)
    latest_path = checkpoint_root / "latest.pt"
    best_path = checkpoint_root / "best.pt"

    train_loader = _build_levjepa_loader(cfg, rank, world_size)

    encoder = LeVJEPAEncoder(cfg.model).to(device)
    projector = Projector(
        embed_dim=int(cfg.model.embed_dim),
        hidden_dim=int(cfg.model.get("projector_hidden", 2048)),
        output_dim=int(cfg.model.get("embed_dim_k", 256)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(projector.parameters()),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.get("weight_decay", 0.0)),
    )
    ema = WeightEMA(encoder, decay=float(cfg.model.get("polyak_decay", 0.9999)))
    polyak_update_every = int(cfg.model.get("polyak_update_every", 32))
    scheduler = _build_scheduler(optimizer, cfg)

    start_epoch = 0
    disk_best = ckpt_io.peek_best(best_path) if rank == 0 else None
    if bool(cfg.train.get("resume", True)) and latest_path.exists():
        start_epoch, _ = ckpt_io.load(
            latest_path,
            model=encoder,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            expected_model_name="levjepa_encoder",
            expected_optimizer_name="adamw",
        )
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        if "projector_state_dict" in ckpt:
            projector.load_state_dict(ckpt["projector_state_dict"])
        if "ema_state_dict" in ckpt:
            ema.shadow = {k: v.to(device) for k, v in ckpt["ema_state_dict"].items()}

    writer = make_writer(checkpoint_root / "tb", rank=rank)
    last_result: dict = {}
    global_step = start_epoch  # coarse but monotonic across resumes

    encoder.train()
    projector.train()
    for epoch in range(start_epoch, int(cfg.train.epochs)):
        loss_sums: dict[str, float] = {}
        num_batches = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            total_loss, components = _levjepa_step(
                batch,
                encoder,
                projector,
                device,
                sigreg_lambda=float(cfg.model.get("sigreg_lambda", 0.02)),
                sigreg_directions=int(cfg.model.get("sigreg_directions", 1024)),
            )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(projector.parameters()),
                max_norm=float(cfg.train.get("grad_clip_norm", 1.0)),
            )
            optimizer.step()

            global_step += 1
            if global_step % polyak_update_every == 0:
                ema.update(encoder)

            num_batches += 1
            loss_sums["total_loss"] = loss_sums.get("total_loss", 0.0) + float(
                total_loss.item()
            )
            for key, value in components.items():
                loss_sums[key] = loss_sums.get(key, 0.0) + float(value.item())

        # Epoch-mean, not the last batch's value -- a single-batch value
        # would make loss curves single-sample noise.
        last_result = {
            "total_loss": loss_sums.get("total_loss", 0.0) / max(num_batches, 1),
            "invariance_loss": loss_sums.get("invariance", 0.0) / max(num_batches, 1),
            "sigreg_loss": loss_sums.get("sigreg", 0.0) / max(num_batches, 1),
        }
        z_global_norm = loss_sums.get("z_global_norm", 0.0) / max(num_batches, 1)
        z_locals_norm = loss_sums.get("z_locals_norm", 0.0) / max(num_batches, 1)

        if writer is not None:
            writer.add_scalar("train/total_loss", last_result["total_loss"], epoch)
            writer.add_scalar("train/sigreg_loss", last_result["sigreg_loss"], epoch)
            writer.add_scalar("train/z_global_norm", z_global_norm, epoch)
            writer.add_scalar("train/z_locals_norm", z_locals_norm, epoch)
            writer.flush()

        # Step before the checkpoint save (not after, unlike pretrain.py's
        # ordering): the saved scheduler snapshot must be the one the next
        # (resumed) epoch should use. Stepping after save leaves the
        # checkpoint holding the LR this epoch already consumed, so a
        # resumed run would silently redo one step of warmup/decay instead
        # of continuing where training left off. This reordering doesn't
        # change the LR trajectory used for actual training in an
        # uninterrupted run -- that's driven by the live optimizer object,
        # not by when we snapshot it for checkpointing.
        scheduler.step()

        if rank == 0:
            payload = _payload(
                encoder,
                projector,
                optimizer,
                ema,
                epoch,
                last_result["total_loss"],
                scheduler=scheduler,
            )
            ckpt_io.save(latest_path, payload)
            _, disk_best = ckpt_io.save_best(
                best_path,
                payload,
                last_result["total_loss"],
                is_ssl=True,
                cached_disk_best=disk_best,
            )

    if rank == 0:
        results_root = cfg.train.get("results_root", checkpoint_root)
        write_result_json(last_result, cfg, results_root=results_root, overwrite=True)

    if writer is not None:
        writer.close()

    return {"stage": "pretrain_levjepa", **last_result}


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> dict:
    """Hydra entrypoint. Single-node only."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return run_levjepa_pretrain_stage(
        cfg, rank=0, world_size=1, local_rank=0, device=device
    )


if __name__ == "__main__":
    main()
