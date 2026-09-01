"""Hydra entrypoint: LeVJEPA collapse-free self-supervised video pretraining.

Single-node only (no DDP wrapping) per the design's rollout order -- mesh
wiring is a separate, later step in training/diloco/mesh_round.py.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from video_encoding.data.loaders import build_ssl_loader
from video_encoding.data.ssl_windows import SSLWindowDataset
from video_encoding.models.levjepa.encoder import LeVJEPAEncoder
from video_encoding.models.levjepa.losses import levjepa_loss
from video_encoding.models.levjepa.projector import Projector
from video_encoding.models.levjepa.views import build_views
from video_encoding.training import checkpoint as ckpt_io
from video_encoding.training.tb_logger import make_writer, write_result_json


def _build_levjepa_loader(cfg: DictConfig, rank: int, world_size: int) -> DataLoader:
    ssl_dataset_cfg = build_ssl_loader(cfg.dataset, cfg.preprocess.window)
    dataset = SSLWindowDataset(
        ssl_dataset_cfg,
        num_frames=int(cfg.model.num_frames),
        image_size=int(cfg.model.image_size),
        overlap=float(cfg.preprocess.window.overlap),
        frame_stride=int(cfg.preprocess.window.get("frame_stride", 1)),
    )
    return DataLoader(
        dataset,
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        num_workers=int(cfg.train.get("num_workers", 4)),
    )


def _levjepa_step(
    batch: torch.Tensor,
    encoder: LeVJEPAEncoder,
    projector: Projector,
    device: torch.device,
    *,
    num_local_views: int,
    local_image_size: int,
    sigreg_lambda: float,
    sigreg_directions: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run one LeVJEPA forward pass over a batch of raw decoded clips.

    `batch` is `(B, T, H, W, 3)` uint8 -- build_views's expected per-clip
    layout, batched.
    """
    global_views = []
    local_views_list = []
    for clip in batch:
        global_view, local_views = build_views(
            clip, num_local=num_local_views, local_size=local_image_size
        )
        global_views.append(global_view)
        local_views_list.append(local_views)

    global_batch = torch.stack(global_views, dim=0).to(device)  # (B, 3, T, H, W)
    local_batch = torch.stack(local_views_list, dim=0).to(device)  # (B, V, 3, T, h, w)
    B, V = local_batch.shape[0], local_batch.shape[1]
    local_flat = local_batch.reshape(B * V, *local_batch.shape[2:])

    z_global = projector(encoder.forward_cls(global_batch))  # (B, K)
    z_locals_flat = projector(encoder.forward_cls(local_flat))  # (B*V, K)
    z_locals = z_locals_flat.reshape(B, V, -1)

    return levjepa_loss(
        z_global, z_locals, lam=sigreg_lambda, num_directions=sigreg_directions
    )


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

    writer = make_writer(checkpoint_root / "tb", rank=rank)
    last_result: dict = {}

    encoder.train()
    projector.train()
    for epoch in range(int(cfg.train.epochs)):
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            total_loss, components = _levjepa_step(
                batch,
                encoder,
                projector,
                device,
                num_local_views=int(cfg.model.get("num_local_views", 4)),
                local_image_size=int(cfg.model.get("local_image_size", 96)),
                sigreg_lambda=float(cfg.model.get("sigreg_lambda", 0.02)),
                sigreg_directions=int(cfg.model.get("sigreg_directions", 1024)),
            )
            total_loss.backward()
            optimizer.step()

            last_result = {
                "total_loss": float(total_loss.item()),
                "invariance_loss": float(components["invariance"].item()),
                "sigreg_loss": float(components["sigreg"].item()),
            }

        if writer is not None:
            writer.add_scalar("train/total_loss", last_result["total_loss"], epoch)
            writer.add_scalar("train/sigreg_loss", last_result["sigreg_loss"], epoch)
            writer.flush()

    if rank == 0:
        payload = ckpt_io.base_payload(
            model=encoder,
            optimizer=optimizer,
            epoch=int(cfg.train.epochs) - 1,
            best_val_metric=last_result.get("total_loss", float("nan")),
            model_name="levjepa_encoder",
            optimizer_name="adamw",
        )
        ckpt_io.save(checkpoint_root / "latest.pt", payload)

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
