"""Hydra entrypoint for all experiments, invoked by
scripts/run_experiment.sh as `python -m video_encoding.probe.train experiment=<name> [overrides...]`.

Not unit-tested: wires real encoder weights + real data end-to-end, the same
category as VideoMaeV2Encoder/ssl_adapt (see those modules' docstrings) --
smoke-tested on the cluster via a short real run, not a mocked unit test.

DDP contract: `run_training` owns the process-group lifecycle for the whole
job. Both stages (SSL now, probe once unblocked) inherit the same
(rank, world_size, local_rank, device) tuple, so the probe stage gets DDP
for free the day its label-loading is wired up.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.distributed import DistributedSampler


import numpy as np

from cache.store import clip_key, read_embeddings, write_embeddings
from data.labels import load_clip_va_label
from video_encoding.data.loaders import build_probe_loader, build_ssl_loader
from video_encoding.encoders.videomae import VideoMaeV2Encoder
from video_encoding.probe import train_probe
from video_encoding.ssl import ssl_adapt
from video_encoding.training import checkpoint as ckpt_io
from video_encoding.training.ddp import setup_ddp, teardown_ddp
from video_encoding.training.logger import write_result_json


class _SSLClipDataset(Dataset):
    """Wraps an SSLDataset's manifest records as a flat, decode-on-access
    torch Dataset of processed pixel_values tensors."""

    def __init__(self, ssl_dataset, num_frames: int, processor):
        self._records = [
            (identity_id, record.clip_path)
            for identity_id, records in ssl_dataset.records_by_identity.items()
            for record in records
        ]
        self._dataset = ssl_dataset
        self._num_frames = num_frames
        self._processor = processor

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> torch.Tensor:
        identity_id, clip_path = self._records[idx]
        frames = self._dataset.load_frames(
            identity_id, clip_path, num_frames=self._num_frames
        )
        return self._processor(frames, return_tensors="pt")["pixel_values"][0]


def run_ssl_stage(
    cfg: DictConfig,
    *,
    rank: int,
    world_size: int,
    local_rank: int,
    device: torch.device,
) -> dict:
    """LoRA + masked-reconstruction adaptation on an unlabelled (role: ssl)
    dataset, e.g. talkingcelebs_mini. Used by experiments with
    ``use_ssl: true`` (e.g. videomae_lora_ssl_voxmini). Saves the adapted
    encoder for a later, separate probe stage once labelled data exists."""
    checkpoint_root = Path(cfg.train.checkpoint_root) / cfg.experiment.name
    if rank == 0:
        checkpoint_root.mkdir(parents=True, exist_ok=True)

    ssl_dataset = build_ssl_loader(cfg.dataset, cfg.preprocess.window)
    probe_encoder = VideoMaeV2Encoder(cfg.model)  # only used for its processor
    clip_dataset = _SSLClipDataset(
        ssl_dataset, num_frames=cfg.model.num_frames, processor=probe_encoder.processor
    )

    train_loader, val_loader = _split_loaders(
        clip_dataset,
        val_frac=0.1,
        batch_size=int(cfg.train.batch_size),
        seed=int(cfg.train.seed),
        rank=rank,
        world_size=world_size,
    )

    adapted_model = ssl_adapt(
        cfg.model.checkpoint,
        train_loader,
        val_dataloader=val_loader,
        lora_rank=int(cfg.train.lora_rank),
        mask_ratio=float(cfg.train.ssl.mask_ratio),
        epochs=int(cfg.train.ssl.epochs),
        lr=float(cfg.train.ssl.learning_rate),
        accum_steps=int(cfg.train.get("accum_steps", 1)),
        use_amp=bool(cfg.train.get("use_amp", False)),
        grad_clip=float(cfg.train.get("grad_clip", 0.0)),
        warmup_epochs=int(cfg.train.get("warmup_epochs", 0)),
        checkpoint_path=checkpoint_root / "ssl_best.pt",
        resume=bool(cfg.train.get("resume", True)),
        save_latest=bool(cfg.train.get("save_latest", True)),
        results_root=cfg.train.results_root,
        log_dir=checkpoint_root / "tb_ssl",
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
        cfg=cfg,
        seed=int(cfg.train.seed),
    )
    if rank == 0:
        raw = ckpt_io.unwrap(adapted_model)
        torch.save(raw.videomae.state_dict(), checkpoint_root / "adapted_encoder.pt")
    return {"stage": "ssl"}


def run_probe_stage(
    cfg: DictConfig,
    *,
    rank: int,
    world_size: int,
    local_rank: int,
    device: torch.device,
) -> dict:
    """Frozen / LoRA-supervised experiments, or the post-SSL tail of a
    LoRA+SSL experiment ("freeze LoRA + run cache+probe" per the approved
    spec): cache embeddings off the frozen encoder, then train the VA head
    against real labels.

    Runs only on rank 0 -- caching embeddings and training a linear head is
    cheap (Recipe A is "embarrassingly parallel, no NCCL needed" per the
    cluster tier-1 findings), so sharding the caching step across ranks is
    left as a follow-up rather than adding DDP complexity a linear head
    doesn't need. Other ranks return a stage-only dict.
    """
    checkpoint_root = Path(cfg.train.checkpoint_root) / cfg.experiment.name
    if rank != 0:
        return {"stage": "probe"}
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    probe_dataset = build_probe_loader(cfg.dataset, cfg.preprocess.window)
    encoder = VideoMaeV2Encoder(cfg.model)
    cache_path = checkpoint_root / "embeddings.h5"
    va_range = tuple(cfg.dataset.va_range)
    annotation_hz = float(cfg.dataset.annotation_hz)

    embeddings: list[np.ndarray] = []
    labels: list[tuple[float, float]] = []
    for record in probe_dataset.records:
        key = clip_key(record.clip_path)
        try:
            embedding = read_embeddings(cache_path, key)
        except (FileNotFoundError, KeyError):
            frames = probe_dataset.load_frames(
                record.clip_path, num_frames=cfg.model.num_frames
            )
            embedding = encoder.encode_clip(frames)
            write_embeddings(cache_path, key, embedding)
        embeddings.append(embedding)
        labels.append(
            load_clip_va_label(
                probe_dataset.root / record.label_path, annotation_hz, va_range
            )
        )

    embeddings_arr = np.stack(embeddings).astype(np.float32)
    labels_arr = np.array(labels, dtype=np.float32)
    train_x, train_y, val_x, val_y = _split_probe_arrays(
        embeddings_arr, labels_arr, val_frac=0.1, seed=int(cfg.train.seed)
    )

    result = train_probe(
        train_x,
        train_y,
        val_x,
        val_y,
        embed_dim=int(cfg.model.embed_dim),
        epochs=int(cfg.train.probe.epochs),
        lr=float(cfg.train.probe.learning_rate),
        log_dir=checkpoint_root / "tb_probe",
    )
    head = result.pop("head")
    torch.save(head.state_dict(), checkpoint_root / "probe_head.pt")
    write_result_json(result, cfg, results_root=cfg.train.results_root)
    return {"stage": "probe", **result}


def _split_probe_arrays(
    embeddings: np.ndarray, labels: np.ndarray, *, val_frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic train/val split at the array level -- train_probe takes
    embeddings/labels directly, not a DataLoader, so this mirrors
    _split_loaders' 90/10 determinism without needing a Dataset wrapper.

    Always keeps at least 1 clip on each side (unlike _split_loaders, which
    allows an empty val split below 10 clips) because train_probe computes
    CCC over the val split unconditionally and CCC is undefined on an empty
    array.
    """
    n_total = embeddings.shape[0]
    if n_total < 2:
        raise ValueError(
            f"need at least 2 clips to split into train/val, got {n_total}"
        )
    n_val = min(max(1, int(round(n_total * val_frac))), n_total - 1)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    return (
        embeddings[train_idx],
        labels[train_idx],
        embeddings[val_idx],
        labels[val_idx],
    )


def run_training(cfg: DictConfig) -> dict:
    torch.manual_seed(cfg.train.seed)
    rank, world_size, local_rank, device = setup_ddp(cfg)
    try:
        stage_kwargs = dict(
            rank=rank, world_size=world_size, local_rank=local_rank, device=device
        )
        if cfg.experiment.use_ssl:
            return run_ssl_stage(cfg, **stage_kwargs)
        return run_probe_stage(cfg, **stage_kwargs)
    finally:
        teardown_ddp()


def _split_loaders(
    dataset: Dataset,
    *,
    val_frac: float,
    batch_size: int,
    seed: int,
    rank: int,
    world_size: int,
) -> tuple[DataLoader, DataLoader | None]:
    """Deterministic 90/10 train/val split with a DistributedSampler on the
    train side when running multi-node.

    Val side uses the default (non-distributed) sampler because we run val
    on every rank and mean-reduce the loss — DistributedSampler would drop
    a fraction of val samples on each rank without adding meaningful
    throughput.
    """
    n_total = len(dataset)
    n_val = max(1, int(round(n_total * val_frac))) if n_total >= 10 else 0
    n_train = n_total - n_val
    gen = torch.Generator().manual_seed(seed)
    if n_val > 0:
        train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=gen)
    else:
        train_ds, val_ds = dataset, None

    train_sampler = (
        DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        if world_size > 1
        else None
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
        if val_ds is not None
        else None
    )
    return train_loader, val_loader


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> dict:
    return run_training(cfg)


if __name__ == "__main__":
    main()
