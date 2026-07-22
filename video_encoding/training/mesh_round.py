"""Worker-side round loop for decentralized mesh SSL pretraining.

Entry point invoked from `pretrain.py`'s `main()` when `MESH_RUN_ID` is set in
the environment (set by `run_mesh_worker.sh`, not this module). Every mesh
job is `--nodes=1 --ntasks=1`: this module must NEVER call
`torch.distributed.init_process_group` or otherwise assume DDP is active.

One `run_mesh_pretrain_stage` call covers the worker's ENTIRE lifetime for
this run -- it loops over every round itself (not resubmitted per round via
SLURM), so `round_num` ranges over `[0, cfg.mesh.rounds)` inside one process.
Each round:

  1. If `cfg.mesh.merge` and `round_num > 0`, pull the freshly-merged
     `master.pt` into the local model (the outer-optimizer merge itself runs
     on the watcher side -- Task 3 -- between this worker's rounds, never
     here).
  2. Train `H = cfg.mesh.steps_per_round` local steps over this node's
     assigned shards (`train_one_round`).
  3. Evaluate a fixed ~2000-window val subsample (`evaluate_val_loss`).
  4. Save a local checkpoint, then push it to the shared `pending/` dir and
     (if merging) block until the watcher's sync flag appears
     (`push_checkpoint_and_wait`).
"""

from __future__ import annotations

import logging
import os
import socket
import sys
import time
from pathlib import Path

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from video_encoding.data.mesh_shard_dataset import MeshShardIterableDataset
from video_encoding.mesh.assignment import read_my_shards
from video_encoding.mesh.shard_map import load_shard_map
from video_encoding.models.videomaev2.mae import VideoMAEForPreTraining
from video_encoding.models.videomaev2.masking import (
    MotionGuidedTubeMasking,
    RandomTubeMasking,
)
from video_encoding.scripts.seed_mesh import NODE_SUBNETS
from video_encoding.training import checkpoint as ckpt_io
from video_encoding.training.compile_utils import maybe_compile
from video_encoding.training.pretrain import _scaled_lr, _train_epoch_ssl
from video_encoding.training.tb_logger import make_writer

logger = logging.getLogger(__name__)

# Fixed poll cadence for the sync-flag wait -- see push_checkpoint_and_wait.
# Not busy-looping: a 30-minute window at 5s intervals is ~360 checks, cheap
# for both this process and the shared filesystem.
_SYNC_POLL_INTERVAL_S = 5.0

# Every window this dataset yields already carries no batch dimension
# (MeshShardIterableDataset.__iter__ does `yield pixel_values.squeeze(0)`),
# so a DataLoader batch_size of 1 is the natural unit here. Effective batch
# size is built via `accum_steps` (gradient accumulation) instead of a wider
# micro-batch -- these are memory-constrained single-GPU fish nodes, and
# accum_steps trades wall-clock for peak VRAM without changing the interface
# train_one_round/evaluate_val_loss were specified with.
_MESH_BATCH_SIZE = 1


def _resolve_subnet(hostname: str, cfg: DictConfig) -> str:
    """This node's real subnet, for mesh/shard_map.py's same-subnet peer
    preference (cross-switch uplink caps at ~0.94 Gbps total vs. a
    non-blocking intra-switch backplane -- shard_map.py's docstring).

    NODE_SUBNETS (seed_mesh.py) is the same hostname->subnet mapping the
    rest of the mesh design is keyed on (fish=/254, birds=/253, cars=/252,
    countries=/254, orchids=/253) -- a single yaml-wide constant would
    misreport every birds/cars/orchids node as fish's "/254", silently
    defeating the same-subnet preference for more than half the roster.
    `cfg.mesh.subnet` is only a fallback for hosts NODE_SUBNETS doesn't know
    about (e.g. a manual test run off the real roster).
    """
    return NODE_SUBNETS.get(hostname, str(cfg.mesh.get("subnet", "/254")))


def run_mesh_pretrain_stage(cfg: DictConfig, *, device: torch.device) -> dict:
    """Entry point called from pretrain.py's main() when MESH_RUN_ID is set.

    Reads MESH_RUN_ID/MESH_SHARED/MESH_ASSIGNMENTS/MESH_SHARD_MAP/MESH_ROOT/
    STERN_CHECKPOINT_ROOT from os.environ (all guaranteed present --
    run_mesh_worker.sh already fails fast if any is missing before this
    process starts). Single-GPU, no DDP, rank is always 0.
    """
    run_id = os.environ["MESH_RUN_ID"]
    mesh_shared = Path(os.environ["MESH_SHARED"])
    assignments_path = Path(os.environ["MESH_ASSIGNMENTS"])
    shard_map_path = Path(os.environ["MESH_SHARD_MAP"])
    mesh_root = Path(os.environ["MESH_ROOT"])
    checkpoint_root = Path(os.environ["STERN_CHECKPOINT_ROOT"])
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    node = socket.gethostname()
    subnet = _resolve_subnet(node, cfg)

    rounds = int(cfg.mesh.get("rounds", 10))
    steps_per_round = int(cfg.mesh.get("steps_per_round", 200))
    merge = bool(cfg.mesh.get("merge", True))
    outer_lr = float(cfg.mesh.get("outer_lr", 0.7))
    outer_momentum = float(cfg.mesh.get("outer_momentum", 0.9))
    sync_timeout_s = float(cfg.mesh.get("sync_timeout_s", 1800.0))

    num_frames = int(cfg.model.num_frames)
    image_size = int(cfg.model.image_size)
    overlap = float(cfg.preprocess.window.overlap)

    shard_map = load_shard_map(shard_map_path)
    train_shard_names = read_my_shards(assignments_path, node)
    train_dataset = MeshShardIterableDataset(
        train_shard_names,
        mesh_root=mesh_root,
        shard_map=shard_map,
        subnet=subnet,
        num_frames=num_frames,
        image_size=image_size,
        overlap=overlap,
    )

    # Val shards are replicated in full to every node under mesh_root/val/
    # (seed_mesh.py's _VAL_SUBDIR) -- always locally resolvable, so an empty
    # shard_map is fine here (resolve_shard finds them as seeded copies).
    val_root = mesh_root / "val"
    val_shard_names = (
        sorted(p.name for p in val_root.glob("*.tar")) if val_root.is_dir() else []
    )
    val_dataset = MeshShardIterableDataset(
        val_shard_names,
        mesh_root=val_root,
        shard_map={},
        subnet=subnet,
        num_frames=num_frames,
        image_size=image_size,
        overlap=overlap,
    )

    model = VideoMAEForPreTraining(cfg.model).to(device)
    # No DDP on this path (single-GPU worker, see module docstring), so
    # compile happens directly on the raw model -- no wrap_ddp ordering to
    # worry about. Compiled once here; surviving every later master.pt
    # reload since load_state_dict mutates parameters in place.
    model = maybe_compile(model, cfg)
    amp = str(cfg.train.get("amp", "off")).lower()
    accum_steps = int(cfg.train.get("accum_steps", 1))
    grad_clip = float(cfg.train.get("grad_clip", 0.0))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        # local_batch=_MESH_BATCH_SIZE: the mesh path's real per-optimizer-step
        # sample count is _MESH_BATCH_SIZE * accum_steps, NOT cfg.train.batch_size
        # (16 in ssl.yaml, never overridden here) -- _scaled_lr's linear-scaling
        # rule must scale off the batch actually seen per step, or the LR is
        # silently mistuned for every mesh run regardless of accum_steps.
        lr=_scaled_lr(cfg, world_size=1, local_batch=_MESH_BATCH_SIZE),
        betas=(0.9, 0.95),
        weight_decay=float(cfg.train.get("weight_decay", 0.0)),
    )
    if bool(cfg.model.get("motion_mask", False)):
        masker = MotionGuidedTubeMasking(mask_ratio=float(cfg.model.mask_ratio))
    else:
        masker = RandomTubeMasking(mask_ratio=float(cfg.model.mask_ratio))

    master_path = mesh_shared / "checkpoints" / run_id / "master.pt"
    writer = make_writer(checkpoint_root / "tb", rank=0)

    last_train_loss = float("nan")
    final_round = -1

    for round_num in range(0, rounds):
        if round_num > 0 and merge:
            if master_path.exists():
                master_payload = torch.load(
                    master_path, map_location=device, weights_only=False
                )
                load_result = ckpt_io.unwrap(model).load_state_dict(
                    master_payload["model_state_dict"], strict=False
                )
                if load_result.missing_keys or load_result.unexpected_keys:
                    # strict=False silently skips mismatched keys rather than
                    # raising -- without this check a key-name regression (e.g.
                    # a stray _orig_mod. prefix) would leave this worker
                    # training on stale weights for the rest of the run with
                    # no signal in the round bookkeeping.
                    logger.error(
                        "run_mesh_pretrain_stage: master state_dict key "
                        "mismatch at round %d (run_id=%s) -- missing=%s "
                        "unexpected=%s",
                        round_num,
                        run_id,
                        load_result.missing_keys,
                        load_result.unexpected_keys,
                    )
            else:
                # Expected only if the watcher's merge hasn't landed yet for
                # some reason -- push_checkpoint_and_wait already blocked
                # until the sync flag appeared, so this should be rare.
                # Warn and keep training on this node's own weights rather
                # than crashing a multi-day run (mirrors run_outer_step.py's
                # missing-momentum-buffer handling).
                logger.warning(
                    "run_mesh_pretrain_stage: merge=True but no master "
                    "checkpoint at %s (run_id=%s round=%d) -- continuing "
                    "with this node's own weights",
                    master_path,
                    run_id,
                    round_num,
                )

        train_stats = train_one_round(
            model,
            optimizer,
            device,
            masker,
            train_dataset,
            steps_per_round=steps_per_round,
            amp=amp,
            accum_steps=accum_steps,
            grad_clip=grad_clip,
            round_num=round_num,
        )
        last_train_loss = train_stats["loss"]

        val_loss = evaluate_val_loss(model, val_dataset, device, masker, amp)

        if writer is not None:
            writer.add_scalar("round/recon_loss", train_stats["loss"], round_num)
            writer.add_scalar("round/val_loss", val_loss, round_num)
            writer.add_scalar(
                "round/samples_per_s", train_stats["samples_per_s_per_gpu"], round_num
            )
            writer.flush()

        payload = ckpt_io.base_payload(
            model=model,
            optimizer=optimizer,
            epoch=round_num,
            best_val_metric=val_loss,
            model_name="videomae_for_pretraining",
            optimizer_name="adamw",
        )
        payload["n_samples"] = train_stats["samples"]
        payload["val_loss"] = val_loss
        payload["round"] = round_num
        payload["samples_per_s"] = train_stats["samples_per_s_per_gpu"]
        payload["outer_lr"] = outer_lr
        payload["outer_momentum"] = outer_momentum

        local_path = checkpoint_root / f"{run_id}.pt"
        ckpt_io.save(local_path, payload)

        push_checkpoint_and_wait(
            payload,
            mesh_shared=mesh_shared,
            run_id=run_id,
            node=node,
            round_num=round_num,
            merge=merge,
            sync_timeout_s=sync_timeout_s,
        )

        final_round = round_num
        print(
            f"mesh round {round_num + 1}/{rounds}  train_loss={train_stats['loss']:.4f}  "
            f"val_loss={val_loss:.4f}  {train_stats['samples_per_s_per_gpu']:.1f} samp/s",
            flush=True,
        )

    if writer is not None:
        writer.close()

    return {
        "stage": "mesh_pretrain",
        "final_round": final_round,
        "recon_loss": float(last_train_loss),
    }


def train_one_round(
    model,
    optimizer,
    device,
    masker,
    dataset: MeshShardIterableDataset,
    *,
    steps_per_round: int,
    amp: str,
    accum_steps: int,
    grad_clip: float,
    round_num: int,
) -> dict:
    """Thin wrapper around _train_epoch_ssl for one mesh round.

    Builds a DataLoader over `dataset` (num_workers=0 -- MeshShardIterableDataset's
    own prefetch thread is the concurrency; a second DataLoader worker layer
    would race it for the same cache dir), calls _train_epoch_ssl with
    max_steps=steps_per_round and progress_desc=f"round {round_num}", and
    returns its stats dict unchanged.
    """
    loader = DataLoader(dataset, batch_size=_MESH_BATCH_SIZE, num_workers=0)
    return _train_epoch_ssl(
        model,
        loader,
        optimizer,
        device,
        masker,
        accum_steps=accum_steps,
        grad_clip=grad_clip,
        max_steps=steps_per_round,
        amp=amp,
        progress_desc=f"round {round_num}",
    )


def evaluate_val_loss(
    model,
    val_dataset: MeshShardIterableDataset,
    device,
    masker,
    amp: str,
    *,
    max_windows: int = 2000,
) -> float:
    """No-grad pass over up to max_windows windows from val_dataset.

    Returns mean reconstruction loss. Per the design spec, per-round val
    runs on a fixed ~2000-window subsample, not the full 7GB val set.
    Leaves `model.training is True` on return (matches train_one_round's
    expectation for the next round), regardless of whether this function
    raises.
    """
    model.eval()
    core = ckpt_io.unwrap(model)
    token_grid = core.encoder.token_grid
    autocast_enabled = amp in ("bf16", "fp16")
    amp_dtype = torch.bfloat16 if amp == "bf16" else torch.float16

    loader = DataLoader(val_dataset, batch_size=_MESH_BATCH_SIZE, num_workers=0)
    loss_sum = torch.zeros((), device=device)
    finite_count = 0
    windows_seen = 0
    try:
        with torch.no_grad():
            for pixel_values in loader:
                if windows_seen >= max_windows:
                    break
                windows_seen += 1

                pixel_values = pixel_values.to(device)
                keep_indices, mask_indices = masker(pixel_values, token_grid, device)
                with torch.autocast(
                    device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled
                ):
                    pred, target = model(pixel_values, keep_indices, mask_indices)
                    loss = core.compute_loss(pred, target)
                if torch.isfinite(loss):
                    loss_sum += loss.detach()
                    finite_count += 1
    finally:
        model.train()

    return float(loss_sum.item()) / finite_count if finite_count > 0 else float("nan")


def push_checkpoint_and_wait(
    payload: dict,
    *,
    mesh_shared: Path,
    run_id: str,
    node: str,
    round_num: int,
    merge: bool,
    sync_timeout_s: float = 1800.0,
) -> None:
    """Push `payload` to the shared pending/ dir, then (if merging) wait.

    Writes to mesh_shared/checkpoints/pending/{run_id}_{node}.pt. If `merge`
    is False (the ab-single_node reference arm), returns immediately -- no
    wait, the caller proceeds to the next round locally. If `merge` is True,
    polls for mesh_shared/checkpoints/flags/sync_complete_{run_id}_{round_num}.flag
    every 5s up to sync_timeout_s; on timeout, logs an ERROR naming
    run_id/round_num/node and calls sys.exit(1) -- NOT escalate.py, which
    operates on watcher-process state, not a worker job's own state. The
    existing, unmodified dead/stalled-job detectors catch this exit through
    the normal cron path.
    """
    mesh_shared = Path(mesh_shared)
    pending_path = mesh_shared / "checkpoints" / "pending" / f"{run_id}_{node}.pt"
    pending_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_io.save(pending_path, payload)

    if not merge:
        return

    flag_path = (
        mesh_shared
        / "checkpoints"
        / "flags"
        / f"sync_complete_{run_id}_{round_num}.flag"
    )
    start = time.monotonic()
    while not flag_path.exists():
        elapsed = time.monotonic() - start
        if elapsed >= sync_timeout_s:
            logger.error(
                "push_checkpoint_and_wait: timed out after %.0fs waiting for %s "
                "(run_id=%s round_num=%d node=%s)",
                sync_timeout_s,
                flag_path,
                run_id,
                round_num,
                node,
            )
            sys.exit(1)
        time.sleep(_SYNC_POLL_INTERVAL_S)
