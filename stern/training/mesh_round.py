"""Worker-side round loop for decentralized mesh SSL pretraining.

Worker-side round loop for mesh SSL pretraining. `pretrain.py`'s `main()`
calls into this module when `MESH_RUN_ID` is set (`run_mesh_worker.sh` sets it).

Every mesh job runs as one task on one node, so this module must never call
`torch.distributed.init_process_group` or assume DDP is active.

One `run_mesh_pretrain_stage` call covers the worker's whole lifetime. It loops
over every round itself, so `round_num` runs over `[0, cfg.mesh.rounds)` in one
process. Each round:

  1. If `cfg.mesh.merge` and `round_num > 0`, load the freshly merged
     `master.pt` into the local model. The merge runs on the watcher side,
     never here.
  2. Train local steps over this node's shards (`train_one_round`). The step
     count is `cfg.mesh.steps_per_round`, unless `cfg.mesh.round_budget_s` is
     set, in which case it is whatever fits the budget at this node's measured
     rate (`mesh/pacing.py`).
  3. Evaluate a fixed ~2000-window validation subsample (`evaluate_val_loss`).
  4. Save a local checkpoint, push it to the shared `pending/` dir, and, when
     merging, wait for the watcher's sync flag (`push_checkpoint_and_wait`).
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import socket
import sys
import time
from collections.abc import Callable
from pathlib import Path

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from stern.data.mesh_shard_dataset import MeshShardIterableDataset
from stern.mesh.assignment import read_my_shards
from stern.mesh.shard_map import load_shard_map
from stern.models.levjepa.encoder import LeVJEPAEncoder
from stern.models.levjepa.losses import (
    SigregOptions,
    _no_tf32,
    invariance_loss,
    sample_directions,
    sigreg_from_sums,
    sigreg_sums,
    view_directions,
)
from stern.models.levjepa.projector import Projector
from stern.models.levjepa.views import derive_seed
from stern.models.videomae.mae import VideoMAEForPreTraining
from stern.models.videomae.masking import build_masker
from stern.mesh.topology import NODE_SUBNETS
from stern.training import checkpoint as ckpt_io
from stern.training.compile_utils import maybe_compile
from stern.training.loop_utils import WeightEMA
from stern.mesh.outer_step import pseudo_gradient, to_fp32_state_dict
from stern.mesh.pacing import steps_for_budget, update_rate_ema
from stern.mesh.pending_files import pending_filename
from stern.mesh.quantize import quantize_with_error_feedback
from stern.training.pretrain import _scaled_lr, _train_epoch_ssl
from stern.training.pretrain_levjepa import (
    LeVJEPAViewIterableDataset,
    _levjepa_embed,
    _levjepa_step,
)

logger = logging.getLogger(__name__)

# Poll interval for the sync-flag wait (see push_checkpoint_and_wait).
# A 30-minute wait at 5 s is about 360 checks, which is cheap on NFS.
_SYNC_POLL_INTERVAL_S = 5.0

# Each window this dataset yields already has no batch dimension
# (MeshShardIterableDataset.__iter__ squeezes it), so a DataLoader batch size
# of 1 is the natural unit. The effective batch comes from `accum_steps`
# instead, which trades wall-clock for peak VRAM on these single-GPU nodes.
_MESH_BATCH_SIZE = 1
# LeVJEPA validation batch: SIGReg is a batch statistic, so scoring one clip
# at a time would make val loss meaningless.
_LEVJEPA_VAL_BATCH_SIZE = 16

# Keys dropped from the round payload before it is pushed to the shared NFS
# pending/ dir. The merge reads only model_state_dict, n_samples and val_loss.
# The optimizer buffers roughly double the payload, and the shared copy is
# unlinked unread. The local checkpoint keeps these keys.
_SHARED_PAYLOAD_DROP_KEYS = ("optimizer_state_dict", "optimizer_name")


def _fp16_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Cast floating-point tensors to fp16 and leave other buffers alone.

    Halves the per-round NFS write in push_checkpoint_and_wait. The merge
    upcasts back to fp32 before doing any arithmetic.
    """
    return {
        k: (v.half() if torch.is_floating_point(v) else v)
        for k, v in state_dict.items()
    }


# Three consecutive abandoned rounds mean a persistently slow node, not a
# transient one: it stays in the roster and contributes nothing.
_ABANDON_ALERT_STREAK = 3


def _append_round_metrics(
    mesh_shared: Path,
    run_id: str,
    node: str,
    round_1_based: int,
    train_loss: float,
    val_loss: float,
    samp_s: float,
    n_shards: int,
    lr: float,
    *,
    n_samples: int,
    n_steps: int,
    abandoned: bool = False,
    consecutive_abandons: int = 0,
    val_loss_ema: float | None = None,
) -> None:
    """Append one JSONL record to mesh_shared/metrics/<run_id>/<node>.jsonl.

    Per-round metrics otherwise live only in each job's stdout. A failed write
    is logged and swallowed, since metrics must never abort a round.
    """
    record = {
        "run_id": run_id,
        "node": node,
        "round": round_1_based,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "samp_s": samp_s,
        "n_shards": n_shards,
        "lr": lr,
        # Exact counts. samp_s x round budget would be wrong for a node that is
        # starved for data.
        "n_samples": n_samples,
        "n_steps": n_steps,
        # An abandoned round still writes a record, so a node that stopped
        # contributing shows up in the stream instead of as a silent gap.
        "abandoned": abandoned,
        "consecutive_abandons": consecutive_abandons,
        "val_loss_ema": val_loss_ema,
        "ts": time.time(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    metrics_path = mesh_shared / "metrics" / run_id / f"{node}.jsonl"
    try:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except OSError as exc:
        logger.warning(
            "_append_round_metrics: failed to write %s (run_id=%s round=%d) "
            "-- continuing without it: %s",
            metrics_path,
            run_id,
            round_1_based,
            exc,
        )


def _resolve_subnet(hostname: str, cfg: DictConfig) -> str:
    """Return this node's subnet, for the same-subnet peer preference in mesh/shard_map.py.

    The cross-switch uplink is much slower than the intra-switch backplane, so
    peers on the same subnet are preferred. NODE_SUBNETS (mesh/topology.py)
    maps each hostname to its subnet. `cfg.mesh.subnet` is only a fallback for
    hosts it does not know, such as a manual test run.
    """
    return NODE_SUBNETS.get(hostname, str(cfg.mesh.get("subnet", "/254")))


def _require_local_shards(
    shard_names: list[str], mesh_root: Path, *, run_id: str, node: str
) -> None:
    """Exit non-zero if this node was assigned a shard it does not hold.

    Assignment is static, so this fires only when the assignment file and the
    disk disagree. A worker that cannot train on its shards must fail at
    startup. Otherwise it spends the round at zero samples per second, and a
    skipped-shard warning looks like a success.
    """
    absent = [name for name in shard_names if not (mesh_root / name).exists()]
    if not absent:
        return
    raise SystemExit(
        f"mesh_round: node {node!r} (run {run_id!r}) was assigned "
        f"{len(absent)} of {len(shard_names)} shard(s) it does not hold: "
        + ", ".join(absent[:10])
        + (" ..." if len(absent) > 10 else "")
        + f" -- {mesh_root} is out of sync with the shard assignment. "
        "Re-run python -m stern.mesh.validate_seeding; do not restart this worker."
    )


def _write_current_round_marker(
    mesh_shared: Path, run_id: str, node: str, round_num: int
) -> None:
    marker = mesh_shared / "state" / "requeue" / f"{run_id}-{node}-current_round"
    marker.parent.mkdir(parents=True, exist_ok=True)
    tmp = marker.with_suffix(".tmp")
    tmp.write_text(str(round_num))
    tmp.replace(marker)


def mesh_round_lr(
    peak_lr: float,
    round_num: int,
    total_rounds: int,
    *,
    warmup_frac: float = 0.05,
    min_lr: float = 1e-5,
) -> float:
    """Cosine decay with linear warmup, evaluated at round granularity.

    One value per round, constant across that round's H inner steps -- mirrors
    pretrain.py's `_build_scheduler` (per-epoch stepping) but is a pure
    function of round_num rather than persisted optimizer/scheduler state, so
    a worker relaunched mid-run (chunk_rounds) reconstructs the exact same LR
    for round_num without needing to have lived through the earlier rounds.
    """
    warmup_rounds = max(1, round(warmup_frac * total_rounds))
    if round_num < warmup_rounds:
        return peak_lr * (round_num + 1) / warmup_rounds
    progress = (round_num - warmup_rounds) / max(1, total_rounds - warmup_rounds)
    progress = min(1.0, progress)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))


def round_already_merged(mesh_shared: Path, run_id: str, round_num: int) -> bool:
    """True once the coordinator has merged `round_num` -- with or without us.

    The sync flag is written by _do_syncing the moment a merge completes, and
    a merge completes as soon as `quorum_k` nodes have checked in. So a node
    that sees its OWN round's flag while still training is a node the merge
    went ahead without: every remaining inner step is computed against a
    master that is already superseded, and its checkpoint would be rejected
    by the merge's master_version guard even if it arrived.

    Naming the flag by round is what makes this safe to poll mid-round: a
    flag from an earlier round can never match the round being trained.
    """
    flag = (
        Path(mesh_shared)
        / "checkpoints"
        / "flags"
        / f"sync_complete_{run_id}_{round_num}.flag"
    )
    return flag.exists()


def chunk_stop_round(rounds: int, start_round: int, chunk_rounds: int | None) -> int:
    """Return the last round (exclusive) this worker job runs.

    `chunk_rounds=None` runs the whole run in one job. Otherwise the watcher
    relaunches successive chunks, each resuming from `master.pt` via
    `mesh.start_round`. Exiting cleanly before the walltime cap matters: a
    walltime kill takes every worker down mid-round at once, so all of them
    miss the same round.

    Clamped to `rounds`, and never below `start_round`, so a non-positive
    chunk_rounds exits cleanly rather than skipping rounds.
    """
    if chunk_rounds is None:
        return rounds
    return max(start_round, min(rounds, start_round + int(chunk_rounds)))


class _LeVJEPAPretrainModule(torch.nn.Module):
    """Wraps `LeVJEPAEncoder` + `Projector` as a single `nn.Module` so
    `train_one_round`'s `model.parameters()`/`state_dict()`/DiLoCo-outer-step
    calls keep working unmodified -- those all operate on whatever `model`
    is, without caring that it's a VideoMAEForPreTraining instance."""

    def __init__(self, model_cfg: DictConfig):
        super().__init__()
        self.encoder = LeVJEPAEncoder(model_cfg)
        self.projector = Projector(
            embed_dim=int(model_cfg.embed_dim),
            hidden_dim=int(model_cfg.get("projector_hidden", 2048)),
            output_dim=int(model_cfg.get("embed_dim_k", 256)),
        )


def _sigreg_options(cfg: DictConfig) -> SigregOptions:
    """model.sigreg_seed / sigreg_orthogonal / sigreg_moment_weight and
    train.sigreg_pool; all absent -> the original SIGReg estimator."""
    seed = cfg.model.get("sigreg_seed", None)
    return SigregOptions(
        seed=None if seed is None else int(seed),
        orthogonal=bool(cfg.model.get("sigreg_orthogonal", False)),
        moment_weight=float(cfg.model.get("sigreg_moment_weight", 0.0)),
        pool=bool(cfg.train.get("sigreg_pool", False)),
    )


def _maybe_compile_levjepa_encoder(model: torch.nn.Module, cfg: DictConfig) -> None:
    """train.levjepa_compile: compile the LeVJEPA encoder's forward_cls in place.

    train.compile wraps the outer module, which _train_epoch_levjepa bypasses
    (it calls core.encoder.forward_cls), so it is a no-op for LeVJEPA. The
    compiled method is an instance attribute: state_dict keys are unchanged
    and it survives every load_state_dict.
    """
    core = ckpt_io.unwrap(model)
    if not bool(cfg.train.get("levjepa_compile", False)) or not isinstance(
        core, _LeVJEPAPretrainModule
    ):
        return
    mode = str(cfg.train.get("compile_mode", "default"))
    core.encoder.forward_cls = torch.compile(core.encoder.forward_cls, mode=mode)


def _build_pretrain_model(cfg: DictConfig, *, device: torch.device) -> torch.nn.Module:
    """Dispatch model construction on `cfg.model.encoder`.

    A missing or `"videomae"` value builds `VideoMAEForPreTraining` exactly
    as before this dispatch existed -- no config migration needed for
    existing mesh runs. `"levjepa"` builds a `LeVJEPAEncoder` + `Projector`
    pair instead.
    """
    encoder_kind = str(cfg.model.get("encoder", "videomae"))
    if encoder_kind == "levjepa":
        return _LeVJEPAPretrainModule(cfg.model).to(device)
    return VideoMAEForPreTraining(cfg.model).to(device)


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

    # gethostname() may return a short name or an FQDN, depending on DNS config,
    # and that has changed between runs. Shard assignments, NODE_SUBNETS and
    # checkpoint names all use the short name, so normalise once here.
    node = socket.gethostname().split(".", 1)[0]
    subnet = _resolve_subnet(node, cfg)

    rounds = int(cfg.mesh.get("rounds", 10))
    start_round = int(cfg.mesh.get("start_round", 0))
    # Rounds this job runs before it exits cleanly. None runs the whole run.
    stop_round = chunk_stop_round(
        rounds, start_round, cfg.mesh.get("chunk_rounds", None)
    )
    steps_per_round = int(cfg.mesh.get("steps_per_round", 200))
    merge = bool(cfg.mesh.get("merge", True))
    outer_lr = float(cfg.mesh.get("outer_lr", 0.7))
    outer_momentum = float(cfg.mesh.get("outer_momentum", 0.9))
    sync_timeout_s = float(cfg.mesh.get("sync_timeout_s", 1800.0))
    # Per-node pacing. None keeps the uniform steps_per_round. When set, each
    # node trains whatever fits in round_budget_s at its own measured rate, so
    # rounds take equal time on mixed hardware and fast nodes stop idling at the
    # barrier. The clamps are fractions of steps_per_round.
    round_budget_s = cfg.mesh.get("round_budget_s", None)
    round_budget_s = None if round_budget_s is None else float(round_budget_s)
    step_floor = int(float(cfg.mesh.get("step_floor_frac", 0.25)) * steps_per_round)
    step_ceil = int(float(cfg.mesh.get("step_ceil_frac", 2.0)) * steps_per_round)
    rate_ema: float | None = None

    encoder_kind = str(cfg.model.get("encoder", "videomae"))
    num_frames = int(cfg.model.num_frames)
    image_size = int(cfg.model.image_size)
    overlap = float(cfg.preprocess.window.overlap)
    frame_stride = int(cfg.preprocess.window.get("frame_stride", 1))
    augment = bool(cfg.preprocess.get("augment", False))

    shard_map = load_shard_map(shard_map_path)
    train_shard_names = read_my_shards(assignments_path, node)
    _require_local_shards(train_shard_names, mesh_root, run_id=run_id, node=node)
    train_dataset = MeshShardIterableDataset(
        train_shard_names,
        mesh_root=mesh_root,
        shard_map=shard_map,
        subnet=subnet,
        num_frames=num_frames,
        image_size=image_size,
        overlap=overlap,
        frame_stride=frame_stride,
        augment=augment,
        seed=int(cfg.train.get("seed", 42)),
        raw_frames=(encoder_kind == "levjepa"),
    )

    # Val shards are replicated in full to every node under mesh_root/val/, so
    # they are always local and an empty shard_map is fine.
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
        frame_stride=frame_stride,
        raw_frames=(encoder_kind == "levjepa"),
    )

    # Opt-in (train.rng_split): views, val views and token drop each get a
    # private generator, so an arm that changes model init (or any other
    # global-RNG consumer) still sees the control's exact views and masks.
    rng_split = bool(cfg.train.get("rng_split", False))
    train_seed = int(cfg.train.get("seed", 42))
    init_seed = cfg.model.get("init_seed", None)
    if init_seed is not None and not rng_split:
        raise ValueError(
            "model.init_seed without train.rng_split=true would also reseed "
            "views and token drop; set +train.rng_split=true"
        )
    if rng_split and encoder_kind != "levjepa":
        # Views and token drop are only split on the LeVJEPA path; accepting
        # the flag elsewhere would silently isolate nothing.
        raise ValueError("train.rng_split is only implemented for model.encoder=levjepa")
    if rng_split and cfg.model.get("sigreg_seed", None) is None:
        # Unseeded SIGReg directions come from the global RNG, whose state
        # after model build differs between init arms.
        raise ValueError("train.rng_split=true requires +model.sigreg_seed")
    if rng_split and bool(cfg.train.get("levjepa_compile", False)):
        # The private drop generator fails inductor codegen once forward_cls
        # recompiles with a dynamic batch dim (GPU, torch 2.11).
        raise ValueError("train.rng_split is incompatible with train.levjepa_compile")

    if encoder_kind == "levjepa":
        # LeVJEPAViewIterableDataset wraps the raw-frames stream and builds
        # the (global_view, local_views, crop_boxes) tuple per clip -- the
        # same view-construction _levjepa_step expects, mirroring the
        # single-node pretrain_levjepa.py path.
        train_dataset = LeVJEPAViewIterableDataset(
            train_dataset,
            num_local=int(cfg.model.get("num_local_views", 4)),
            global_size=image_size,
            local_size=int(cfg.model.get("local_image_size", 96)),
            view_seed=train_seed if rng_split else None,
        )
        # Val never calls set_round, so with rng_split every val pass (every
        # round, every arm sharing train.seed) scores the same views.
        val_dataset = LeVJEPAViewIterableDataset(
            val_dataset,
            num_local=int(cfg.model.get("num_local_views", 4)),
            global_size=image_size,
            local_size=int(cfg.model.get("local_image_size", 96)),
            view_seed=derive_seed(train_seed, "val") if rng_split else None,
        )

    # Seed right before the first RNG consumer (model init, then masking), so
    # two cold starts with the same config are reproducible.
    torch.manual_seed(train_seed if init_seed is None else int(init_seed))
    model = _build_pretrain_model(cfg, device=device)
    # No DDP on this path, so compile the raw model directly. The compiled
    # module survives later master.pt reloads, since load_state_dict mutates
    # parameters in place.
    model = maybe_compile(model, cfg)
    _maybe_compile_levjepa_encoder(model, cfg)
    amp = str(cfg.train.get("amp", "off")).lower()
    if bool(cfg.train.get("tf32", False)) and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    sigreg_opts = _sigreg_options(cfg)
    accum_steps = int(cfg.train.get("accum_steps", 1))
    grad_clip = float(cfg.train.get("grad_clip", 0.0))
    micro_batch_size = int(cfg.mesh.get("micro_batch_size", _MESH_BATCH_SIZE))
    # The LR scales with the batch actually seen per step (micro_batch_size,
    # times accum_steps), not cfg.train.batch_size, which the mesh never uses.
    # This is the schedule's peak; each round's value comes from mesh_round_lr.
    # LeVJEPA configs have no base_lr fields, so they use the flat `lr`.
    peak_lr = (
        float(cfg.train.lr)
        if encoder_kind == "levjepa"
        else _scaled_lr(cfg, world_size=1, local_batch=micro_batch_size)
    )
    lr_warmup_frac = float(cfg.mesh.get("lr_warmup_frac", 0.05))
    lr_min = float(cfg.mesh.get("lr_min", 1e-5))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=peak_lr,
        betas=(0.9, 0.95),
        weight_decay=float(cfg.train.get("weight_decay", 0.0)),
    )
    # build_masker reads cfg.model.mask_ratio, which levjepa_base.yaml has no
    # field for -- masker is unused on the LeVJEPA path anyway (train_one_
    # round/evaluate_val_loss dispatch to _levjepa_step instead).
    masker = build_masker(cfg.model) if encoder_kind != "levjepa" else None

    # Eval/checkpoint-only Polyak average of the weights (see WeightEMA
    # docstring) -- never fed into the trained model or the pushed
    # checkpoint DiLoCo's outer step merges. 0/absent disables it, matching
    # every arm that doesn't set train.ema_decay.
    ema_decay = float(cfg.train.get("ema_decay", 0.0) or 0.0)
    ema = WeightEMA(ckpt_io.unwrap(model), decay=ema_decay) if ema_decay > 0 else None

    master_path = mesh_shared / "checkpoints" / run_id / "master.pt"

    last_train_loss = float("nan")
    final_round = -1
    consecutive_abandons = 0

    mesh_stop_file = os.environ.get("MESH_STOP_FILE")
    for round_num in range(start_round, stop_round):
        if mesh_stop_file and Path(mesh_stop_file).exists():
            # Checked only at round boundaries, so at most one round of work is lost.
            break
        _write_current_round_marker(mesh_shared, run_id, node, round_num)
        lr = mesh_round_lr(
            peak_lr, round_num, rounds, warmup_frac=lr_warmup_frac, min_lr=lr_min
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Load master.pt whenever it exists, not only after this process's first
        # round. A relaunched worker starts round_num back at start_round, but
        # master.pt may already reflect many rounds it never saw. Loading it
        # keeps a relaunch a resume, rather than overwriting the mesh average
        # with a stale model.
        io_stats = {
            "master_load_s": 0.0,
            "master_bytes": 0,
            "local_save_s": 0.0,
            "pending_push_s": 0.0,
            "pending_bytes": 0,
            "flag_wait_s": 0.0,
        }

        # Snapshot of the master this round started from, used to push a
        # pseudo-gradient instead of full weights. None on a cold start or when
        # merge=False, which keeps push_checkpoint_and_wait's full-weight path.
        master_sd_for_delta: dict[str, torch.Tensor] | None = None
        master_version: int | None = None

        if merge:
            if master_path.exists():
                _t0 = time.monotonic()
                io_stats["master_bytes"] = master_path.stat().st_size
                master_payload = torch.load(
                    master_path, map_location=device, weights_only=False
                )
                # The tensors here are independent of the model's, so this
                # snapshot is not changed by the load below.
                master_sd_for_delta = to_fp32_state_dict(
                    master_payload["model_state_dict"]
                )
                master_version = master_payload.get("master_version")
                load_result = ckpt_io.unwrap(model).load_state_dict(
                    master_payload["model_state_dict"], strict=False
                )
                io_stats["master_load_s"] = time.monotonic() - _t0
                if load_result.missing_keys or load_result.unexpected_keys:
                    # strict=False skips mismatched keys silently. Without this
                    # check, a key-name regression (such as a stray _orig_mod.
                    # prefix) would train on stale weights with no signal.
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
                # Rare: push_checkpoint_and_wait already waited for the sync
                # flag. Warn and keep this node's own weights rather than
                # crash a multi-day run.
                logger.warning(
                    "run_mesh_pretrain_stage: merge=True but no master "
                    "checkpoint at %s (run_id=%s round=%d) -- continuing "
                    "with this node's own weights",
                    master_path,
                    run_id,
                    round_num,
                )

        # Without set_round, every round would replay the dataset from the same
        # start, and max_steps would cut the identical windows each round.
        train_dataset.set_round(round_num)
        if rng_split and encoder_kind == "levjepa":
            ckpt_io.unwrap(model).encoder.reseed_drop(
                derive_seed(train_seed, "drop", round_num)
            )
        # rate_ema is None until a round has actually been timed, so the first
        # round this process runs is always the calibration round at the
        # configured step count -- there is nothing to pace against yet.
        round_steps = (
            steps_for_budget(
                rate_steps_per_s=rate_ema,
                budget_s=round_budget_s,
                floor_steps=step_floor,
                ceil_steps=step_ceil,
            )
            if round_budget_s is not None and rate_ema is not None
            else steps_per_round
        )
        train_stats = train_one_round(
            model,
            optimizer,
            device,
            masker,
            train_dataset,
            steps_per_round=round_steps,
            amp=amp,
            accum_steps=accum_steps,
            grad_clip=grad_clip,
            round_num=round_num,
            micro_batch_size=micro_batch_size,
            # Only when merging: a solo run has no coordinator writing flags,
            # and nothing to resync to if one appeared.
            abandon_check=(
                (lambda: round_already_merged(mesh_shared, run_id, round_num))
                if merge
                else None
            ),
            ema=ema,
            sigreg_lambda=float(cfg.model.get("sigreg_lambda", 0.02)),
            sigreg_directions=int(cfg.model.get("sigreg_directions", 1024)),
            sigreg_opts=sigreg_opts,
        )
        if train_stats.get("abandoned"):
            # The pool merged this round without us. Drop the partial round with
            # no val pass, save or push. Bailing as soon as the flag appears
            # lets the next iteration resync to the pool's round. Otherwise a
            # node slower than its pool falls further behind every round.
            logger.warning(
                "run_mesh_pretrain_stage: round %d merged without this node "
                "after %d/%d steps (run_id=%s node=%s) -- abandoning the "
                "round and resyncing to the new master",
                round_num,
                train_stats["steps"],
                round_steps,
                run_id,
                node,
            )
            consecutive_abandons += 1
            if consecutive_abandons >= _ABANDON_ALERT_STREAK:
                logger.error(
                    "run_mesh_pretrain_stage: node %s has been merged around "
                    "for %d consecutive rounds (run_id=%s) -- it is holding a "
                    "node and contributing to no merge; check for a foreign "
                    "process on its CPUs",
                    node,
                    consecutive_abandons,
                    run_id,
                )
            print(
                f"mesh round {round_num + 1}/{rounds}  ABANDONED at step "
                f"{train_stats['steps']}/{round_steps} (merged without us, "
                f"{consecutive_abandons} in a row)",
                flush=True,
            )
            _append_round_metrics(
                mesh_shared,
                run_id,
                node,
                round_num + 1,
                train_stats["loss"],
                float("nan"),
                train_stats["samples_per_s_per_gpu"],
                len(train_shard_names),
                lr,
                n_samples=train_stats["samples"],
                n_steps=train_stats["steps"],
                abandoned=True,
                consecutive_abandons=consecutive_abandons,
            )
            final_round = round_num
            continue

        consecutive_abandons = 0

        last_train_loss = train_stats["loss"]
        observed_rate = float(train_stats["samples_per_s_per_gpu"])
        if observed_rate > 0:
            rate_ema = update_rate_ema(rate_ema, observed_rate)

        val_loss = evaluate_val_loss(
            model,
            val_dataset,
            device,
            masker,
            amp,
            sigreg_lambda=float(cfg.model.get("sigreg_lambda", 0.02)),
            sigreg_directions=int(cfg.model.get("sigreg_directions", 1024)),
            sigreg_opts=sigreg_opts,
        )

        val_loss_ema: float | None = None
        if ema is not None:
            # Evaluate the EMA shadow by swapping it into the live model just
            # for this pass, then restoring -- keeps a single model instance
            # rather than a second full copy, and next round's training (and
            # the checkpoint saved below) sees only the real trained weights.
            core = ckpt_io.unwrap(model)
            live_sd = {k: v.detach().clone() for k, v in core.state_dict().items()}
            core.load_state_dict(ema.state_dict(), strict=False)
            val_loss_ema = evaluate_val_loss(
                model,
                val_dataset,
                device,
                masker,
                amp,
                sigreg_lambda=float(cfg.model.get("sigreg_lambda", 0.02)),
                sigreg_directions=int(cfg.model.get("sigreg_directions", 1024)),
                sigreg_opts=sigreg_opts,
            )
            core.load_state_dict(live_sd, strict=False)

        if train_stats["samples"] == 0:
            # A worker with no shards for this round produced an untrained
            # model -- pushing it would count toward merge quorum with
            # train_loss=nan and corrupt the average. Skip the write/push
            # entirely and just keep going; this is not a failure.
            logger.warning(
                "run_mesh_pretrain_stage: round %d fetched zero samples "
                "(run_id=%s node=%s) -- skipping local save and push",
                round_num,
                run_id,
                node,
            )
        else:
            payload = ckpt_io.base_payload(
                model=model,
                optimizer=optimizer,
                epoch=round_num,
                best_val_metric=val_loss,
                model_name="videomae_for_pretraining",
                optimizer_name="adamw",
            )
            payload["n_samples"] = train_stats["samples"]
            # Step count travels with the checkpoint because round_budget_s
            # makes it vary per node: without it the merge cannot tell a node
            # that took few steps from one that was assigned less data.
            payload["n_steps"] = train_stats["steps"]
            payload["val_loss"] = val_loss
            payload["val_loss_ema"] = val_loss_ema
            payload["round"] = round_num
            payload["samples_per_s"] = train_stats["samples_per_s_per_gpu"]
            payload["outer_lr"] = outer_lr
            # The cold-start seed. Diagnostics need it to recover this run's own
            # random init.
            payload["init_seed"] = int(cfg.train.get("seed", 42))
            payload["outer_momentum"] = outer_momentum

            local_path = checkpoint_root / f"{run_id}.pt"
            _t0 = time.monotonic()
            ckpt_io.save(local_path, payload)
            io_stats["local_save_s"] = time.monotonic() - _t0

            shared_payload = {
                k: v for k, v in payload.items() if k not in _SHARED_PAYLOAD_DROP_KEYS
            }
            io_stats.update(
                push_checkpoint_and_wait(
                    shared_payload,
                    mesh_shared=mesh_shared,
                    run_id=run_id,
                    node=node,
                    round_num=round_num,
                    merge=merge,
                    sync_timeout_s=sync_timeout_s,
                    master_sd=master_sd_for_delta,
                    master_version=master_version,
                    residual_path=checkpoint_root / f"{run_id}_push_residual.pt",
                )
            )

            # One structured line per round. The point is to separate real
            # NFS transfer time from barrier time: only the former is what
            # shrinking the payload can fix, and if flag_wait dominates then
            # the whole byte-reduction premise is wrong and the round
            # barrier is the thing to attack instead.
            logger.info(
                "mesh_io round=%d node=%s master_load_s=%.2f master_bytes=%d "
                "local_save_s=%.2f pending_push_s=%.2f pending_bytes=%d "
                "flag_wait_s=%.2f",
                round_num,
                node,
                io_stats["master_load_s"],
                io_stats["master_bytes"],
                io_stats["local_save_s"],
                io_stats["pending_push_s"],
                io_stats["pending_bytes"],
                io_stats["flag_wait_s"],
            )

        final_round = round_num
        print(
            f"mesh round {round_num + 1}/{rounds}  train_loss={train_stats['loss']:.4f}  "
            f"val_loss={val_loss:.4f}  {train_stats['samples_per_s_per_gpu']:.1f} samp/s",
            flush=True,
        )
        _append_round_metrics(
            mesh_shared,
            run_id,
            node,
            round_num + 1,
            train_stats["loss"],
            val_loss,
            train_stats["samples_per_s_per_gpu"],
            len(train_shard_names),
            lr,
            n_samples=train_stats["samples"],
            n_steps=train_stats["steps"],
            val_loss_ema=val_loss_ema,
        )

    return {
        "stage": "mesh_pretrain",
        "final_round": final_round,
        "recon_loss": float(last_train_loss),
    }


def _train_epoch_levjepa(
    model,
    loader,
    optimizer,
    device,
    *,
    accum_steps: int = 1,
    grad_clip: float = 0.0,
    max_steps: int = 0,
    amp: str = "off",
    progress_desc: str | None = None,
    abandon_check: Callable[[], bool] | None = None,
    abandon_check_every: int = 50,
    ema: WeightEMA | None = None,
    sigreg_lambda: float,
    sigreg_directions: int,
    sigreg_opts: SigregOptions | None = None,
    sigreg_key: tuple[int, ...] = (),
) -> dict:
    """LeVJEPA counterpart to `_train_epoch_ssl` for one mesh round: same
    accum/grad-clip/abandon-check/timing contract and return-dict shape, but
    the per-batch forward is `_levjepa_step` (multi-view + shared-BN
    projector + invariance/SIGReg loss) instead of `masker` + reconstruction
    loss -- the two objectives' training steps differ too much to share a
    per-batch body without an if/else that would obscure both.
    """
    model.train()
    core = ckpt_io.unwrap(model)
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
    timing_warmup_steps = 10
    timed_samples = 0
    timed_steps = 0
    executed_steps = 0
    nan_steps = 0
    abandoned = False
    pool = sigreg_opts is not None and sigreg_opts.pool
    window: list = []

    def run_window(step: int) -> None:
        nonlocal window, loss_sum, count, nan_steps
        window_n = sum(b[0].shape[0] for b in window)
        loss, components = _levjepa_pooled_window(
            window,
            core.encoder,
            core.projector,
            device,
            lam=sigreg_lambda,
            num_directions=sigreg_directions,
            opts=sigreg_opts,
            # Per optimizer step, so accum_steps=1 matches the unpooled
            # path's per-step key.
            key=(*sigreg_key, step // accum_steps),
            accum_steps=accum_steps,
            autocast_ctx=lambda: torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled
            ),
        )
        window = []
        if torch.isfinite(loss):
            loss_sum += loss.detach() * window_n
            count += window_n
        else:
            # Some micro-batches may already have backpropagated: drop the
            # whole window rather than step on part of it.
            nan_steps += 1
            optimizer.zero_grad(set_to_none=True)
            _warn_nonfinite(step, components, nan_steps)

    iterable = loader
    if progress_desc is not None:
        from tqdm import tqdm

        iterable = tqdm(loader, desc=progress_desc, total=num_steps)

    for step, batch in enumerate(iterable):
        if step == timing_warmup_steps:
            if is_cuda:
                torch.cuda.synchronize(device)
            t_start = time.perf_counter()

        batch_size = batch[0].shape[0]

        is_boundary = (step + 1) % accum_steps == 0 or (
            num_steps is not None and step + 1 == num_steps
        )
        no_sync_ctx = (
            model.no_sync()
            if not is_boundary and hasattr(model, "no_sync")
            else contextlib.nullcontext()
        )
        if pool:
            window.append(batch)
            if is_boundary:
                run_window(step)
        else:
            with no_sync_ctx:
                with torch.autocast(
                    device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled
                ):
                    loss, _components = _levjepa_step(
                        batch,
                        core.encoder,
                        core.projector,
                        device,
                        sigreg_lambda=sigreg_lambda,
                        sigreg_directions=sigreg_directions,
                        sigreg_opts=sigreg_opts,
                        sigreg_key=(*sigreg_key, step),
                    )
                    loss = loss / accum_steps
                if not torch.isfinite(loss):
                    nan_steps += 1
                    _warn_nonfinite(step, _components, nan_steps)
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
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)

        if (
            abandon_check is not None
            and is_boundary
            and executed_steps % abandon_check_every == 0
            and abandon_check()
        ):
            abandoned = True
            break

        if max_steps > 0 and executed_steps >= max_steps:
            break

    if window:
        # Loader ran out before a boundary: count and backprop the partial
        # window like the unpooled path does its trailing micro-batches.
        run_window(executed_steps - 1)

    if is_cuda:
        torch.cuda.synchronize(device)
    elapsed = (time.perf_counter() - t_start) if t_start is not None else 0.0

    samples_per_s = timed_samples / elapsed if elapsed > 0 else 0.0
    step_time_ms = (elapsed / timed_steps * 1000.0) if timed_steps > 0 else 0.0
    peak_vram_gb = torch.cuda.max_memory_allocated(device) / 1e9 if is_cuda else 0.0
    loss_val = float(loss_sum.item()) / count if count > 0 else float("nan")
    return {
        "loss": loss_val,
        "samples_per_s_per_gpu": samples_per_s,
        "step_time_ms_mean": step_time_ms,
        "peak_vram_gb": peak_vram_gb,
        "steps": executed_steps,
        "samples": count,
        "nan_steps": nan_steps,
        "abandoned": abandoned,
    }


def _warn_nonfinite(step: int, components: dict, nan_steps: int) -> None:
    # Once per round: enough to tell which term (and whether z itself) blew
    # up, without a line per skipped micro-batch.
    if nan_steps == 1:
        logger.warning(
            "levjepa step %d: non-finite loss, skipped (%s)",
            step,
            ", ".join(f"{k}={float(v):.4g}" for k, v in components.items()),
        )


def _view_sums(views: torch.Tensor, dirs: torch.Tensor) -> dict[str, torch.Tensor]:
    # views (B, V+1, K), dirs (V+1, K, M) -> sigreg_sums stacked over views.
    per_view = [sigreg_sums(views[:, v], dirs[v]) for v in range(views.shape[1])]
    return {k: torch.stack([s[k] for s in per_view]) for k in per_view[0]}


def _levjepa_pooled_window(
    batches: list,
    encoder: LeVJEPAEncoder,
    projector: Projector,
    device: torch.device,
    *,
    lam: float,
    num_directions: int,
    opts: SigregOptions,
    key: tuple[int, ...],
    accum_steps: int,
    autocast_ctx: Callable[[], contextlib.AbstractContextManager],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Backpropagate one optimizer step's micro-batches with SIGReg pooled over them.

    SIGReg depends on the batch only through per-sample sums, so the pooled
    gradient can be computed exactly in two passes. Pass 1 (no grad) sums each
    micro-batch's statistics under one shared direction set per view, then
    differentiates the pooled SIGReg with respect to those sums. Pass 2
    re-forwards each micro-batch with grad. It backprops the invariance term
    divided by accum_steps, plus the SIGReg sums' gradient dotted with that
    micro-batch's sums. Dividing by accum_steps rather than len(batches) keeps
    a short last window weighted like the unpooled path. CPU and CUDA RNG are
    replayed per micro-batch, so in-model randomness matches across passes.

    Known approximation: the projector's BatchNorm normalises each micro-batch
    with its own statistics, not the pooled batch's, and its running stats
    update in both passes.
    """
    is_cuda = device.type == "cuda"
    states, total, n, dirs = [], None, 0, None
    with torch.no_grad():
        for batch in batches:
            states.append(
                (
                    torch.get_rng_state(),
                    torch.cuda.get_rng_state(device) if is_cuda else None,
                )
            )
            with autocast_ctx():
                z_global, z_locals = _levjepa_embed(batch, encoder, projector, device)
            with torch.autocast(device_type=device.type, enabled=False), _no_tf32():
                views = torch.cat([z_global.unsqueeze(1), z_locals], dim=1).float()
                if dirs is None:
                    V1, K = views.shape[1], views.shape[2]
                    dirs = view_directions(
                        opts, key, V1, K, num_directions, device
                    ) or [
                        sample_directions(K, num_directions, device) for _ in range(V1)
                    ]
                    dirs = torch.stack(dirs)
                sums = _view_sums(views, dirs)
            total = sums if total is None else {k: total[k] + sums[k] for k in sums}
            n += views.shape[0]

    leaves = {k: v.detach().requires_grad_() for k, v in total.items()}
    with torch.autocast(device_type=device.type, enabled=False), _no_tf32():
        sigreg = sigreg_from_sums(leaves, n, moment_weight=opts.moment_weight)
    grads = dict(
        zip(
            leaves,
            torch.autograd.grad(lam * sigreg, list(leaves.values()), allow_unused=True),
        )
    )
    inv_sum = torch.zeros((), device=device)
    for batch, (cpu_state, cuda_state) in zip(batches, states):
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, device)
        with autocast_ctx():
            z_global, z_locals = _levjepa_embed(batch, encoder, projector, device)
        with torch.autocast(device_type=device.type, enabled=False), _no_tf32():
            z_global, z_locals = z_global.float(), z_locals.float()
            inv = invariance_loss(z_global, z_locals)
            sums = _view_sums(torch.cat([z_global.unsqueeze(1), z_locals], dim=1), dirs)
            surrogate = (1 - lam) * inv / accum_steps + sum(
                (g * sums[k]).sum() for k, g in grads.items() if g is not None
            )
        if torch.isfinite(surrogate):
            surrogate.backward()
        inv_sum += inv.detach()
    inv_mean = inv_sum / len(batches)
    sigreg = sigreg.detach()
    return (1 - lam) * inv_mean + lam * sigreg, {
        "invariance": inv_mean,
        "sigreg": sigreg,
    }


def _evaluate_val_loss_levjepa(
    model,
    val_dataset,
    device,
    amp: str,
    *,
    max_windows: int = 2000,
    sigreg_lambda: float,
    sigreg_directions: int,
    sigreg_opts: SigregOptions | None = None,
) -> float:
    """LeVJEPA counterpart to evaluate_val_loss: same no-grad/max_windows/
    model.train()-on-exit contract, using _levjepa_step instead of masker +
    reconstruction loss."""
    model.eval()
    core = ckpt_io.unwrap(model)
    autocast_enabled = amp in ("bf16", "fp16")
    amp_dtype = torch.bfloat16 if amp == "bf16" else torch.float16

    loader = DataLoader(val_dataset, batch_size=_LEVJEPA_VAL_BATCH_SIZE, num_workers=0)
    loss_sum = torch.zeros((), device=device)
    finite_count = 0
    windows_seen = 0
    try:
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if windows_seen >= max_windows:
                    break
                batch_size = batch[0].shape[0]
                if batch_size < 2:  # 1-clip tail: _levjepa_step rejects it
                    break
                windows_seen += batch_size

                with torch.autocast(
                    device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled
                ):
                    loss, _components = _levjepa_step(
                        batch,
                        core.encoder,
                        core.projector,
                        device,
                        sigreg_lambda=sigreg_lambda,
                        sigreg_directions=sigreg_directions,
                        sigreg_opts=sigreg_opts,
                        # Round-independent key: with a seed, every round's
                        # val pass sees the same directions per batch.
                        sigreg_key=(-1, i),
                    )
                if torch.isfinite(loss):
                    loss_sum += loss.detach()
                    finite_count += 1
    finally:
        model.train()

    return float(loss_sum.item()) / finite_count if finite_count > 0 else float("nan")


def train_one_round(
    model,
    optimizer,
    device,
    masker,
    dataset,
    *,
    steps_per_round: int,
    amp: str,
    accum_steps: int,
    grad_clip: float,
    round_num: int,
    micro_batch_size: int = _MESH_BATCH_SIZE,
    abandon_check: Callable[[], bool] | None = None,
    ema: WeightEMA | None = None,
    sigreg_lambda: float | None = None,
    sigreg_directions: int | None = None,
    sigreg_opts: SigregOptions | None = None,
) -> dict:
    """Run one mesh round through the VideoMAE or LeVJEPA epoch loop.

    The loop is picked from the unwrapped model's type. The loader uses
    num_workers=0, since MeshShardIterableDataset already prefetches on a
    thread, and a second worker layer would race it for the same cache dir.
    Returns the epoch loop's stats dict unchanged.
    """
    core = ckpt_io.unwrap(model)
    is_levjepa = isinstance(core, _LeVJEPAPretrainModule)
    # drop_last for LeVJEPA: a 1-clip tail batch would trip _levjepa_step's guard.
    loader = DataLoader(
        dataset, batch_size=micro_batch_size, num_workers=0, drop_last=is_levjepa
    )
    if is_levjepa:
        return _train_epoch_levjepa(
            model,
            loader,
            optimizer,
            device,
            accum_steps=accum_steps,
            grad_clip=grad_clip,
            max_steps=steps_per_round,
            amp=amp,
            progress_desc=f"round {round_num}",
            abandon_check=abandon_check,
            ema=ema,
            sigreg_lambda=float(sigreg_lambda if sigreg_lambda is not None else 0.02),
            sigreg_directions=int(
                sigreg_directions if sigreg_directions is not None else 1024
            ),
            sigreg_opts=sigreg_opts,
            sigreg_key=(round_num,),
        )
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
        abandon_check=abandon_check,
        ema=ema,
    )


def evaluate_val_loss(
    model,
    val_dataset,
    device,
    masker,
    amp: str,
    *,
    max_windows: int = 2000,
    sigreg_lambda: float | None = None,
    sigreg_directions: int | None = None,
    sigreg_opts: SigregOptions | None = None,
) -> float:
    """No-grad pass over up to max_windows windows from val_dataset.

    Returns mean reconstruction loss (VideoMAE) or mean levjepa_loss (LeVJEPA),
    chosen from the unwrapped model's type. Each round uses a fixed subsample of
    about 2000 windows. Leaves `model.training` True on return, even if this
    raises, because the next round's training expects that.
    """
    core = ckpt_io.unwrap(model)
    if isinstance(core, _LeVJEPAPretrainModule):
        return _evaluate_val_loss_levjepa(
            model,
            val_dataset,
            device,
            amp,
            max_windows=max_windows,
            sigreg_lambda=float(sigreg_lambda if sigreg_lambda is not None else 0.02),
            sigreg_directions=int(
                sigreg_directions if sigreg_directions is not None else 1024
            ),
            sigreg_opts=sigreg_opts,
        )

    model.eval()
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
                keep_indices, mask_indices = masker(
                    pixel_values, token_grid, device, pixel_values.shape[0]
                )
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
    master_sd: dict[str, torch.Tensor] | None = None,
    master_version: int | None = None,
    residual_path: Path | None = None,
) -> dict[str, float]:
    """Push `payload` to the shared pending/ dir, then wait if merging.

    Returns `{"pending_push_s", "pending_bytes", "flag_wait_s", "pushed"}` for
    the caller's per-round I/O log.

    The file goes to mesh_shared/checkpoints/pending/ via `checkpoint.try_save`,
    so a transient NFS failure does not raise. On exhausted retries it returns
    `pushed=False` and skips the barrier, and the caller carries on from its
    own weights.

    When `master_sd` is given, the push is `master_sd - model_state_dict`, a
    quantized pseudo-gradient, not the full weights. Quantizing the small
    difference loses far less precision than quantizing whole weights.
    `master_version` lets the merge reject a delta computed against a stale
    master. With `master_sd=None` (round 0, or a solo run), the full state dict
    is pushed in fp16.

    With `merge=False`, returns right after the push. With `merge=True`, polls
    for mesh_shared/checkpoints/flags/sync_complete_{run_id}_{round_num}.flag
    every 5 s, up to sync_timeout_s. On timeout it logs an error and exits the
    process. The watcher's existing stall detectors pick up that exit.
    """
    mesh_shared = Path(mesh_shared)
    pending_path = (
        mesh_shared
        / "checkpoints"
        / "pending"
        / pending_filename(run_id, round_num, node)
    )
    pending_path.parent.mkdir(parents=True, exist_ok=True)
    shared_payload = dict(payload)
    if master_sd is not None:
        delta_sd = pseudo_gradient(master_sd, payload["model_state_dict"])
        shared_payload["delta_state_dict"] = quantize_with_error_feedback(
            delta_sd, residual_path
        )
        shared_payload["master_version"] = master_version
        del shared_payload["model_state_dict"]
    else:
        shared_payload["model_state_dict"] = _fp16_state_dict(
            payload["model_state_dict"]
        )
    _t0 = time.monotonic()
    pushed = ckpt_io.try_save(pending_path, shared_payload)
    stats = {
        "pending_push_s": time.monotonic() - _t0,
        "pending_bytes": pending_path.stat().st_size if pending_path.exists() else 0,
        "flag_wait_s": 0.0,
        "pushed": pushed,
    }

    if not pushed:
        logger.error(
            "push_checkpoint_and_wait: checkpoint push failed after retries, "
            "skipping this round's barrier (run_id=%s round_num=%d node=%s)",
            run_id,
            round_num,
            node,
        )
        return stats

    if not merge:
        return stats

    flag_path = (
        mesh_shared
        / "checkpoints"
        / "flags"
        / f"sync_complete_{run_id}_{round_num}.flag"
    )
    start = time.monotonic()
    while not flag_path.exists():
        elapsed = time.monotonic() - start
        stats["flag_wait_s"] = elapsed
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

    stats["flag_wait_s"] = time.monotonic() - start
    return stats
