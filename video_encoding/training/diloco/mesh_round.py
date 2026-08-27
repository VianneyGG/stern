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
  2. Train `H` local steps over this node's assigned shards
     (`train_one_round`). `H` is `cfg.mesh.steps_per_round` unless
     `cfg.mesh.round_budget_s` is set, in which case it is whatever fits in
     that budget at this node's own measured rate (`mesh/pacing.py`).
  3. Evaluate a fixed ~2000-window val subsample (`evaluate_val_loss`).
  4. Save a local checkpoint, then push it to the shared `pending/` dir and
     (if merging) block until the watcher's sync flag appears
     (`push_checkpoint_and_wait`).
"""

from __future__ import annotations

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

from video_encoding.data.mesh_shard_dataset import MeshShardIterableDataset
from video_encoding.mesh.assignment import read_my_shards
from video_encoding.mesh.shard_map import load_shard_map
from video_encoding.models.videomaev2.mae import VideoMAEForPreTraining
from video_encoding.models.videomaev2.masking import build_masker
from video_encoding.mesh.topology import NODE_SUBNETS
from video_encoding.training import checkpoint as ckpt_io
from video_encoding.training.compile_utils import maybe_compile
from video_encoding.mesh.outer_step import pseudo_gradient, to_fp32_state_dict
from video_encoding.mesh.pacing import steps_for_budget, update_rate_ema
from video_encoding.mesh.pending_files import pending_filename
from video_encoding.mesh.quantize import quantize_with_error_feedback
from video_encoding.training.pretrain import _scaled_lr, _train_epoch_ssl

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

# Keys present in the full round payload (checkpoint.base_payload()) that are
# dropped before the payload is pushed to the shared NFS pending/ dir.
# run_outer_step.py::_load_worker_checkpoints only ever reads
# model_state_dict/n_samples/val_loss from a pending file, and AdamW's
# exp_avg+exp_avg_sq buffers roughly double payload size -- shipping them to
# NFS is pure waste, read by nobody, held until the next merge tick or
# ROUND_TIMEOUT_S, then unlinked. The LOCAL checkpoint (checkpoint_root/
# {run_id}.pt) keeps these keys unchanged; only the shared push drops them.
_SHARED_PAYLOAD_DROP_KEYS = ("optimizer_state_dict", "optimizer_name")


def _fp16_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Casts every floating-point tensor to fp16, leaving any non-float
    buffer untouched. Halves the ~365MB/round NFS write in
    push_checkpoint_and_wait; run_outer_step.py upcasts back to fp32
    before any merge arithmetic runs on it."""
    return {
        k: (v.half() if torch.is_floating_point(v) else v)
        for k, v in state_dict.items()
    }


# Three rounds in a row merged without us is no longer a transient: it means
# this node's cadence is persistently slower than its pool's, and preemption
# is silently keeping it in the roster while it contributes nothing.
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
) -> None:
    """Append one JSONL record to mesh_shared/metrics/<run_id>/<node>.jsonl.

    Per-round loss otherwise exists only in this process's stdout, scattered
    across one SLURM job per node per run -- an 8-run sweep must not repeat
    the bespoke multi-job log scraper that took to extract it last time. A
    failed write is logged and swallowed: metrics are diagnostic, never a
    reason to kill a training round.
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
        # Exact counts, not samp_s x round_budget_s: the inferred axis is
        # wrong for a fetch-starved node, which is what a scaling sweep
        # measures.
        "n_samples": n_samples,
        "n_steps": n_steps,
        # A preempted node writes a record too, so that "this node stopped
        # contributing" is visible in the same stream we already aggregate
        # rather than showing up as a silent gap. Preemption trades a loud
        # failure (the pool stalls behind a straggler) for a quiet one (the
        # straggler holds a whole node and merges into nothing), and this is
        # what keeps the quiet one countable.
        "abandoned": abandoned,
        "consecutive_abandons": consecutive_abandons,
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
    """This node's real subnet, for mesh/shard_map.py's same-subnet peer
    preference (cross-switch uplink caps at ~0.94 Gbps total vs. a
    non-blocking intra-switch backplane -- shard_map.py's docstring).

    NODE_SUBNETS (mesh/topology.py) is the same hostname->subnet mapping the
    rest of the mesh design is keyed on (fish=/254, birds=/253, cars=/252,
    countries=/254, orchids=/253) -- a single yaml-wide constant would
    misreport every birds/cars/orchids node as fish's "/254", silently
    defeating the same-subnet preference for more than half the roster.
    `cfg.mesh.subnet` is only a fallback for hosts NODE_SUBNETS doesn't know
    about (e.g. a manual test run off the real roster).
    """
    return NODE_SUBNETS.get(hostname, str(cfg.mesh.get("subnet", "/254")))


def _require_local_shards(
    shard_names: list[str], mesh_root: Path, *, run_id: str, node: str
) -> None:
    """Exit non-zero if this node was assigned a shard it does not hold.

    Assignment is static (mesh/assignment.py::assign_static), so this can
    only fire when the assignment file disagrees with the disk -- the exact
    condition that ran undetected for two weeks before 2026-08-20, because
    the only symptom was a skipped-shard WARNING and a zero-sample round
    that the worker treated as "not a failure".

    A worker that cannot train on what it was given must die loudly at
    startup, not spend the round burning a GPU at 0 samp/s.
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
        "Re-run validate_seeding.py; do not restart this worker."
    )


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
    """Last round (exclusive) THIS worker job runs, vs `rounds` for the run.

    `chunk_rounds is None` means one job spans the whole run -- the
    pre-2026-08-22 behaviour, and the default, so every already-registered
    run is unaffected. Chunking became necessary when the rebuilt cluster
    capped SallesInfo at a 2h walltime: an ~11h 40-round run no longer fits
    in one job, so the watcher relaunches successive chunks that resume via
    +mesh.start_round and master.pt.

    Exiting cleanly under the cap is the whole point. A walltime kill instead
    takes all 16 workers down mid-round at the same instant, every one misses
    the same round, and that is precisely what permanently excludes an entire
    roster.

    Clamped to `rounds` so the final chunk stops at the run's real end rather
    than overshooting, and never below `start_round` (a non-positive
    chunk_rounds yields an empty range and an immediate clean exit rather
    than a backwards range that would silently skip the round).
    """
    if chunk_rounds is None:
        return rounds
    return max(start_round, min(rounds, start_round + int(chunk_rounds)))


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

    # socket.gethostname() reports whatever the node's NSS/DNS config
    # returns -- a short name ("node008") or an FQDN ("node008.
    # polytechnique.fr") depending on cluster configuration, and this has
    # been observed to change cluster-wide between mesh runs. shard
    # assignments, NODE_SUBNETS, and the per-node checkpoint naming
    # convention are all keyed by the short name, so normalize here once
    # rather than at each lookup site.
    node = socket.gethostname().split(".", 1)[0]
    subnet = _resolve_subnet(node, cfg)

    rounds = int(cfg.mesh.get("rounds", 10))
    start_round = int(cfg.mesh.get("start_round", 0))
    # How many rounds THIS job runs before exiting cleanly, vs `rounds` which
    # is the run's absolute length. None means one job spans the whole run --
    # the pre-2026-08-22 behaviour, kept as the default so every already
    # registered run is unaffected. It became necessary when the rebuilt
    # cluster capped SallesInfo at a 2h walltime: an 11h 40-round run no
    # longer fits in one job, so the watcher relaunches successive chunks that
    # resume through +mesh.start_round and master.pt. Exiting under the cap is
    # the point -- being walltime-killed instead kills all 16 workers mid-round
    # simultaneously, and every one then misses the same round, which is what
    # permanently excludes a whole roster.
    stop_round = chunk_stop_round(
        rounds, start_round, cfg.mesh.get("chunk_rounds", None)
    )
    steps_per_round = int(cfg.mesh.get("steps_per_round", 200))
    merge = bool(cfg.mesh.get("merge", True))
    outer_lr = float(cfg.mesh.get("outer_lr", 0.7))
    outer_momentum = float(cfg.mesh.get("outer_momentum", 0.9))
    sync_timeout_s = float(cfg.mesh.get("sync_timeout_s", 1800.0))
    # Per-node pacing (spec 2.3'): None keeps the uniform steps_per_round every
    # node uses today. When set, each node instead trains whatever fits in
    # round_budget_s at its own measured rate, so rounds take equal *time*
    # across heterogeneous hardware and the fast nodes stop idling at the
    # barrier. The clamps are expressed as fractions of steps_per_round so a
    # run only has to tune one number.
    round_budget_s = cfg.mesh.get("round_budget_s", None)
    round_budget_s = None if round_budget_s is None else float(round_budget_s)
    step_floor = int(float(cfg.mesh.get("step_floor_frac", 0.25)) * steps_per_round)
    step_ceil = int(float(cfg.mesh.get("step_ceil_frac", 2.0)) * steps_per_round)
    rate_ema: float | None = None

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
        frame_stride=frame_stride,
    )

    # Production never seeds torch before this function runs (the only
    # manual_seed call in the whole training path is pretrain.py's Stage-2
    # branch), so model init and RandomTubeMasking's torch.rand/torch.randperm
    # calls were previously left to whatever RNG state the process happened
    # to be in -- making cold-start runs statistically incomparable to each
    # other. Reseed from cfg.train.seed here, right before the first RNG
    # consumer, so two cold starts with the same cfg are reproducible.
    torch.manual_seed(int(cfg.train.get("seed", 42)))
    model = VideoMAEForPreTraining(cfg.model).to(device)
    # No DDP on this path (single-GPU worker, see module docstring), so
    # compile happens directly on the raw model -- no wrap_ddp ordering to
    # worry about. Compiled once here; surviving every later master.pt
    # reload since load_state_dict mutates parameters in place.
    model = maybe_compile(model, cfg)
    amp = str(cfg.train.get("amp", "off")).lower()
    accum_steps = int(cfg.train.get("accum_steps", 1))
    grad_clip = float(cfg.train.get("grad_clip", 0.0))
    micro_batch_size = int(cfg.mesh.get("micro_batch_size", _MESH_BATCH_SIZE))
    # local_batch=micro_batch_size: the mesh path's real per-optimizer-step
    # sample count is micro_batch_size * accum_steps, NOT cfg.train.batch_size
    # (16 in ssl.yaml, never overridden here) -- _scaled_lr's linear-scaling
    # rule must scale off the batch actually seen per step, or the LR is
    # silently mistuned for every mesh run regardless of accum_steps. Kept in
    # its own variable (not inlined into the AdamW call) so it can also be
    # reported alongside per-round metrics. It is the schedule's *peak*: the
    # per-round value is set from mesh_round_lr below, before each round runs.
    peak_lr = _scaled_lr(cfg, world_size=1, local_batch=micro_batch_size)
    lr_warmup_frac = float(cfg.mesh.get("lr_warmup_frac", 0.05))
    lr_min = float(cfg.mesh.get("lr_min", 1e-5))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=peak_lr,
        betas=(0.9, 0.95),
        weight_decay=float(cfg.train.get("weight_decay", 0.0)),
    )
    masker = build_masker(cfg.model)

    master_path = mesh_shared / "checkpoints" / run_id / "master.pt"

    last_train_loss = float("nan")
    final_round = -1
    consecutive_abandons = 0

    for round_num in range(start_round, stop_round):
        lr = mesh_round_lr(
            peak_lr, round_num, rounds, warmup_frac=lr_warmup_frac, min_lr=lr_min
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Not "round_num > start_round": a worker relaunched mid-run (e.g.
        # after the watcher process died) starts its local round_num back at
        # start_round, but master.pt may already reflect many prior rounds
        # this process never saw. Loading it whenever it exists -- not just
        # after this process's own first local round -- is what makes a
        # relaunch a resume (and a late joiner's first round a real join)
        # instead of silently overwriting the mesh average with a
        # fresh-init/stale model's contribution.
        io_stats = {
            "master_load_s": 0.0,
            "master_bytes": 0,
            "local_save_s": 0.0,
            "pending_push_s": 0.0,
            "pending_bytes": 0,
            "flag_wait_s": 0.0,
        }

        # Snapshot of the master this round started from, kept around to push
        # a pseudo-gradient (Phase 3.2) instead of the full trained weights.
        # None on a cold start (no master.pt yet) or merge=False -- both keep
        # push_checkpoint_and_wait's pre-3.2 full-weight path.
        master_sd_for_delta: dict[str, torch.Tensor] | None = None
        master_version: int | None = None

        if merge:
            if master_path.exists():
                _t0 = time.monotonic()
                io_stats["master_bytes"] = master_path.stat().st_size
                master_payload = torch.load(
                    master_path, map_location=device, weights_only=False
                )
                # torch.load returns tensors independent of what
                # load_state_dict copies into the model below, so this stays
                # an untouched snapshot of the round's starting master.
                master_sd_for_delta = to_fp32_state_dict(
                    master_payload["model_state_dict"]
                )
                master_version = master_payload.get("master_version")
                load_result = ckpt_io.unwrap(model).load_state_dict(
                    master_payload["model_state_dict"], strict=False
                )
                io_stats["master_load_s"] = time.monotonic() - _t0
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

        # Without this, a fresh DataLoader built inside train_one_round every
        # round replayed the dataset's fixed insertion order from the top,
        # so max_steps=steps_per_round truncated at the identical windows
        # every round regardless of round_num.
        train_dataset.set_round(round_num)
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
        )
        if train_stats.get("abandoned"):
            # The pool merged this round without us. Drop the partial round
            # whole -- no val pass, no local save, no push. The next
            # iteration's master load is the resync, and because we bail the
            # moment the flag appears, that iteration's round_num is the one
            # the pool is now on: the lag stops here instead of compounding
            # by (our cadence - the pool's) every round. Observed 2026-08-23:
            # node055 ran 442s/round slower than its pool for 23 rounds,
            # contributing to six merges out of 32 and costing 3.08h of
            # chunk-boundary stalls.
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

        val_loss = evaluate_val_loss(model, val_dataset, device, masker, amp)

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
            payload["round"] = round_num
            payload["samples_per_s"] = train_stats["samples_per_s_per_gpu"]
            payload["outer_lr"] = outer_lr
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
        )

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
    micro_batch_size: int = _MESH_BATCH_SIZE,
    abandon_check: Callable[[], bool] | None = None,
) -> dict:
    """Thin wrapper around _train_epoch_ssl for one mesh round.

    Builds a DataLoader over `dataset` (num_workers=0 -- MeshShardIterableDataset's
    own prefetch thread is the concurrency; a second DataLoader worker layer
    would race it for the same cache dir), calls _train_epoch_ssl with
    max_steps=steps_per_round and progress_desc=f"round {round_num}", and
    returns its stats dict unchanged.
    """
    loader = DataLoader(dataset, batch_size=micro_batch_size, num_workers=0)
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
                keep_indices, mask_indices = masker(pixel_values, token_grid, device, pixel_values.shape[0])
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
    """Push `payload` to the shared pending/ dir, then (if merging) wait.

    Returns `{"pending_push_s", "pending_bytes", "flag_wait_s", "pushed"}`
    for the caller's per-round I/O log -- the push and the barrier wait are
    the two costs that have to be told apart before any NFS optimization can
    be judged, and they are only separable here.

    Writes to mesh_shared/checkpoints/pending/{run_id}_{round_num}_{node}.pt
    (mesh/pending_files.pending_filename -- the round segment is what lets
    every reader tell this round's submission apart from a straggler still
    pushing a prior round's checkpoint after that round's merge already
    cleared pending/) via `checkpoint.try_save` so a transient NFS write
    failure degrades instead of raising: on exhausted retries this returns
    `pushed=False` and skips the barrier entirely, leaving the caller to
    proceed to the next round from its own weights rather than dying
    mid-run.

    When `master_sd` is given (every round this worker had a master to load,
    i.e. every round but round 0), pushes `master_sd - model_state_dict`
    (Phase 3.2) instead of the full weights: DiLoCo's actual signal is that
    small difference, and quantizing it directly (with the same error
    feedback as the master broadcast, via `residual_path`) wastes far less
    precision than quantizing the large weight values before the merge side
    subtracts them. `master_version` travels alongside so the merge can
    reject a delta computed against a master that's no longer current rather
    than silently applying it to the wrong one. When `master_sd` is None
    (round 0, or a merge=False solo run with nothing to diff against), keeps
    pushing the full `model_state_dict` cast to fp16, unchanged from before
    this parameter existed.

    If the push succeeded and `merge` is False (the ab-single_node reference
    arm), returns immediately -- no wait, the caller proceeds to the next
    round locally. If the push succeeded and `merge` is True, polls for
    mesh_shared/checkpoints/flags/sync_complete_{run_id}_{round_num}.flag
    every 5s up to sync_timeout_s; on timeout, logs an ERROR naming
    run_id/round_num/node and calls sys.exit(1) -- NOT escalate.py, which
    operates on watcher-process state, not a worker job's own state. The
    existing, unmodified dead/stalled-job detectors catch this exit through
    the normal cron path.
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
