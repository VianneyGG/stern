#!/usr/bin/env bash
# video_encoding/slurm/train.sbatch
# Generic SLURM submission wrapper around scripts/run_experiment.sh.
# Submitted via submit_login-node.sh, which overrides --nodes/--ntasks/--nodelist
# at submission time via pick_nodes.sh. Directives here are fallback defaults.
# Also submitted directly (same --nodes/--ntasks/--nodelist flag contract, no
# pick_nodes.sh) by the training watcher's auto-resubmit path,
# video_encoding/watcher/remediate.py -- keep both callers in sync if this
# script's flag/argument contract ever changes.
#
# Cluster facts (T1.1/T1.2/T1.7): single partition SallesInfo, no GPU GRES
# (whole-node allocation — request nodes not gres), 16 cores/node, ~63 GB RAM,
# 3-day max walltime, no preemption.
#
# Usage: sbatch [--nodes=N --ntasks=N --nodelist=...] train.sbatch <experiment-name> [overrides]
#SBATCH --job-name=stern-train
#SBATCH --partition=SallesInfo
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --exclusive
# --exclusive: SLURM here does NOT enforce GPU isolation, so without this other
# students' jobs land on the same node and the 24 GB GPU OOMs (observed). WHN
# uses the same guard. Costs scheduling latency (needs a fully-free node).
#SBATCH --time=3-00:00:00
#SBATCH --signal=USR1@120
#SBATCH --output=video_encoding/slurm/logs/%x-%j.out

# --signal=USR1@120: SLURM sends SIGUSR1 two minutes before walltime. Nothing
# in this repo currently catches it (no signal handler exists) -- the actual
# safety net is the retry loop below (up to 30 sbatch resubmissions relying
# on cfg.train.resume=true), not a graceful in-process handler. This
# directive is harmless/inert; kept in case a handler is added later.

set -euo pipefail

EXPERIMENT="${1:?usage: sbatch train.sbatch <experiment-name> [hydra overrides...]}"
shift

# DDP rendezvous env vars. All ranks must agree on MASTER_ADDR/MASTER_PORT
# BEFORE torch.distributed.init_process_group runs (video_encoding/training/ddp.py:setup_ddp).
# Port derived from SLURM_JOB_ID to avoid collisions between concurrent jobs
# sharing a login node's port namespace.
export MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)"
export MASTER_PORT="$((15000 + SLURM_JOB_ID % 5000))"

# Checkpoints go to node-local /Data (1TB, persistent across jobs), NOT $HOME:
# the student home is a 30 GB quota with only ~11 GB free, and there is no
# shared large filesystem on this cluster. /Data is where the VoxCeleb2 frames
# are staged too, so a run is already pinned to its data node — same-node
# resubmit resumes cleanly from here. Retrieve checkpoints with an explicit
# rsync from the node (they are not in the sync_login-node pull set). Overridable.
export STERN_CHECKPOINT_ROOT="${STERN_CHECKPOINT_ROOT:-/Data/$USER/stern_checkpoints}"

# VoxCeleb2 SSL data. Staged node-locally under /Data on the fish nodes
# (node001/node002/node003/node004, plus node023 for single-node traditional
# training) as balanced WDS shards (video_encoding/scripts/build_wds_shards.py
# + distribute_shards.sh). The job MUST be pinned to a data node (see
# submit_login-node.sh's FIXED_NODELIST).
export VOXMINI_PRESENT="${VOXMINI_PRESENT:-true}"
export VOXMINI_ROOT="${VOXMINI_ROOT:-/Data/$USER/voxceleb2}"
export VOXMINI_SHARDS_ROOT="${VOXMINI_SHARDS_ROOT:-/Data/$USER/voxceleb2_shards}"
# Expected shard count -- override if a node stages a different-sized subset
# (the full corpus, staged 2026-07-17, is 260 shards; the original 218h
# voxmini subset was 100).
export VOXMINI_N_SHARDS="${VOXMINI_N_SHARDS:-260}"

# Fail fast if this node doesn't actually have the shard set, instead of
# relying on a deep-dataloader FileNotFoundError plus the 30-min NCCL
# heartbeat timeout to notice (train.sbatch's existing --kill-on-bad-exit
# comment documents that exact 48-min-hang failure mode for a similar gap).
if [[ "${VOXMINI_PRESENT}" == "true" ]]; then
  n_shards=$(find "${VOXMINI_SHARDS_ROOT}" -maxdepth 1 -name 'shard-*.tar' 2>/dev/null | wc -l)
  if [[ ! -f "${VOXMINI_SHARDS_ROOT}/manifest.json" || "${n_shards}" -ne "${VOXMINI_N_SHARDS}" ]]; then
    echo "ERROR: VOXMINI_SHARDS_ROOT=${VOXMINI_SHARDS_ROOT} on $(hostname) has ${n_shards}/${VOXMINI_N_SHARDS} shards" \
         "or is missing manifest.json -- run build_wds_shards.py + distribute_shards.sh before submitting." >&2
    exit 1
  fi
fi

# Keep uv's wheel cache + torch caches on node-local /tmp, NOT the 30 GB student
# home quota (~11 GB free): cu128 torch wheels are multi-GB and would blow it.
# (Mirrors the What_Happens_Next slurm scripts on this cluster.)
export UV_CACHE_DIR="/tmp/uv_cache_${SLURM_JOB_ID}"
export TORCH_HOME="/tmp/torch_home_${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="/tmp/torch_inductor_${SLURM_JOB_ID}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Give straggler nodes up to 30 min in a collective before the NCCL watchdog
# kills the job (transient CPU jitter on shared classroom PCs).
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800

# SLURM executes the batch script from a spool copy, so BASH_SOURCE is unreliable
# here. Use SLURM_SUBMIT_DIR (the repo root ~/stern that submit_login-node ran
# `sbatch` from); run_experiment.sh re-cds to the repo root itself.
cd "${SLURM_SUBMIT_DIR:-$HOME/stern}"
# --kill-on-bad-exit=1: without this, if some ranks crash early (e.g. missing
# per-node data) while another survives into a DDP collective, the survivor
# hangs until the NCCL heartbeat watchdog eventually fires (up to 30min,
# TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC below) instead of the whole job dying in
# seconds. Observed in the wild: 7/8 ranks FileNotFoundError'd instantly,
# rank 0 hung 48min waiting for them at wrap_ddp's parameter broadcast.
#
# Retry on crash, relying on train.resume=true (default) to continue from the
# last checkpoint each attempt. Needed for unattended multi-day runs: transient
# CUDA/NCCL failures (illegal memory access, collective timeout under GPU
# contention) hit roughly 1 in 5 short cells in practice -- without this, one
# bad moment on a shared classroom GPU kills an 800-epoch run outright with
# nobody watching. Capped at 30 attempts (not unbounded): a genuinely broken
# config should give up in minutes, not silently burn the full 3-day walltime
# retrying something that can never succeed.
attempt=0
while true; do
  attempt=$((attempt + 1))
  echo "=== attempt ${attempt} $(date +%H:%M:%S) ==="
  if srun --kill-on-bad-exit=1 bash "video_encoding/scripts/run_experiment.sh" "${EXPERIMENT}" "$@"; then
    echo "=== succeeded on attempt ${attempt} $(date +%H:%M:%S) ==="
    break
  fi
  if [[ "${attempt}" -ge 30 ]]; then
    echo "=== gave up after ${attempt} attempts $(date +%H:%M:%S) ===" >&2
    exit 1
  fi
  echo "=== attempt ${attempt} failed, retrying in 30s (resume=true continues from last checkpoint) ==="
  sleep 30
done
