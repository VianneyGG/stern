#!/usr/bin/env bash
# video_encoding/scripts/run_mesh_worker.sh
# Single-GPU worker entrypoint for the decentralized P2P training mesh.
#
# Unlike train.sbatch (DDP, --nodes=8, rendezvous, retry loop), each mesh
# worker is an independent single-process trainer on its own shard subset:
# no MASTER_ADDR/MASTER_PORT, no srun, no in-job retry. A dead worker is
# excluded and its shards reassigned by the watcher
# (video_encoding/watcher/epoch_advance.py's missed-round threshold), not
# retried here -- an in-job retry would hide the failure from round
# bookkeeping instead of surfacing a missed round for the next merge.
#
# Usage: run_mesh_worker.sh <experiment> <run-id> [hydra overrides...]
set -euo pipefail

EXPERIMENT="${1:?usage: run_mesh_worker.sh <experiment> <run-id> [hydra overrides...]}"
RUN_ID="${2:?usage: run_mesh_worker.sh <experiment> <run-id> [hydra overrides...]}"
shift 2

# Run from the repo root, same resolution pattern as run_experiment.sh.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# sbatch propagates the submitter's environment by default (--export=ALL).
# When the watcher submits this job from cron, cron's own PATH
# (/usr/bin:/bin) doesn't include ~/.local/bin, so an inherited-PATH-only
# `uv run` below fails instantly with "uv: command not found" regardless of
# who submitted the job. Make this script self-sufficient rather than
# depending on the submitter having a correct PATH.
export PATH="${HOME}/.local/bin:${PATH}"

export MESH_RUN_ID="${RUN_ID}"
# Deliberately OUTSIDE ~/stern: sync_login-node.sh pushes source with
# `rsync --delete`, which would wipe round-local mesh state (assignments,
# shard map, pending checkpoints) if it lived under the synced repo tree.
export MESH_SHARED="${MESH_SHARED:-${HOME}/stern_mesh}"
export MESH_ROOT="${MESH_ROOT:-/Data/$USER/mesh}"
export MESH_ASSIGNMENTS="${MESH_ASSIGNMENTS:-${MESH_SHARED}/metadata/shard_assignments_${MESH_RUN_ID}.json}"
export MESH_SHARD_MAP="${MESH_SHARD_MAP:-${MESH_SHARED}/metadata/shard_map.json}"

# Fail fast, before the trainer's dataloader would otherwise discover any of
# this deep inside its first batch. train.sbatch documents a 48-minute hang
# from exactly this class of gap (a straggler waiting on data that was never
# going to arrive) -- catch it here in seconds instead.
if [[ ! -f "${MESH_ASSIGNMENTS}" ]]; then
  echo "ERROR: MESH_ASSIGNMENTS=${MESH_ASSIGNMENTS} not found on $(hostname)" >&2
  exit 1
fi
if [[ ! -f "${MESH_SHARD_MAP}" ]]; then
  echo "ERROR: MESH_SHARD_MAP=${MESH_SHARD_MAP} not found on $(hostname)" >&2
  exit 1
fi
if [[ ! -d "${MESH_ROOT}" ]]; then
  echo "ERROR: MESH_ROOT=${MESH_ROOT} not found on $(hostname) -- run seed_mesh.py before submitting workers" >&2
  exit 1
fi

# Node-local caches under /tmp, NOT the student home quota (~5 GB free): the
# cu128 torch wheels uv installs are multi-GB. Mirrors train.sbatch.
export UV_CACHE_DIR="/tmp/uv_cache_${SLURM_JOB_ID:-$$}"
export TORCH_HOME="/tmp/torch_home_${SLURM_JOB_ID:-$$}"
export TORCHINDUCTOR_CACHE_DIR="/tmp/torch_inductor_${SLURM_JOB_ID:-$$}"

# Checkpoints go to node-local /Data, not $HOME, for the same quota reason.
export STERN_CHECKPOINT_ROOT="${STERN_CHECKPOINT_ROOT:-/Data/$USER/stern_checkpoints}"
mkdir -p "${STERN_CHECKPOINT_ROOT}"
chmod 700 "${STERN_CHECKPOINT_ROOT}"

uv run python -m video_encoding.training.pretrain experiment="${EXPERIMENT}" "$@"
