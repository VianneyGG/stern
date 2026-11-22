#!/usr/bin/env bash
# cluster/run_mesh_worker.sh
# Single-GPU worker entrypoint for the decentralized P2P training mesh.
#
# Unlike train.sbatch (DDP, --nodes=8, rendezvous, retry loop), each mesh
# worker is an independent single-process trainer on its own shard subset:
# no MASTER_ADDR/MASTER_PORT, no srun, no in-job retry. A dead worker is
# excluded and its shards reassigned by the watcher
# (stern/watcher/epoch_advance.py's missed-round threshold), not
# retried here -- an in-job retry would hide the failure from round
# bookkeeping instead of surfacing a missed round for the next merge.
#
# Usage: run_mesh_worker.sh <experiment> <run-id> [hydra overrides...]
set -euo pipefail

EXPERIMENT="${1:?usage: run_mesh_worker.sh <experiment> <run-id> [hydra overrides...]}"
RUN_ID="${2:?usage: run_mesh_worker.sh <experiment> <run-id> [hydra overrides...]}"
shift 2

source "$(dirname "${BASH_SOURCE[0]}")/cluster_env.sh"

# Run from the repo root, same resolution pattern as run_experiment.sh.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# sbatch propagates the submitter's environment by default (--export=ALL).
# When the watcher submits this job from cron, cron's own PATH
# (/usr/bin:/bin) doesn't include ~/.local/bin, so an inherited-PATH-only
# `uv run` below fails instantly with "uv: command not found" regardless of
# who submitted the job. Make this script self-sufficient rather than
# depending on the submitter having a correct PATH.
export PATH="${HOME}/.local/bin:${PATH}"

export MESH_RUN_ID="${RUN_ID}"
# Deliberately OUTSIDE ~/stern: sync_to_cluster.sh pushes source with
# `rsync --delete`, which would wipe round-local mesh state (assignments,
# shard map, pending checkpoints) if it lived under the synced repo tree.
export MESH_SHARED="${MESH_SHARED:-${HOME}/stern_mesh}"
export MESH_ROOT="${MESH_ROOT:-${STERN_DATA_ROOT}/mesh}"
export MESH_ASSIGNMENTS="${MESH_ASSIGNMENTS:-${MESH_SHARED}/metadata/shard_assignments_${MESH_RUN_ID}.json}"
export MESH_SHARD_MAP="${MESH_SHARD_MAP:-${MESH_SHARED}/metadata/shard_map.json}"

# Fail fast, before the trainer's dataloader would discover any of this deep
# inside its first batch. A straggler waiting on data that never arrives can
# hang a DDP job for close to an hour (see train.sbatch), so catch it here.
if [[ ! -f "${MESH_ASSIGNMENTS}" ]]; then
  echo "ERROR: MESH_ASSIGNMENTS=${MESH_ASSIGNMENTS} not found on $(hostname)" >&2
  exit 1
fi
if [[ ! -f "${MESH_SHARD_MAP}" ]]; then
  echo "ERROR: MESH_SHARD_MAP=${MESH_SHARD_MAP} not found on $(hostname)" >&2
  exit 1
fi
if [[ ! -d "${MESH_ROOT}" ]]; then
  echo "ERROR: MESH_ROOT=${MESH_ROOT} not found on $(hostname) -- run python -m stern.mesh.seed_nodes before submitting workers" >&2
  exit 1
fi

# val/manifest.json can be missing for a moment after seeding even though the
# seeding step has written it: an NFS/automount visibility race on freshly
# seeded nodes, not missing data. Retry briefly instead of failing fast.
_VAL_MANIFEST="${MESH_ROOT}/val/manifest.json"
for _i in $(seq 1 10); do
  [[ -f "${_VAL_MANIFEST}" ]] && break
  sleep 3
done
if [[ ! -f "${_VAL_MANIFEST}" ]]; then
  echo "ERROR: ${_VAL_MANIFEST} not found on $(hostname) after 30s -- run python -m stern.mesh.seed_nodes before submitting workers" >&2
  exit 1
fi

# Node-local caches under /tmp, NOT the NFS home quota, which is small: the
# cu128 torch wheels uv installs are multi-GB. Mirrors train.sbatch.
export UV_CACHE_DIR="/tmp/uv_cache_${SLURM_JOB_ID:-$$}"
export TORCH_HOME="/tmp/torch_home_${SLURM_JOB_ID:-$$}"
export TORCHINDUCTOR_CACHE_DIR="/tmp/torch_inductor_${SLURM_JOB_ID:-$$}"

# Checkpoints go to node-local /Data, not $HOME, for the same quota reason.
export STERN_CHECKPOINT_ROOT="${STERN_CHECKPOINT_ROOT:-${STERN_DATA_ROOT}/stern_checkpoints}"
mkdir -p "${STERN_CHECKPOINT_ROOT}"
chmod 700 "${STERN_CHECKPOINT_ROOT}"

# --no-sync: the project venv lives on the NFS home (9.4 GB, ~53k files) and
# `uv sync` reinstalls into it unconditionally -- `uv sync --dry-run` reports
# an uninstall+install even against an up-to-date lockfile. Without this,
# every worker in the pool re-downloads and rewrites the same package into
# the same shared NFS venv at launch, all at once. The venv must already be
# provisioned (launch_mesh_pool.sh runs `uv sync` once, from the login node).
export MESH_STOP_FILE="${MESH_STOP_FILE:-${MESH_SHARED}/state/stop/${MESH_RUN_ID}-$(hostname -s)}"
mkdir -p "$(dirname "${MESH_STOP_FILE}")"

_on_usr1() {
  touch "${MESH_STOP_FILE}"
}
trap _on_usr1 USR1

uv run --no-sync python -m stern.training.pretrain \
  experiment="${EXPERIMENT}" "$@" &
TRAINER_PID=$!
wait "${TRAINER_PID}"
TRAINER_EXIT=$?

if [[ "${TRAINER_EXIT}" -ne 0 ]]; then
  exit "${TRAINER_EXIT}"
fi

if bash "${REPO_ROOT}/cluster/mesh_continue_allowed.sh" "${MESH_RUN_ID}" "$(hostname -s)"; then
  scontrol requeue "${SLURM_JOB_ID}"
  exit 0
fi
exit 0
