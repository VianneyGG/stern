#!/usr/bin/env bash
# video_encoding/slurm/probe_bakeoff.sbatch
# One arm of the AFEW-VA frozen-probe bake-off (scripts/probe_bakeoff.py).
#
# Usage: sbatch [--nodelist=NODE] probe_bakeoff.sbatch <run_id> [layers] [extra args...]
#
# One arm per job on purpose: arms are independent, each writes its own report,
# and a node dying costs one arm rather than the sweep. Single node, no DDP --
# the probe is a frozen-encoder forward pass plus a small fit, not training.
#SBATCH --job-name=stern-probe-bakeoff
#SBATCH --partition=SallesInfo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --exclusive
# --exclusive: SLURM here does not enforce GPU isolation, so a co-scheduled job
# takes VRAM out from under this one (see train.sbatch).
#SBATCH --time=01:50:00
#SBATCH --output=video_encoding/slurm/logs/%x-%j.out

set -euo pipefail

# Class-hours gate: no job may run Mon-Fri 08:00-20:00 Europe/Paris. Exits 0 so
# nothing retries it. Same contract as train.sbatch.
python3 "${SLURM_SUBMIT_DIR:-$HOME/stern}/video_encoding/scripts/class_hours.py" \
  --walltime "$(squeue -h -j "${SLURM_JOB_ID:-0}" -o %L 2>/dev/null || true)" || exit 0

RUN_ID="${1:?usage: sbatch probe_bakeoff.sbatch <run_id> [layers] [extra args...]}"
shift
LAYERS="${1:-2,4,6,9,12}"
if [[ $# -gt 0 ]]; then shift; fi

# Caches on node-local /tmp, never the 30 GB student home quota (train.sbatch).
export UV_CACHE_DIR="/tmp/uv_cache_${SLURM_JOB_ID}"
export TORCH_HOME="/tmp/torch_home_${SLURM_JOB_ID}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# The CLI prints its table only at the end, so without this a job that is
# materialising tokens for 20 minutes is indistinguishable in the log from one
# that died before producing a line.
export PYTHONUNBUFFERED=1

AFEWVA_ROOT="${AFEWVA_ROOT:-/Data/$USER/afewva/raw}"
MANIFEST="${PROBE_MANIFEST:-${AFEWVA_ROOT}/manifest.csv}"
if [[ ! -f "${MANIFEST}" ]]; then
  echo "ERROR: no probe manifest at ${MANIFEST} on $(hostname) -- AFEW-VA is staged" \
       "on login-node only (see scripts/afewva_staging_notes.md)." >&2
  exit 1
fi

cd "${SLURM_SUBMIT_DIR:-$HOME/stern}"
OUT_DIR="runs/results/probe"
mkdir -p "${OUT_DIR}"
OUT="${OUT_DIR}/${RUN_ID}-L${LAYERS//,/_}.json"

# Absolute uv path and `-m`: cron/SLURM PATH lacks ~/.local/bin, and there is no
# editable install, so sys.path[0] must be the repo root. --no-sync: the venv is
# 9.4 GB on NFS (conf/CONFIGS.md).
~/.local/bin/uv run --no-sync python -m video_encoding.scripts.probe_bakeoff \
  --run-id "${RUN_ID}" \
  --manifest "${MANIFEST}" \
  --layers "${LAYERS}" \
  --out "${OUT}" \
  "$@"
