#!/usr/bin/env bash
# video_encoding/slurm/train.sbatch
# Generic SLURM submission wrapper around scripts/run_experiment.sh.
# Submitted via submit_login-node.sh, which overrides --nodes/--ntasks/--nodelist
# at submission time via pick_nodes.sh. Directives here are fallback defaults.
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
#SBATCH --time=2-00:00:00
#SBATCH --signal=USR1@120
#SBATCH --output=video_encoding/slurm/logs/%x-%j.out

# --signal=USR1@120: SLURM sends SIGUSR1 two minutes before walltime.
# ssl_adapt's signal handler (video_encoding/ssl/__init__.py) catches it, finishes the
# current epoch, writes the latest checkpoint, and exits cleanly — instead of
# getting SIGKILL'd mid-epoch and losing whatever the current epoch produced.

set -euo pipefail

EXPERIMENT="${1:?usage: sbatch train.sbatch <experiment-name> [hydra overrides...]}"
shift

# DDP rendezvous env vars. All ranks must agree on MASTER_ADDR/MASTER_PORT
# BEFORE torch.distributed.init_process_group runs (src/training/ddp.py:setup_ddp).
# Port derived from SLURM_JOB_ID to avoid collisions between concurrent jobs
# sharing a login node's port namespace.
export MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)"
export MASTER_PORT="$((15000 + SLURM_JOB_ID % 5000))"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
srun bash "scripts/run_experiment.sh" "${EXPERIMENT}" "$@"
