#!/usr/bin/env bash
#SBATCH --job-name=t5_recipe_c
#SBATCH --partition=SallesInfo
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --time=45
#SBATCH --nodelist=node001,node002,node003,node004,node005,node006,node007,node008
#SBATCH --output=%x_%j.out
# T5 Recipe C synthetic benchmark: real V-JEPA2 + real LoRA + real DDP,
# synthetic-tensor inputs.  N=8 fish RTX3090 nodes, local_batch=4.
# See docs/superpowers/specs/2026-07-01-t5-real-recipe-c-benchmark-design.md
#
# Results written to: $HOME/stern/.claude/results/t5_recipe_c.jsonl (rank 0 only)

set -euo pipefail

# Use the venv python directly — uv run would trigger uv sync which reverts
# CUDA torch (uv.lock pins the CPU build; CUDA torch installed via uv pip install).
PYTHON="$HOME/stern/.venv/bin/python"

# Shared NFS HF cache — pre-downloaded once to avoid 8 simultaneous unauthenticated
# HF Hub downloads hitting rate limits (observed timeout at N=8 without pre-cache).
export HF_HOME="$HOME/stern/.hf_cache"

# DDP rendezvous via env:// (matches dist.init_process_group in bench_recipe_c.py)
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -1)
export MASTER_PORT=29501
export NCCL_SOCKET_IFNAME=enp0s31f6  # confirmed 1GbE interface from T3

cd "$HOME/stern"

echo "job=$SLURM_JOB_ID date=$(date -Is) master=$MASTER_ADDR nodes=$SLURM_NODELIST"
echo "concurrent_running_jobs: $(squeue -h -t running -o '%j' | tr '\n' ',')"

# srun sets SLURM_PROCID (0-7) and SLURM_NTASKS (8) automatically per task,
# which bench_recipe_c.py reads as DDP rank/world.
srun --ntasks=8 --ntasks-per-node=1 \
    "$PYTHON" video_encoding/scripts/bench_recipe_c.py \
        --batch-size 4 \
        --results-path .claude/results/t5_recipe_c.jsonl

echo "===ALLDONE==="
