#!/usr/bin/env bash
# video_encoding/slurm/mesh_train.sbatch
# SLURM submission wrapper for ONE worker in the decentralized P2P training
# mesh (video_encoding/scripts/run_mesh_worker.sh).
#
# Unlike train.sbatch, this is deliberately NOT a DDP job: single node,
# single task, no distributed rendezvous env vars, no srun, no in-job
# retry loop. Each worker trains its own shard subset independently; a
# missing/failed worker is detected and its shards reassigned by the
# watcher (video_encoding/watcher/epoch_advance.py), not retried here -- an
# in-job retry would hide the failure from round bookkeeping.
#
# Cluster facts (T1.1/T1.2/T1.7): single partition SallesInfo, no GPU GRES
# (whole-node allocation -- request nodes not gres), 16 cores/node, ~63 GB
# RAM, no preemption. Max walltime is 2h: the cluster was rebuilt on
# 2026-08-22 (SLURM 26.05.2, job counter reset, MaxNodes=5) and the new
# slurm.conf sets SallesInfo MaxTime=2:00:00, down from 3 days. SLURM does
# not reject an over-limit --time; it parks the job as PENDING
# (PartitionTimeLimit) forever, so exceeding this is silent -- 64 jobs sat
# unschedulable while the watcher logged them as RUNNING.
#
# Usage: sbatch mesh_train.sbatch <experiment-name> <run-id> [overrides...]
#SBATCH --job-name=stern-mesh-worker
#SBATCH --partition=SallesInfo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --exclusive
# --exclusive is mandatory here, not a tunable default: this cluster does
# NOT enforce GPU isolation between SLURM jobs, so without it another
# student's job can land on the same node and OOM the shared GPU (the same
# failure mode train.sbatch guards against, observed in the wild).
# A worker no longer spans a whole run: at ~1000s/round a 40-round run is
# ~11h, so it is chunked into successive jobs (mesh.chunk_rounds) that each
# exit cleanly under this cap and resume via +mesh.start_round + master.pt.
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@180
# --requeue is a genuine safety net (node failure, not just walltime), but
# it competes with the watcher's own chunk relaunch for the same
# --exclusive node whenever a worker hits its 2h cap instead of exiting
# early at the chunk boundary -- a self-requeued job goes back to PENDING
# under the same job id, requesting the same node. The watcher reconciles
# this itself before resubmitting (chunk_relaunch.stale_jobs_for_nodes,
# scancel'd in mesh_tick._relaunch_chunk_if_finished) rather than dropping
# the flag, so a true node failure still gets SLURM's own retry
# (observed 2026-09-07, lev-fix16c).
#
# Separate from that: run_mesh_worker.sh also self-chains on a CLEAN exit,
# via its own explicit `scontrol requeue` (not this flag). That loop is
# bounded to the job's chunk_rounds by MESH_MAX_REQUEUES
# (mesh_tick._submit_worker_job -> mesh_continue_allowed.sh Layer 2) so it
# stops at the chunk boundary instead of racing the watcher's next-chunk
# submission for the same node (same 2026-09-07 incident).
#SBATCH --requeue
#SBATCH --output=video_encoding/slurm/logs/%x-%j.out

set -euo pipefail

# Class-hours gate (video_encoding/scripts/class_hours.py): no job may run
# Mon-Fri 08:00-20:00 Europe/Paris. Checked against this job's REMAINING
# walltime, so it also stops a --requeue / scontrol requeue restart that
# SLURM schedules on its own. Exits 0 so nothing retries it; an unreadable
# time limit falls back to the 2h partition max (fails closed).
python3 "${SLURM_SUBMIT_DIR:-$HOME/stern}/video_encoding/scripts/class_hours.py" \
  --walltime "$(squeue -h -j "${SLURM_JOB_ID:-0}" -o %L 2>/dev/null || true)" || exit 0

# SLURM executes the batch script from a spool copy, so BASH_SOURCE is
# unreliable here. Use SLURM_SUBMIT_DIR (the repo root ~/stern that the
# submitter ran `sbatch` from), matching train.sbatch's convention;
# run_mesh_worker.sh re-cds to the repo root itself as well.
cd "${SLURM_SUBMIT_DIR:-$HOME/stern}"

bash "video_encoding/scripts/run_mesh_worker.sh" "$@"
