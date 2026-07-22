#!/usr/bin/env bash
# video_encoding/scripts/sweep_mesh_outer.sh
# Outer-optimizer constant sweep for the decentralized mesh -- RUN ONLY
# AFTER video_encoding/scripts/mesh_ab.sh (Task 13) shows a mesh arm
# matching single-node. Tuning constants on an architecture that does not
# converge is wasted cluster time. See
# docs/superpowers/plans/2026-07-16-decentralized-mesh-training.md Task 14.
#
# Why this exists: DiLoCo's outer_lr=0.7/momentum=0.9/H=500 (arXiv
# 2311.08105, 2407.07852) are tuned on C4 transformer LMs. The method is
# architecture-agnostic -- it optimizes a parameter delta -- but the
# constants are not, and VideoMAE's loss landscape is not C4's.
#
# Usage: sweep_mesh_outer.sh <experiment> <nodelist> [rounds]
#   <experiment>  name of a file in conf/experiment/ (without .yaml)
#   <nodelist>    comma-separated hostnames; every grid cell gets one
#                 single-GPU job PER node in this list
#   [rounds]      number of DiLoCo merge rounds per cell; default 10
#
# Expected shape if the DiLoCo-on-LMs literature transfers to VideoMAE:
# momentum=0.9 beats momentum=0.0 at equal lr, lr=0.7 is near-optimal, and
# lr=1.0/momentum=0.0 (the naive-averaging anchor) is the floor. A null
# result is worth recording, not discarding: if momentum=0.9 does not beat
# 0.0 here, the outer-momentum assumption does not transfer from LMs to
# video MAE, and the 172 MB momentum buffer persisted every round is not
# earning its place.
set -euo pipefail

EXPERIMENT="${1:?usage: sweep_mesh_outer.sh <experiment> <nodelist> [rounds]}"
NODELIST="${2:?usage: sweep_mesh_outer.sh <experiment> <nodelist> [rounds]}"
ROUNDS="${3:-10}"

IFS=',' read -r -a NODES <<< "${NODELIST}"

# Shared-home quota constraint (25.6/30 GB used, ~15-20 GB freeable): N x
# 172 MB (bf16 ViT-B checkpoint) + master + momentum buffer must fit.
MAX_MESH_NODES=16
if (( ${#NODES[@]} > MAX_MESH_NODES )); then
  echo "sweep_mesh_outer: ${#NODES[@]} node(s) exceeds MAX_MESH_NODES=${MAX_MESH_NODES} (shared home quota) -- reduce nodelist" >&2
  exit 1
fi

SBATCH_SCRIPT="video_encoding/slurm/mesh_train.sbatch"

# Deliberately small grid -- each cell is a full short run (18 cells total).
OUTER_LRS=(0.3 0.7 1.0)
OUTER_MOMENTUMS=(0.0 0.9)
MERGE_EVERY=(0.5 1.0 2.0)

echo "sweep_mesh_outer: experiment=${EXPERIMENT} nodelist=${NODELIST} rounds=${ROUNDS}"
echo "sweep_mesh_outer: grid = ${#OUTER_LRS[@]} lr x ${#OUTER_MOMENTUMS[@]} momentum x ${#MERGE_EVERY[@]} merge_every = $((${#OUTER_LRS[@]} * ${#OUTER_MOMENTUMS[@]} * ${#MERGE_EVERY[@]})) cells"

for lr in "${OUTER_LRS[@]}"; do
  for mom in "${OUTER_MOMENTUMS[@]}"; do
    for every in "${MERGE_EVERY[@]}"; do
      run_tag="outer-lr${lr}-mom${mom}-H${every}"
      run_id="${EXPERIMENT}-${run_tag}"
      for node in "${NODES[@]}"; do
        echo "sweep_mesh_outer: submitting ${run_tag} on ${node}"
        sbatch --nodes=1 --ntasks=1 --nodelist="${node}" \
          "${SBATCH_SCRIPT}" "${EXPERIMENT}" "${run_id}" \
          "+run_tag=${run_tag}" \
          "+mesh.outer_lr=${lr}" "+mesh.outer_momentum=${mom}" \
          "+mesh.merge_every_passes=${every}" "+mesh.rounds=${ROUNDS}"
      done
    done
  done
done

cat <<'EOF'

sweep_mesh_outer: join results on run_tag outer-lr${lr}-mom${mom}-H${every}
via video_encoding/watcher/results.py. Compare cells on final SSL
reconstruction loss over the REPLICATED val subset only (identical across
nodes by construction -- a partitioned val set would make the numbers
incomparable).

Expected shape if the literature transfers: momentum=0.9 beats momentum=0.0
at equal lr; lr=0.7 is near-optimal; lr=1.0/momentum=0.0 (the naive-averaging
anchor) is the floor every other cell is measured against. A null result
(momentum=0.9 not beating 0.0) is itself the finding: the LM-tuned
outer-momentum assumption would not transfer to VideoMAE, and the persisted
172 MB momentum buffer would not be earning its place.
EOF
