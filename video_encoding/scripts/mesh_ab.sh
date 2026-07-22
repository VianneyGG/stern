#!/usr/bin/env bash
# video_encoding/scripts/mesh_ab.sh
# Convergence A/B gate for the decentralized mesh architecture -- run this
# BEFORE any long mesh training run or before video_encoding/scripts/
# sweep_mesh_outer.sh's constant tuning (Task 14). See
# docs/superpowers/plans/2026-07-16-decentralized-mesh-training.md Task 13.
#
# Why this exists: if a mesh arm does not at least match the single-node
# reference, no amount of infrastructure correctness saves the run, and
# tuning the outer optimizer's constants afterward is meaningless. This
# gate is cheap because outer_lr=1.0/outer_momentum=0.0 reproduces plain
# weighted averaging through the SAME run_outer_step.py code path used by
# ab-diloco (video_encoding/mesh/outer_step.py's naive-averaging
# equivalence) -- so ab-naive_avg and ab-diloco compare a hyperparameter
# setting, not two different codebases.
#
# Usage: mesh_ab.sh <experiment> <nodelist> [rounds]
#   <experiment>  name of a file in conf/experiment/ (without .yaml)
#   <nodelist>    comma-separated hostnames for the mesh arms; the mesh
#                 arms (ab-naive_avg, ab-diloco) each get one single-GPU
#                 job PER node in this list. The single-node reference
#                 arm (ab-single_node) gets exactly one job, on the first
#                 listed node -- it needs only one GPU by definition.
#   [rounds]      number of DiLoCo merge rounds for the mesh arms; default 10
set -euo pipefail

EXPERIMENT="${1:?usage: mesh_ab.sh <experiment> <nodelist> [rounds]}"
NODELIST="${2:?usage: mesh_ab.sh <experiment> <nodelist> [rounds]}"
ROUNDS="${3:-10}"

IFS=',' read -r -a NODES <<< "${NODELIST}"
REFERENCE_NODE="${NODES[0]}"

# Shared-home quota constraint (25.6/30 GB used, ~15-20 GB freeable): N x
# 172 MB (bf16 ViT-B checkpoint) + master + momentum buffer must fit.
MAX_MESH_NODES=16
if (( ${#NODES[@]} > MAX_MESH_NODES )); then
  echo "mesh_ab: ${#NODES[@]} node(s) exceeds MAX_MESH_NODES=${MAX_MESH_NODES} (shared home quota) -- reduce nodelist" >&2
  exit 1
fi

SBATCH_SCRIPT="video_encoding/slurm/mesh_train.sbatch"

echo "mesh_ab: experiment=${EXPERIMENT} nodelist=${NODELIST} rounds=${ROUNDS}"

# --- Arm 1: ab-single_node -- one GPU, no merge. The reference. ---
echo "mesh_ab: submitting ab-single_node on ${REFERENCE_NODE}"
sbatch --nodes=1 --ntasks=1 --nodelist="${REFERENCE_NODE}" \
  "${SBATCH_SCRIPT}" "${EXPERIMENT}" "${EXPERIMENT}-ab-single_node" \
  +run_tag=ab-single_node +mesh.merge=false

# --- Arm 2: ab-naive_avg -- mesh, outer_lr=1.0/outer_momentum=0.0 ---
# (the original design, routed through the DiLoCo code path's lr=1/momentum=0
# equivalence to plain weighted averaging -- see outer_step.py).
for node in "${NODES[@]}"; do
  echo "mesh_ab: submitting ab-naive_avg on ${node}"
  sbatch --nodes=1 --ntasks=1 --nodelist="${node}" \
    "${SBATCH_SCRIPT}" "${EXPERIMENT}" "${EXPERIMENT}-ab-naive_avg" \
    +run_tag=ab-naive_avg +mesh.outer_lr=1.0 +mesh.outer_momentum=0.0 \
    +mesh.rounds="${ROUNDS}"
done

# --- Arm 3: ab-diloco -- mesh, outer_lr=0.7/outer_momentum=0.9 ---
for node in "${NODES[@]}"; do
  echo "mesh_ab: submitting ab-diloco on ${node}"
  sbatch --nodes=1 --ntasks=1 --nodelist="${node}" \
    "${SBATCH_SCRIPT}" "${EXPERIMENT}" "${EXPERIMENT}-ab-diloco" \
    +run_tag=ab-diloco +mesh.outer_lr=0.7 +mesh.outer_momentum=0.9 \
    +mesh.rounds="${ROUNDS}"
done

cat <<'EOF'

mesh_ab: decision rule (join results on run_tag ab-single_node /
ab-naive_avg / ab-diloco via video_encoding/watcher/results.py):
  - ab-diloco's final reconstruction loss <= ab-single_node's -> proceed
    to Task 14's outer-constant sweep.
  - both mesh arms are much worse than ab-single_node -> STOP. Shorten H
    (mesh.merge_every_passes) once and re-run this gate; if that does not
    close the gap, the decentralized-mesh architecture is not a fit for
    this training recipe.
EOF
