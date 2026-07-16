#!/usr/bin/env bash
# video_encoding/scripts/slurm_wrapper.sh
# Minimal allowlisted gateway to SLURM commands for the training watcher's
# escalation call (docs/superpowers/specs/2026-07-15-slurm-training-watcher-design.md
# §3/§5). The escalation `claude -p` invocation gets Bash access scoped to
# ONLY this script (not general shell) -- this script in turn only forwards
# to exactly these four SLURM binaries, so an escalated call cannot shell out
# to anything else even if it tried.
#
# Usage: slurm_wrapper.sh <sbatch|scancel|squeue|sacct> [args...]
set -euo pipefail

CMD="${1:?usage: slurm_wrapper.sh <sbatch|scancel|squeue|sacct> [args...]}"
shift

case "${CMD}" in
  sbatch|scancel|squeue|sacct)
    exec "${CMD}" "$@"
    ;;
  *)
    echo "slurm_wrapper.sh: command '${CMD}' not allowed (only sbatch/scancel/squeue/sacct)" >&2
    exit 2
    ;;
esac
