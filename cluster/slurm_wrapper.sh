#!/usr/bin/env bash
# cluster/slurm_wrapper.sh
# Minimal allowlisted gateway to SLURM commands for the training watcher's
# escalation call. The escalation `claude -p` call gets Bash access scoped to
# this script only, and this script forwards to exactly four SLURM binaries,
# so an escalated call cannot shell out to anything else.
#
# Usage: slurm_wrapper.sh <sbatch|scancel|squeue|sacct> [args...]
set -euo pipefail

CMD="${1:?usage: slurm_wrapper.sh <sbatch|scancel|squeue|sacct> [args...]}"
shift

case "${CMD}" in
  sbatch|scancel|squeue|sacct)
    # Class-hours gate (class_hours.py): no job may run Mon-Fri 08:00-20:00
    # Europe/Paris. Only sbatch launches anything.
    if [[ "${CMD}" == "sbatch" ]]; then
      python3 "$(dirname "${BASH_SOURCE[0]}")/../stern/watcher/class_hours.py" || exit 1
    fi
    exec "${CMD}" "$@"
    ;;
  *)
    echo "slurm_wrapper.sh: command '${CMD}' not allowed (only sbatch/scancel/squeue/sacct)" >&2
    exit 2
    ;;
esac
