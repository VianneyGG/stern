#!/usr/bin/env bash
# video_encoding/scripts/slurm_class_hours_shim.sh
# Installed as ~/.local/bin/{sbatch,srun,salloc} (symlinks to this file) on
# the shared cluster home, ahead of /usr/bin on PATH, so a bare `sbatch` typed
# by hand, by a one-off `--wrap` readout or by the watcher's escalation agent
# goes through the class-hours gate (class_hours.py): no job may run Mon-Fri
# 08:00-20:00 Europe/Paris. No override.
#
# srun/salloc inside an existing allocation are job steps of a job that was
# already gated at its own start (see the *.sbatch files), so they pass.
# sbatch from inside a job is a NEW job and is gated like any other.
set -euo pipefail
name="$(basename "$0")"
real="/usr/bin/${name}"
if [[ "${name}" != "sbatch" && -n "${SLURM_JOB_ID:-}" ]]; then
  exec "${real}" "$@"
fi
python3 "$(dirname "$(readlink -f "$0")")/class_hours.py" || exit 1
exec "${real}" "$@"
