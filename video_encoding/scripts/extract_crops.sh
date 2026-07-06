#!/usr/bin/env bash
# video_encoding/scripts/extract_crops.sh
# Crop + landmark extraction is CPU-bound: spray across ALL rooms incl. slow /252.
# Always exclude `node114` (rivers-named node that silently sits on /252, 10-11ms
# cross-subnet path) — see REPO_SETUP.md §6.
set -euo pipefail
echo "TODO(human): wire this to the cluster's actual job submission command" \
     "once sbatch_filtered.sh (Task 6 step 5) and crop output paths are confirmed." >&2
exit 1
