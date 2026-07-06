#!/usr/bin/env bash
# video_encoding/scripts/cache_embeddings.sh
# Frozen inference / embedding extraction: GPU, no allreduce, embarrassingly
# parallel -> fish + birds nodes for throughput. See REPO_SETUP.md §6.
set -euo pipefail
echo "TODO(human): wire this to the cluster's actual job submission command" \
     "once an encoder implementation (Task 5 step 7) exists to run." >&2
exit 1
