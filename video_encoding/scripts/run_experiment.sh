#!/usr/bin/env bash
# video_encoding/scripts/run_experiment.sh
# Single dispatch: `run_experiment.sh <experiment-name> [hydra overrides...]`.
# Replaces the old per-letter wrappers (run_recipe_{a,b,c}.sh) — experiment
# names now follow the WHN convention <encoder>_<strategy>[_<variant>]_<dataset>,
# see conf/experiment/ and conf/CONFIGS.md.
set -euo pipefail

EXPERIMENT="${1:?usage: run_experiment.sh <experiment-name> [hydra overrides...]}"
shift

# Run from the repo root so the flat top-level packages (video_encoding, data,
# eval, cache) all resolve via pyproject's pythonpath=["."]. No cwd/PYTHONPATH
# hack needed anymore — the old `src.*` layout that required it is gone.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
uv run python -m video_encoding.probe.train experiment="${EXPERIMENT}" "$@"
