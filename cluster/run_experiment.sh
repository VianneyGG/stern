#!/usr/bin/env bash
# cluster/run_experiment.sh
# Single dispatch: `run_experiment.sh <experiment-name> [hydra overrides...]`.
# Routes to stern.training.pretrain (stage: pretrain) or stern.training.finetune (stage: va_finetune/default).
# Experiment names follow <encoder>_<strategy>[_<variant>]_<dataset>.
set -euo pipefail

EXPERIMENT="${1:?usage: run_experiment.sh <experiment-name> [hydra overrides...]}"
shift

# Run from the repo root so the top-level packages (stern and its subpackages)
# resolve via pyproject's pythonpath=["."].
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Determine the target module by querying the experiment config's `stage` field.
# Default to 'va_finetune' (probe training) if stage is unset.
STAGE=$(uv run python -c "
from hydra import initialize_config_dir, compose
from pathlib import Path
cfg_dir = Path('${REPO_ROOT}/configs').resolve()
with initialize_config_dir(config_dir=str(cfg_dir), version_base=None):
    cfg = compose(config_name='config', overrides=['experiment=${EXPERIMENT}'])
    print(cfg.experiment.get('stage', 'va_finetune'))
" 2>/dev/null || echo "va_finetune")

case "$STAGE" in
  pretrain)
    # Each pretrain-stage experiment config selects its own dataset and train
    # group through its `override /dataset:` and `override /train:` defaults,
    # so no shell-level flags are needed. "$@" still lets a caller override further.
    uv run python -m stern.training.pretrain experiment="${EXPERIMENT}" "$@"
    ;;
  va_finetune|*)
    uv run python -m stern.training.finetune experiment="${EXPERIMENT}" "$@"
    ;;
esac
