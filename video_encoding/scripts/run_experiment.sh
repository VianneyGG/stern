#!/usr/bin/env bash
# video_encoding/scripts/run_experiment.sh
# Single dispatch: `run_experiment.sh <experiment-name> [hydra overrides...]`.
# Routes to video_encoding.training.pretrain (stage: pretrain) or video_encoding.training.train (stage: va_finetune/default).
# Experiment names follow WHN: <encoder>_<strategy>[_<variant>]_<dataset>.
set -euo pipefail

EXPERIMENT="${1:?usage: run_experiment.sh <experiment-name> [hydra overrides...]}"
shift

# Run from the repo root so the flat top-level packages (video_encoding, data,
# eval, cache) all resolve via pyproject's pythonpath=["."]. No cwd/PYTHONPATH
# hack needed anymore — the old `src.*` layout that required it is gone.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# Determine the target module by querying the experiment config's `stage` field.
# Default to 'va_finetune' (probe training) if stage is unset.
STAGE=$(uv run python -c "
from hydra import initialize_config_dir, compose
from pathlib import Path
cfg_dir = Path('${REPO_ROOT}/video_encoding/conf').resolve()
with initialize_config_dir(config_dir=str(cfg_dir), version_base=None):
    cfg = compose(config_name='config', overrides=['experiment=${EXPERIMENT}'])
    print(cfg.experiment.get('stage', 'va_finetune'))
" 2>/dev/null || echo "va_finetune")

case "$STAGE" in
  pretrain)
    # Each pretrain-stage experiment config now self-selects its dataset and
    # train group via its own `override /dataset:`/`override /train:`
    # defaults (see conf/experiment/videomae_ssl_pretrain_voxmini.yaml) --
    # no shell-level flags needed. "$@" still lets a caller override further.
    uv run python -m video_encoding.training.pretrain experiment="${EXPERIMENT}" "$@"
    ;;
  va_finetune|*)
    uv run python -m video_encoding.training.train experiment="${EXPERIMENT}" "$@"
    ;;
esac
