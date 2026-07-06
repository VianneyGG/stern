#!/usr/bin/env bash
# video_encoding/scripts/cron/aggregate_results.sh
# Synthesize .claude/results/*.json into a sorted summary.tsv via a headless
# Haiku call, so the interactive (Opus) session reads one small file instead
# of re-parsing every result JSON on every "how are runs doing" ask. Not
# installed to crontab yet; see conf/CONFIGS.md for the rationale and the
# guardrails below (flock, --max-turns, scoped tools, atomic write).
#
# Usage: scripts/cron/aggregate_results.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
STATE_DIR="${REPO_ROOT}/.claude/state"
RESULTS_DIR="${REPO_ROOT}/.claude/results"
LOG="${STATE_DIR}/cron.log"
LOCK="${STATE_DIR}/aggregate_results.lock"
mkdir -p "${STATE_DIR}" "${RESULTS_DIR}"

exec 200>"${LOCK}"
if ! flock -n 200; then
  echo "$(date -Iseconds) aggregate_results skip: previous run still holding lock" >> "${LOG}"
  exit 0
fi

if ! command -v claude >/dev/null 2>&1; then
  echo "$(date -Iseconds) aggregate_results FAILED: claude not on PATH" >> "${LOG}"
  exit 0
fi

if [ -z "$(find "${RESULTS_DIR}" -maxdepth 1 -name '*.json' -print -quit 2>/dev/null)" ]; then
  echo "$(date -Iseconds) aggregate_results skip: no result JSONs yet" >> "${LOG}"
  exit 0
fi

TMP="${STATE_DIR}/summary.tsv.tmp"
OUT="${STATE_DIR}/summary.tsv"

# No Write grant: the sub-agent only ever reads result JSONs (which embed
# operator-controlled free-text like run_tag/config) and prints the TSV to
# stdout. This script -- not the sub-agent -- owns the only path it can
# write to, closing off any path-unscoped-write risk from prompt injection
# via result-JSON content.
PROMPT="Read every *.json file directly under ${RESULTS_DIR} (WHN-schema \
per-run result files). Build a TSV with header line: \
experiment\tbest_val\tflops_fwd\tepoch_reached\tgit_sha \
-- best_val from metrics.best_val_recon_loss or metrics.best_val_metric \
(whichever key is present), flops_fwd from cost.flops_fwd_per_clip, \
epoch_reached from metrics.epochs_trained, git_sha from env.git_sha. \
Sort rows by best_val ascending. Print ONLY the TSV to stdout -- no \
commentary, no code fences, no file writes."

if OUTPUT=$(claude -p --model haiku --max-turns 3 \
    --allowedTools "Read,Glob" \
    "${PROMPT}" 2>>"${LOG}"); then
  if [ -n "${OUTPUT}" ]; then
    printf '%s\n' "${OUTPUT}" > "${TMP}"
    mv "${TMP}" "${OUT}"
    echo "$(date -Iseconds) aggregate_results ok" >> "${LOG}"
  else
    echo "$(date -Iseconds) aggregate_results FAILED: empty output" >> "${LOG}"
  fi
else
  echo "$(date -Iseconds) aggregate_results FAILED: claude exited nonzero" >> "${LOG}"
  rm -f "${TMP}"
fi
