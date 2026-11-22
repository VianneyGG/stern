#!/usr/bin/env bash
# cluster/sweep_ssl.sh
#
# Drives the SSL scaling + hyperparameter sweep by emitting one
# submit_login-node.sh call per grid cell, each tagged with +run_tag=<cell> so
# aggregate_sweep.py can join results back to their config.
#
# Design: one-factor-at-a-time (OFAT) around a fixed baseline, not full
# factorial — throughput/loss surfaces are smooth, so this grids every axis at
# a fraction of the run count. See docs/superpowers/plans/2026-07-10-ssl-scaling-sweep.md.
#
# Usage:
#   scripts/sweep_ssl.sh [--dry-run] <phase>
#     <phase>      1 = throughput/scaling benchmark (short, max_steps=200)
#                  2 = convergence quality (long, at the Phase-1 winner config)
#     --dry-run    print the submit commands instead of running them
#
# Phase-2 operating point is read from env vars. Post-Phase-1b default is the
# single-node winner (N=1/B=16 sustains ~121 samp/s, 2x the best multi-node
# cell on this 1GbE fabric -- see build_phase2 comment):
#   WIN_N=1 WIN_B=16 WIN_AMP=bf16 WIN_ACCUM=16 WIN_WORKERS=8
set -euo pipefail

EXPERIMENT="videomae_ssl_pretrain_voxmini"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT="${SCRIPT_DIR}/submit_login-node.sh"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=1; shift; fi
PHASE="${1:?usage: sweep_ssl.sh [--dry-run] <phase 1|2>}"

# One row = "<N>|<hydra overrides>|<run_tag>".
declare -a ROWS=()

submit_row() {
  local n="$1" overrides="$2" tag="$3"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "submit_login-node.sh ${EXPERIMENT} ${n} ${overrides} +run_tag=${tag}"
  else
    echo ">>> [${tag}] N=${n} ${overrides}"
    "${SUBMIT}" "${EXPERIMENT}" "${n}" ${overrides} "+run_tag=${tag}"
  fi
}

build_phase1() {
  # voxmini (VoxCeleb2 subset) frames are staged on 21 of the 23 fish nodes
  # (all but node006), so the full fish tier is available -- node axis goes
  # up to NB=8 as in the original plan (override with SWEEP_MAXN).
  local NB="${SWEEP_MAXN:-8}"
  local NODES="${SWEEP_NODES:-1 2 4 8}"
  # Baseline C0: N=${NB} B=4 A=1 amp=bf16 compress=fp16 bucket=25 workers=8
  local base="train.max_steps=200 train.epochs=1"

  # S1: nodes x local batch (find scaling efficiency + max stable batch)
  for n in ${NODES}; do
    for b in 1 2 4; do
      ROWS+=("${n}|${base} train.batch_size=${b} train.accum_steps=1 train.amp=bf16 train.ddp.grad_compress=fp16 train.ddp.bucket_cap_mb=25 train.num_workers=8|p1-n${n}-b${b}")
    done
  done
  # S2: accumulation at the strong cell (N=NB,B=4) — does amortizing all-reduce help?
  for a in 2 4 8; do
    ROWS+=("${NB}|${base} train.batch_size=4 train.accum_steps=${a} train.amp=bf16 train.ddp.grad_compress=fp16 train.ddp.bucket_cap_mb=25 train.num_workers=8|p1-accum${a}")
  done
  # S3: precision (bf16 is C0; sweep off + fp16)
  for amp in off fp16; do
    ROWS+=("${NB}|${base} train.batch_size=4 train.accum_steps=1 train.amp=${amp} train.ddp.grad_compress=fp16 train.ddp.bucket_cap_mb=25 train.num_workers=8|p1-amp-${amp}")
  done
  # S4: comm — grad compression x bucket size (C0 is fp16/25, skip it).
  # "off" = any non-fp16 value disables the hook (see wrap_ddp); avoids the
  # empty-string quoting hazard through printf %q -> srun -> Hydra.
  for compress in off fp16; do
    for bucket in 25 50 100; do
      [[ "${compress}" == "fp16" && "${bucket}" == "25" ]] && continue
      ROWS+=("${NB}|${base} train.batch_size=4 train.accum_steps=1 train.amp=bf16 train.ddp.grad_compress=${compress} train.ddp.bucket_cap_mb=${bucket} train.num_workers=8|p1-c${compress}-bkt${bucket}")
    done
  done
  # S5: dataloader workers (is the JPEG-cache pipeline starving the GPU?)
  for w in 4 16; do
    ROWS+=("${NB}|${base} train.batch_size=4 train.accum_steps=1 train.amp=bf16 train.ddp.grad_compress=fp16 train.ddp.bucket_cap_mb=25 train.num_workers=${w}|p1-w${w}")
  done
  # S6: local batch ceiling — B=4 only used ~2.8/24GB VRAM (S1), never pushed
  # further. Bigger physical batch does more useful work per sync than
  # accum_steps at the same effective-batch cost (one conv/attn call instead of
  # N sequential ones), so this is a cheap, untested throughput lever at both
  # N=1 (compute-only ceiling) and N=${NB} (does it close the comm gap like
  # accum did?).
  for b in 8 16; do
    ROWS+=("1|${base} train.batch_size=${b} train.accum_steps=1 train.amp=bf16 train.ddp.grad_compress=fp16 train.ddp.bucket_cap_mb=25 train.num_workers=8|p1-n1-b${b}")
    ROWS+=("${NB}|${base} train.batch_size=${b} train.accum_steps=1 train.amp=bf16 train.ddp.grad_compress=fp16 train.ddp.bucket_cap_mb=25 train.num_workers=8|p1-n${NB}-b${b}")
  done
  # S7: node-count curve at a FIXED effective batch (256, matching p1-accum8's
  # N=8,B=4,accum=8 cell) — S2 only ever tested accum at N=${NB}, so we don't
  # know whether fewer nodes at proportionally higher accum reaches similar
  # throughput with a smaller crash blast-radius (the whole motivation: a
  # single flaky node killed the entire N=8 Phase-2 job overnight). accum=64
  # at N=1 here is a comm-free control, not a realistic operating point.
  for n in 1 2 4; do
    local a=$(( 256 / (n * 4) ))
    ROWS+=("${n}|${base} train.batch_size=4 train.accum_steps=${a} train.amp=bf16 train.ddp.grad_compress=fp16 train.ddp.bucket_cap_mb=25 train.num_workers=8|p1-effb256-n${n}")
  done
}

build_phase2() {
  # REDESIGNED (2026-07-12) around the Phase-1b node-scaling finding: on this
  # 1GbE fabric a SINGLE node at B=16 sustains ~121 samp/s -- 2x the best
  # multi-node cell (N=8/B=4/accum=8 = 60.8) and 10x the accum=1 N=8 cell that
  # was crawling at 11.4 samp/s overnight. So Phase 2 is now N=1 per cell:
  #   * each cell is its own 1-node job, so the 11 cells run in PARALLEL across
  #     the fish tier instead of one N=8 job hogging 8 nodes serially;
  #   * zero inter-node all-reduce -> zero NCCL-timeout crashes (the failure
  #     that burned 16 retries and lost a night), and a dead node kills one
  #     cell, not the whole sweep;
  #   * B=16 measured at 3.8-4.4GB / 24GB, so accum still carries eff_batch.
  local N="${WIN_N:-1}" B="${WIN_B:-16}" AMP="${WIN_AMP:-bf16}"
  local WORKERS="${WIN_WORKERS:-8}"
  # eff_batch = N*B*accum = 16*accum at N=1. Baseline D0 eff_batch=256 -> accum=16.
  local ACCUM="${WIN_ACCUM:-16}"
  # Generous epoch horizon so runs are bounded by the 3-day walltime, not the
  # budget: --signal=USR1 checkpoints at walltime and resume=true continues on
  # resubmit. 800 also sets the cosine-decay T_max (shared across all cells for
  # a fair comparison). Override with PHASE2_EPOCHS.
  local EPOCHS="${PHASE2_EPOCHS:-800}"
  # grad_compress/bucket are DDP-only; irrelevant at N=1 but harmless if the
  # config still reads them, so we simply drop them from the override string.
  local sys="train.epochs=${EPOCHS} train.batch_size=${B} train.amp=${AMP} train.num_workers=${WORKERS}"
  # Baseline D0: eff_batch=256 (accum=16 at N=1,B=16), lr_mult=1, warmup=5, mask=0.9, wd=0.05.

  # E1: effective batch via accum_steps {32,64,128,256} = A {2,4,8,16} at N*B=16
  for a in 2 4 8 16; do
    local eb=$(( N * B * a ))
    ROWS+=("${N}|${sys} train.accum_steps=${a} train.lr_mult=1.0 train.warmup_epochs=5 model.mask_ratio=0.9 train.weight_decay=0.05|p2-eb${eb}")
  done
  # E2: LR multiplier around the linear rule (at D0's eff_batch=256)
  for m in 0.5 2.0; do
    ROWS+=("${N}|${sys} train.accum_steps=${ACCUM} train.lr_mult=${m} train.warmup_epochs=5 model.mask_ratio=0.9 train.weight_decay=0.05|p2-lrmult${m}")
  done
  # E3: warmup
  ROWS+=("${N}|${sys} train.accum_steps=${ACCUM} train.lr_mult=1.0 train.warmup_epochs=10 model.mask_ratio=0.9 train.weight_decay=0.05|p2-warmup10")
  # E4: mask ratio (0.9 is D0)
  ROWS+=("${N}|${sys} train.accum_steps=${ACCUM} train.lr_mult=1.0 train.warmup_epochs=5 model.mask_ratio=0.75 train.weight_decay=0.05|p2-mask0.75")
  # E5: weight decay (0.05 is D0)
  ROWS+=("${N}|${sys} train.accum_steps=${ACCUM} train.lr_mult=1.0 train.warmup_epochs=5 model.mask_ratio=0.9 train.weight_decay=0.0|p2-wd0")
  # E6: seed variance of the chosen config (42 is default)
  for s in 1 7; do
    ROWS+=("${N}|${sys} train.accum_steps=${ACCUM} train.lr_mult=1.0 train.warmup_epochs=5 model.mask_ratio=0.9 train.weight_decay=0.05 train.seed=${s}|p2-seed${s}")
  done
}

case "${PHASE}" in
  1) build_phase1 ;;
  2) build_phase2 ;;
  *) echo "unknown phase '${PHASE}' (expected 1 or 2)" >&2; exit 2 ;;
esac

echo "# Phase ${PHASE}: ${#ROWS[@]} cells$([[ ${DRY_RUN} == 1 ]] && echo ' (dry-run)')"
for row in "${ROWS[@]}"; do
  IFS='|' read -r n overrides tag <<<"${row}"
  submit_row "${n}" "${overrides}" "${tag}"
done
