#!/usr/bin/env bash
# cluster/snapshot_mesh_checkpoints.sh
# Preserve everything a round's merge can be reconstructed from, so the SSL
# diagnostic tooling can compare mid-training state afterwards. The mesh
# overwrites master.pt on every merge and sweeps pending/ after each one, so
# without this nothing survives a round boundary except the final weights.
#
# Per captured round boundary N, under ${SNAPSHOT_DEST}/<run_id>/:
#   master_round<N>.pt            the merged master as it stood at round N
#   pending_round<N>/*.pt         the per-node deltas that fed that round
#   outer_momentum_round<N>.pt    the DiLoCo outer-optimizer momentum buffer
#   broadcast_residual_round<N>.pt  watcher-side quantization error feedback
#   push_residual_round<N>/<node>.pt  each worker's quantization error feedback
#   manifest.jsonl                one record per round, gaps listed explicitly
#
# Both residuals are captured because deltas AND the master broadcast are
# quantized with carry-over (mesh_round.py:935, run_outer_step.py:526). Master
# + pending + momentum alone reconstructs a merge only up to an error that is
# correlated across rounds -- exactly the systematic offset a DiLoCo divergence
# check would misread as signal.
#
# Usage: snapshot_mesh_checkpoints.sh <run_id> <every_n> <max_round> [poll_interval_s]
#
# Run under nohup on a worker-adjacent host alongside the watcher; exits 0 on
# its own once the max_round snapshot is taken. Deliberately external to
# watcher/mesh_tick.py -- copying files needs no coupling to the coordinator
# state machine.
#
# NOT a drop-in for pre-2026-09 invocations: output moved off ${HOME} (see the
# NFS guard below), so a running snapshotter must be relaunched with
# SNAPSHOT_DEST on local NVMe. Snapshots taken before that stay where they are.
#
# Round semantics: mesh_tick's _do_syncing increments run.round *after* round
# R's merge and sync complete, so watcher_state's `round == N` means rounds
# 0..N-1 are done and master.pt holds round N-1's merge -- i.e. exactly N
# completed rounds. master.pt is written via checkpoint.save's tmp+rename, so
# a plain cp always reads a complete file; no locking needed.
set -euo pipefail

USAGE="usage: snapshot_mesh_checkpoints.sh <run_id> <every_n> <max_round> [poll_interval_s]"
RUN_ID="${1:?${USAGE}}"
EVERY_N="${2:?${USAGE}}"
MAX_ROUND="${3:?${USAGE}}"
POLL_S="${4:-300}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SNAPSHOT_REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
MESH_SHARED="${SNAPSHOT_MESH_SHARED:-${HOME}/stern_mesh}"
DEST_ROOT="${SNAPSHOT_DEST:?SNAPSHOT_DEST must name a local-NVMe directory}"

# torch is needed only to read a pending payload's provenance fields; run_status
# stays on the system python3, which has no such dependency.
SNAPSHOT_PYTHON="${SNAPSHOT_PYTHON:-python3}"
WATCHER_HOST="${SNAPSHOT_WATCHER_HOST:-}"
WATCHER_MESH_LOCAL_ROOT="${SNAPSHOT_WATCHER_MESH_LOCAL_ROOT:-/Data/${USER:-}/stern_mesh_local}"
# Test-only escape hatch: read the watcher-host buffers from a local directory
# instead of scp'ing them, so CI needs no live ssh.
LOCAL_MOMENTUM_DIR="${SNAPSHOT_LOCAL_MOMENTUM_DIR:-}"
NODE_CHECKPOINT_ROOT="${SNAPSHOT_NODE_CHECKPOINT_ROOT:-/Data/${USER:-}/stern_checkpoints}"
read -r -a EXPECTED_NODES <<<"${SNAPSHOT_EXPECTED_NODES:-}"

WATCHER_STATE="${REPO_ROOT}/runs/state/watcher_state.json"
CKPT_DIR="${MESH_SHARED}/checkpoints/${RUN_ID}"
PENDING_DIR="${MESH_SHARED}/checkpoints/pending"
MASTER="${CKPT_DIR}/master.pt"
OUT_DIR="${DEST_ROOT}/${RUN_ID}"
MANIFEST="${OUT_DIR}/manifest.jsonl"
CAPTURED_MARKER="${OUT_DIR}/.rounds_captured"

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) snapshot[${RUN_ID}] $*"; }

# master.pt alone is 183MB and the NFS-quota-constrained home holds under 55 of
# them, so a dest there fills the quota mid-run and fails partway. Refuse up
# front instead: a run that never started is recoverable, a half-captured round
# axis is not.
abs_dest="$(readlink -m "${DEST_ROOT}")"
for forbidden in "${HOME}" "${REPO_ROOT}"; do
  [[ -n "${forbidden}" ]] || continue
  abs_forbidden="$(readlink -m "${forbidden}")"
  case "${abs_dest}/" in
    "${abs_forbidden}"/*)
      echo "ERROR SNAPSHOT_DEST=${DEST_ROOT} resolves under ${abs_forbidden}," \
           "which is the NFS-quota-constrained home -- point it at local NVMe" >&2
      exit 1
      ;;
  esac
done

# "<status> <round>", or "MISSING 0" -- same idiom as sweep_queue.sh's run_status.
run_status() {
  python3 - "${WATCHER_STATE}" "${RUN_ID}" <<'PYEOF'
import json
import sys
from pathlib import Path

path, run_id = Path(sys.argv[1]), sys.argv[2]
try:
    runs = json.loads(path.read_text()).get("runs", {})
except (OSError, ValueError):
    runs = {}
run = runs.get(run_id)
print(run["status"], run.get("round", 0)) if run else print("MISSING", 0)
PYEOF
}

sha256_of() { sha256sum "$1" | cut -d' ' -f1; }

# "<master_version-or-empty><TAB><payload_kind>" for one pending file.
pending_meta() {
  "${SNAPSHOT_PYTHON}" - "$1" <<'PYEOF'
import sys

import torch

payload = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
for kind in ("delta_state_dict", "model_state_dict"):
    if kind in payload:
        break
else:
    kind = "unknown"
version = payload.get("master_version")
print(f"{'' if version is None else version}\t{kind}")
PYEOF
}

# Fetch one of the watcher host's per-run buffers. Honours the local-dir escape
# hatch; otherwise scp from MESH_LOCAL_ROOT/<run_id>/ on the watcher host.
fetch_watcher_file() {
  local basename="$1" dest="$2"
  if [[ -n "${LOCAL_MOMENTUM_DIR}" ]]; then
    cp "${LOCAL_MOMENTUM_DIR}/${basename}" "${dest}" 2>/dev/null
  else
    [[ -n "${WATCHER_HOST}" ]] || return 1
    scp -q "${WATCHER_HOST}:${WATCHER_MESH_LOCAL_ROOT}/${RUN_ID}/${basename}" \
        "${dest}" 2>/dev/null
  fi
}

# A node is training while this runs, so the fetch is read-only, off the
# training path, and a node that cannot be reached becomes a per-node gap
# rather than failing the whole round's capture.
fetch_node_residual() {
  local node="$1" dest="$2"
  scp -q "${node}:${NODE_CHECKPOINT_ROOT}/${RUN_ID}_push_residual.pt" \
      "${dest}" 2>/dev/null
}

# Every round is eligible for the first ten -- early rounds are where the
# interesting dynamics are and they are cheap -- and only every Nth (plus
# MAX_ROUND) after that. Eligibility is necessary, never sufficient: the
# round-labelling guard still requires that the round was actually observed.
eligible_round() {
  local r="$1"
  (( r > 0 )) || return 1
  (( r <= 10 || r % EVERY_N == 0 || r == MAX_ROUND ))
}

already_captured() {
  [[ -f "${CAPTURED_MARKER}" ]] && grep -qx "$1" "${CAPTURED_MARKER}"
}

write_manifest_record() {
  local round="$1" work="$2"
  MANIFEST_RUN_ID="${RUN_ID}" MANIFEST_ROUND="${round}" \
  MANIFEST_EXPECTED="${SNAPSHOT_EXPECTED_NODES:-}" \
  python3 - "${work}" "${MANIFEST}" <<'PYEOF'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

work, manifest = Path(sys.argv[1]), Path(sys.argv[2])


def lines(name):
    path = work / name
    return [ln for ln in path.read_text().splitlines() if ln.strip()] if path.exists() else []


def scalar(name):
    path = work / name
    return path.read_text().strip() or None if path.exists() else None


pending = []
for line in lines("pending.tsv"):
    node, version, kind, digest = line.split("\t")
    pending.append(
        {
            "node": node,
            # Absent for a node that pushed before any master existed: recorded
            # as null rather than defaulted, since 0 is a real master_version.
            "master_version": int(version) if version else None,
            "payload_kind": kind,
            "sha256": digest,
        }
    )

record = {
    "run_id": os.environ["MANIFEST_RUN_ID"],
    "round": int(os.environ["MANIFEST_ROUND"]),
    "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "expected_nodes": os.environ["MANIFEST_EXPECTED"].split(),
    "master": scalar("master.txt"),
    "master_sha256": scalar("master_sha256.txt"),
    "pending": sorted(pending, key=lambda entry: entry["node"]),
    "momentum": scalar("momentum.txt"),
    "broadcast_residual": scalar("broadcast.txt"),
    "push_residuals": dict(line.split("\t") for line in lines("push.tsv")),
    "gaps": lines("gaps.txt"),
}
with manifest.open("a") as handle:
    handle.write(json.dumps(record) + "\n")
PYEOF
}

capture_round() {
  local round="$1"
  local work
  work="$(mktemp -d)"
  trap 'rm -rf "${work}"' RETURN
  mkdir -p "${OUT_DIR}"

  if [[ -f "${MASTER}" ]]; then
    local master_dest="${OUT_DIR}/master_round${round}.pt"
    cp "${MASTER}" "${master_dest}"
    echo "master_round${round}.pt" >"${work}/master.txt"
    sha256_of "${master_dest}" >"${work}/master_sha256.txt"
  else
    echo "master.pt missing at round ${round}" >>"${work}/gaps.txt"
  fi

  local pending_out="${OUT_DIR}/pending_round${round}"
  local seen_nodes=()
  local src
  for src in "${PENDING_DIR}/${RUN_ID}_${round}"_*.pt; do
    [[ -f "${src}" ]] || continue
    local base node meta
    base="$(basename "${src}")"
    node="${base#"${RUN_ID}_${round}_"}"
    node="${node%.pt}"
    mkdir -p "${pending_out}"
    cp "${src}" "${pending_out}/${base}"
    if meta="$(pending_meta "${pending_out}/${base}" 2>/dev/null)"; then
      printf '%s\t%s\t%s\n' "${node}" "${meta}" \
          "$(sha256_of "${pending_out}/${base}")" >>"${work}/pending.tsv"
    else
      echo "unreadable pending payload for node ${node}" >>"${work}/gaps.txt"
    fi
    seen_nodes+=("${node}")
  done

  # Without the roster, "incomplete" is undecidable from pending/ alone, so a
  # partial set would read as a smaller-but-complete round -- understating
  # divergence and overstating quorum in the DiLoCo checks downstream.
  local expected
  for expected in "${EXPECTED_NODES[@]+"${EXPECTED_NODES[@]}"}"; do
    local found=0 seen
    for seen in "${seen_nodes[@]+"${seen_nodes[@]}"}"; do
      [[ "${seen}" == "${expected}" ]] && found=1 && break
    done
    (( found )) || echo "pending missing for node ${expected}" >>"${work}/gaps.txt"
  done

  if fetch_watcher_file "outer_momentum.pt" "${OUT_DIR}/outer_momentum_round${round}.pt"; then
    echo "outer_momentum_round${round}.pt" >"${work}/momentum.txt"
  else
    echo "outer_momentum fetch failed at round ${round}" >>"${work}/gaps.txt"
  fi

  if fetch_watcher_file "broadcast_residual.pt" \
      "${OUT_DIR}/broadcast_residual_round${round}.pt"; then
    echo "broadcast_residual_round${round}.pt" >"${work}/broadcast.txt"
  else
    echo "broadcast_residual fetch failed at round ${round}" >>"${work}/gaps.txt"
  fi

  local push_out="${OUT_DIR}/push_residual_round${round}"
  local node
  for node in "${EXPECTED_NODES[@]+"${EXPECTED_NODES[@]}"}"; do
    mkdir -p "${push_out}"
    if fetch_node_residual "${node}" "${push_out}/${node}.pt"; then
      printf '%s\t%s\n' "${node}" "push_residual_round${round}/${node}.pt" \
          >>"${work}/push.tsv"
    else
      rm -f "${push_out}/${node}.pt"
      echo "push_residual fetch failed for node ${node}" >>"${work}/gaps.txt"
    fi
  done

  write_manifest_record "${round}" "${work}"
  echo "${round}" >>"${CAPTURED_MARKER}"
  log "captured round ${round} -> ${OUT_DIR} ($(wc -l <"${work}/gaps.txt" 2>/dev/null || echo 0) gaps)"
}

final_snapshot="${OUT_DIR}/master_round${MAX_ROUND}.pt"
log "watching for rounds 1-10 then every ${EVERY_N} up to ${MAX_ROUND} (poll ${POLL_S}s)"

while true; do
  read -r status round <<<"$(run_status)"

  # Only capture the round master.pt CURRENTLY holds -- never backfill earlier
  # boundaries the loop didn't poll at. master.pt is overwritten on every merge,
  # so if the round counter advanced past several boundaries between two polls
  # (poll interval coarser than the round cadence, or the script started
  # mid-run), today's master.pt is NOT those earlier rounds' weights; copying it
  # under an earlier round's filename would silently mislabel a checkpoint (bug
  # caught in review, 2026-08-17). The same reasoning holds for both residuals,
  # which additionally carry quantization error forward across rounds. A
  # boundary missed this way is simply never captured -- logged as a gap, not
  # faked.
  if eligible_round "${round}" && ! already_captured "${round}"; then
    capture_round "${round}"
  fi

  if [[ -f "${final_snapshot}" ]]; then
    log "done -- round ${MAX_ROUND} snapshot captured"
    exit 0
  fi

  if [[ "${status}" == "DONE" || "${status}" == "MISSING" ]] && (( round < MAX_ROUND )); then
    log "ERROR run ended at round ${round} (status ${status}) before ${MAX_ROUND} -- giving up"
    exit 1
  fi

  sleep "${POLL_S}"
done
