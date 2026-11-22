"""Behavioural tests for snapshot_mesh_checkpoints.sh.

The script polls watcher_state.json in an unbounded loop and shells out to
cp/scp, so it is exercised here against a mirrored repo layout under tmp_path.
One "poll" is one whole invocation: a run whose status is terminal (DONE) makes
the loop take exactly one pass and then exit rather than sleep, so a sequence
of polls is a sequence of invocations against successive watcher states.

Remote fetches are covered two different ways, matching the two mechanisms the
script actually offers. The watcher-host fetch (momentum + broadcast residual)
goes through the design's `SNAPSHOT_LOCAL_MOMENTUM_DIR` escape hatch; the
per-node push-residual fetch has no such hatch and is exercised against a
stubbed `scp` on PATH, which is also what lets one node be made unreachable.

`test_round_labelling_guard_never_backfills` pins the 2026-08-17 mislabelling
guard and predates the extended-capture work -- it is a characterization test,
so it passed against the unmodified script and must keep passing after it.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.system

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_SCRIPT = REPO_ROOT / "cluster" / "snapshot_mesh_checkpoints.sh"

# scp stub: resolves `host:/abs/path` against a local mirror tree, and fails for
# any host named in SCP_FAIL_HOSTS so an unreachable node can be simulated.
_SCP_STUB = """#!/usr/bin/env bash
args=()
for a in "$@"; do
  case "$a" in -*) ;; *) args+=("$a") ;; esac
done
src="${args[0]}"; dst="${args[1]}"
host="${src%%:*}"; remote="${src#*:}"
for bad in ${SCP_FAIL_HOSTS:-}; do
  if [[ "${host}" == "${bad}" ]]; then
    echo "ssh: connect to host ${host}: No route to host" >&2
    exit 255
  fi
done
mirror="${SCP_MIRROR}/${host}${remote}"
[[ -f "${mirror}" ]] || { echo "scp: ${remote}: No such file" >&2; exit 1; }
cp "${mirror}" "${dst}"
"""


def repo_root(tmp_path: Path) -> Path:
    """The repo checkout. Deliberately NOT a parent of the capture dest, since
    the NFS guard refuses a dest under it."""
    return tmp_path / "repo"


def mesh_shared(tmp_path: Path) -> Path:
    """The mesh's own shared checkpoint tree -- where master.pt/pending/ live."""
    return tmp_path / "stern_mesh"


def capture_dir(tmp_path: Path, run_id: str) -> Path:
    """Where the snapshotter writes its per-round captures."""
    return tmp_path / "dest" / run_id


def watcher_local_dir(tmp_path: Path) -> Path:
    """Stands in for MESH_LOCAL_ROOT/<run_id> on the watcher host."""
    return tmp_path / "watcher_local"


def node_checkpoint_root() -> str:
    """The per-node STERN_CHECKPOINT_ROOT the push residual is fetched from."""
    return "/Data/stern_checkpoints"


def write_watcher_state(tmp_path: Path, run_id: str, status: str, rnd: int) -> None:
    state = repo_root(tmp_path) / "runs" / "state" / "watcher_state.json"
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps({"runs": {run_id: {"status": status, "round": rnd}}}))


def write_pending(
    tmp_path: Path,
    run_id: str,
    rnd: int,
    node: str,
    *,
    master_version: int | None = None,
    full_model: bool = False,
) -> Path:
    """Write one pending file in the real `{run_id}_{round}_{node}.pt` form."""
    pending = mesh_shared(tmp_path) / "checkpoints" / "pending"
    pending.mkdir(parents=True, exist_ok=True)
    key = "model_state_dict" if full_model else "delta_state_dict"
    payload: dict = {key: {"blocks.0.mlp.fc1.weight": torch.zeros(2, 2)}}
    if master_version is not None:
        payload["master_version"] = master_version
    path = pending / f"{run_id}_{rnd}_{node}.pt"
    torch.save(payload, path)
    return path


def write_node_residual(tmp_path: Path, node: str, run_id: str, payload: str) -> None:
    """Seed the scp mirror with a node's `<run_id>_push_residual.pt`."""
    mirror = tmp_path / "scp_mirror" / node / node_checkpoint_root().lstrip("/")
    mirror.mkdir(parents=True, exist_ok=True)
    (mirror / f"{run_id}_push_residual.pt").write_text(payload)


def write_watcher_artefacts(
    tmp_path: Path, *, momentum: str | None = None, broadcast: str | None = None
) -> None:
    """Seed the watcher host's momentum / broadcast-residual buffers."""
    local = watcher_local_dir(tmp_path)
    local.mkdir(parents=True, exist_ok=True)
    if momentum is not None:
        (local / "outer_momentum.pt").write_text(momentum)
    if broadcast is not None:
        (local / "broadcast_residual.pt").write_text(broadcast)


def manifest_records(tmp_path: Path, run_id: str) -> list[dict]:
    path = capture_dir(tmp_path, run_id) / "manifest.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def invoke(
    tmp_path: Path,
    run_id: str,
    every_n: int,
    max_round: int,
    **extra_env: str,
) -> subprocess.CompletedProcess:
    import sys

    script = (
        repo_root(tmp_path)
        / "cluster"
        / "snapshot_mesh_checkpoints.sh"
    )
    if not script.exists():
        script.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REAL_SCRIPT, script)
        script.chmod(0o755)
    stub_bin = tmp_path / "stub_bin"
    if not stub_bin.exists():
        stub_bin.mkdir()
        (stub_bin / "scp").write_text(_SCP_STUB)
        (stub_bin / "scp").chmod(0o755)
    env = {
        "PATH": f"{stub_bin}:/usr/bin:/bin",
        "HOME": str(tmp_path / "home"),
        "SNAPSHOT_REPO_ROOT": str(repo_root(tmp_path)),
        "SNAPSHOT_MESH_SHARED": str(mesh_shared(tmp_path)),
        "SNAPSHOT_DEST": str(tmp_path / "dest"),
        # torch is needed to read a pending payload's provenance fields; the
        # system python3 that runs run_status does not have it.
        "SNAPSHOT_PYTHON": sys.executable,
        "SNAPSHOT_WATCHER_HOST": "node089",
        "SNAPSHOT_LOCAL_MOMENTUM_DIR": str(watcher_local_dir(tmp_path)),
        "SNAPSHOT_NODE_CHECKPOINT_ROOT": node_checkpoint_root(),
        "SCP_MIRROR": str(tmp_path / "scp_mirror"),
        **extra_env,
    }
    return subprocess.run(
        [str(script), run_id, str(every_n), str(max_round), "1"],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


def run_snapshot_once(
    tmp_path: Path,
    run_id: str,
    rnd: int,
    every_n: int,
    max_round: int,
    master_bytes: bytes | None = None,
    **extra_env: str,
) -> subprocess.CompletedProcess:
    """Drive exactly one poll of the snapshotter at round `rnd`.

    `status=DONE` is what bounds the loop: the script takes its capture pass and
    then exits instead of sleeping, so the test never has to race a poll timer.
    """
    if master_bytes is not None:
        master = mesh_shared(tmp_path) / "checkpoints" / run_id / "master.pt"
        master.parent.mkdir(parents=True, exist_ok=True)
        master.write_bytes(master_bytes)
    write_watcher_state(tmp_path, run_id, "DONE", rnd)
    return invoke(tmp_path, run_id, every_n, max_round, **extra_env)


def test_round_labelling_guard_never_backfills(tmp_path):
    """A boundary the loop never polled at is skipped, not reconstructed.

    master.pt is overwritten on every merge, so when the round counter jumps
    from 0 to 10 between polls the file holds round-10 weights only. Copying it
    to master_round5.pt would mislabel a checkpoint -- the guard forbids it, and
    round 5 is simply never captured.
    """
    run_id = "lev-guard"
    run_snapshot_once(
        tmp_path, run_id, rnd=0, every_n=5, max_round=10, master_bytes=b"round0-weights"
    )
    run_snapshot_once(
        tmp_path,
        run_id,
        rnd=10,
        every_n=5,
        max_round=10,
        master_bytes=b"round10-weights",
    )

    dest = capture_dir(tmp_path, run_id)
    assert not (dest / "master_round5.pt").exists()
    assert (dest / "master_round10.pt").read_bytes() == b"round10-weights"


def test_captures_all_artefacts_per_boundary(tmp_path):
    """Every artefact a merge can be reconstructed from lands for one round."""
    run_id = "lev-all"
    write_pending(tmp_path, run_id, 6, "nodeA", master_version=5)
    write_pending(tmp_path, run_id, 6, "nodeB", master_version=5)
    write_watcher_artefacts(tmp_path, momentum="MOM", broadcast="BCAST")
    write_node_residual(tmp_path, "nodeA", run_id, "PUSH-A")
    write_node_residual(tmp_path, "nodeB", run_id, "PUSH-B")

    run_snapshot_once(
        tmp_path,
        run_id,
        rnd=6,
        every_n=2,
        max_round=6,
        master_bytes=b"master6",
        SNAPSHOT_EXPECTED_NODES="nodeA nodeB",
    )

    dest = capture_dir(tmp_path, run_id)
    assert (dest / "master_round6.pt").read_bytes() == b"master6"
    assert sorted(p.name for p in (dest / "pending_round6").glob("*.pt")) == [
        f"{run_id}_6_nodeA.pt",
        f"{run_id}_6_nodeB.pt",
    ]
    assert (dest / "outer_momentum_round6.pt").read_text() == "MOM"
    assert (dest / "broadcast_residual_round6.pt").read_text() == "BCAST"
    assert (dest / "push_residual_round6" / "nodeA.pt").read_text() == "PUSH-A"
    assert (dest / "push_residual_round6" / "nodeB.pt").read_text() == "PUSH-B"

    [record] = manifest_records(tmp_path, run_id)
    assert record["round"] == 6
    assert record["gaps"] == []


def test_both_residuals_captured_per_round(tmp_path):
    """The two error-feedback buffers live in different places and are distinct.

    broadcast_residual.pt is the watcher's; <run_id>_push_residual.pt is each
    worker's. Capturing only one leaves the merge reconstructable to within a
    round-correlated quantization offset, which is what Plan 6/7 would misread.
    """
    run_id = "lev-resid"
    write_pending(tmp_path, run_id, 4, "nodeA", master_version=3)
    write_watcher_artefacts(tmp_path, momentum="MOM", broadcast="BCAST-ROUND4")
    write_node_residual(tmp_path, "nodeA", run_id, "PUSH-A-ROUND4")

    run_snapshot_once(
        tmp_path,
        run_id,
        rnd=4,
        every_n=2,
        max_round=4,
        master_bytes=b"master4",
        SNAPSHOT_EXPECTED_NODES="nodeA",
    )

    dest = capture_dir(tmp_path, run_id)
    assert (dest / "broadcast_residual_round4.pt").read_text() == "BCAST-ROUND4"
    assert (dest / "push_residual_round4" / "nodeA.pt").read_text() == "PUSH-A-ROUND4"
    assert manifest_records(tmp_path, run_id)[0]["gaps"] == []


def test_partial_pending_set_recorded_as_gap_not_completed(tmp_path):
    """Two expected nodes, one pending file: the absent node is named as a gap.

    Without the roster, "incomplete" is undecidable from pending/ alone, and
    analysing 1-of-2 nodes' deltas as if complete understates divergence.
    """
    run_id = "lev-partial"
    write_pending(tmp_path, run_id, 6, "nodeA", master_version=5)
    write_watcher_artefacts(tmp_path, momentum="MOM", broadcast="BCAST")
    write_node_residual(tmp_path, "nodeA", run_id, "PUSH-A")
    write_node_residual(tmp_path, "nodeB", run_id, "PUSH-B")

    run_snapshot_once(
        tmp_path,
        run_id,
        rnd=6,
        every_n=2,
        max_round=6,
        master_bytes=b"master6",
        SNAPSHOT_EXPECTED_NODES="nodeA nodeB",
    )

    dest = capture_dir(tmp_path, run_id)
    assert [p.name for p in (dest / "pending_round6").glob("*.pt")] == [
        f"{run_id}_6_nodeA.pt"
    ]
    [record] = manifest_records(tmp_path, run_id)
    gaps = " ".join(record["gaps"])
    assert "nodeB" in gaps and "pending" in gaps
    assert "nodeA" not in gaps
    assert [entry["node"] for entry in record["pending"]] == ["nodeA"]


def test_manifest_records_push_mode_and_sha256(tmp_path):
    """Each pending entry records who pushed it, against what, in which form."""
    run_id = "lev-prov"
    delta = write_pending(tmp_path, run_id, 5, "nodeA", master_version=4)
    write_pending(tmp_path, run_id, 5, "nodeB", full_model=True)
    write_watcher_artefacts(tmp_path, momentum="MOM", broadcast="BCAST")
    write_node_residual(tmp_path, "nodeA", run_id, "PUSH-A")
    write_node_residual(tmp_path, "nodeB", run_id, "PUSH-B")

    run_snapshot_once(
        tmp_path,
        run_id,
        rnd=5,
        every_n=5,
        max_round=5,
        master_bytes=b"master5",
        SNAPSHOT_EXPECTED_NODES="nodeA nodeB",
    )

    [record] = manifest_records(tmp_path, run_id)
    entries = {entry["node"]: entry for entry in record["pending"]}
    assert entries["nodeA"]["master_version"] == 4
    assert entries["nodeA"]["payload_kind"] == "delta_state_dict"
    assert entries["nodeA"]["sha256"] == hashlib.sha256(delta.read_bytes()).hexdigest()
    assert len(entries["nodeA"]["sha256"]) == 64
    # A node that pushed before any master existed sends the full model and
    # carries no master_version -- recorded as null, never defaulted to 0.
    assert entries["nodeB"]["payload_kind"] == "model_state_dict"
    assert entries["nodeB"]["master_version"] is None


def test_unreachable_node_push_residual_recorded_as_gap(tmp_path):
    """One unreachable node costs that node's residual, not the whole round."""
    run_id = "lev-unreach"
    write_pending(tmp_path, run_id, 4, "nodeA", master_version=3)
    write_pending(tmp_path, run_id, 4, "nodeB", master_version=3)
    write_watcher_artefacts(tmp_path, momentum="MOM", broadcast="BCAST")
    write_node_residual(tmp_path, "nodeA", run_id, "PUSH-A")
    write_node_residual(tmp_path, "nodeB", run_id, "PUSH-B")

    run_snapshot_once(
        tmp_path,
        run_id,
        rnd=4,
        every_n=2,
        max_round=4,
        master_bytes=b"master4",
        SNAPSHOT_EXPECTED_NODES="nodeA nodeB",
        SCP_FAIL_HOSTS="nodeB",
    )

    dest = capture_dir(tmp_path, run_id)
    assert (dest / "master_round4.pt").exists()
    assert (dest / "push_residual_round4" / "nodeA.pt").read_text() == "PUSH-A"
    assert not (dest / "push_residual_round4" / "nodeB.pt").exists()

    [record] = manifest_records(tmp_path, run_id)
    gaps = " ".join(record["gaps"])
    assert "nodeB" in gaps and "residual" in gaps
    assert record["push_residuals"] == {"nodeA": "push_residual_round4/nodeA.pt"}


def test_exits_nonzero_if_dest_is_nfs_home(tmp_path):
    """A dest under the NFS home is refused up front, not filled until it dies.

    master.pt alone is 183MB and the quota holds under 55 of them, so failing
    partway through a run is strictly worse than refusing to start.
    """
    run_id = "lev-nfs"
    home = tmp_path / "home"
    bad_dest = home / "stern_mesh" / "snapshots"

    result = run_snapshot_once(
        tmp_path,
        run_id,
        rnd=5,
        every_n=5,
        max_round=5,
        master_bytes=b"master5",
        SNAPSHOT_DEST=str(bad_dest),
        SNAPSHOT_EXPECTED_NODES="nodeA",
    )

    assert result.returncode != 0
    combined = (result.stdout + result.stderr).lower()
    assert "nfs" in combined or "quota" in combined
    assert not (bad_dest / run_id / "master_round5.pt").exists()


def test_retention_every_round_first_ten_then_every_n(tmp_path):
    """Rounds 1-10 are all eligible; past 10, only every Nth and MAX_ROUND."""
    run_id = "lev-retain"
    write_watcher_artefacts(tmp_path, momentum="MOM", broadcast="BCAST")

    for rnd in range(1, 13):
        run_snapshot_once(
            tmp_path,
            run_id,
            rnd=rnd,
            every_n=5,
            max_round=12,
            master_bytes=f"master{rnd}".encode(),
            SNAPSHOT_EXPECTED_NODES="",
        )

    dest = capture_dir(tmp_path, run_id)
    captured = sorted(
        int(p.stem.removeprefix("master_round")) for p in dest.glob("master_round*.pt")
    )
    assert captured == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
    assert (dest / "master_round12.pt").read_bytes() == b"master12"


def test_unobserved_eligible_round_is_not_backfilled(tmp_path):
    """Eligibility is necessary, not sufficient: an unpolled round stays absent.

    Rounds 4 and 5 sit inside the first-ten eligibility window, but the counter
    jumped over them -- so they get no file and no manifest record, rather than
    a later master.pt relabelled.
    """
    run_id = "lev-jump"
    write_watcher_artefacts(tmp_path, momentum="MOM", broadcast="BCAST")

    run_snapshot_once(
        tmp_path,
        run_id,
        rnd=3,
        every_n=5,
        max_round=6,
        master_bytes=b"master3",
        SNAPSHOT_EXPECTED_NODES="",
    )
    run_snapshot_once(
        tmp_path,
        run_id,
        rnd=6,
        every_n=5,
        max_round=6,
        master_bytes=b"master6",
        SNAPSHOT_EXPECTED_NODES="",
    )

    dest = capture_dir(tmp_path, run_id)
    assert not (dest / "master_round4.pt").exists()
    assert not (dest / "master_round5.pt").exists()
    assert (dest / "master_round6.pt").read_bytes() == b"master6"
    assert sorted(r["round"] for r in manifest_records(tmp_path, run_id)) == [3, 6]
