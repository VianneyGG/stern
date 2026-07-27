from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import torch

from video_encoding.scripts.run_outer_step import _load_worker_checkpoints, merge_round

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_cli_invocation_as_module_actually_runs(tmp_path):
    """Real subprocess, no mocking: mesh_tick.py's _do_merging shells out to
    this script via `uv run python -m video_encoding.scripts.run_outer_step`
    (not a bare script path -- that puts the script's own directory on
    sys.path[0] instead of the repo root, so the `from video_encoding...`
    import below can never resolve). A mocked subprocess.run in
    test_mesh_tick.py can't catch that class of bug; only actually running
    the command can.
    """
    pending_dir = tmp_path / "pending"
    pending_dir.mkdir()
    master_path = tmp_path / "master.pt"
    momentum_path = tmp_path / "momentum.pt"

    _write_master(master_path, 1.0)
    _write_worker(pending_dir / "run1_a.pt", 0.5, n_samples=100)

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "video_encoding.scripts.run_outer_step",
            "--pending-dir",
            str(pending_dir),
            "--master",
            str(master_path),
            "--momentum",
            str(momentum_path),
            "--outer-lr",
            "0.7",
            "--outer-momentum",
            "0.9",
            "--weighting",
            "data",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr


def _sd(v: float) -> dict[str, torch.Tensor]:
    return {"w": torch.full((2, 2), v), "b": torch.full((2,), v)}


def _write_master(path: Path, v: float) -> None:
    torch.save({"model_state_dict": _sd(v)}, path)


def _write_worker(
    path: Path,
    v: float,
    *,
    n_samples: int | None = 100,
    val_loss: float = 1.0,
    round_: int = 1,
    omit_n_samples: bool = False,
) -> None:
    payload = {
        "model_state_dict": _sd(v),
        "val_loss": val_loss,
        "round": round_,
    }
    if not omit_n_samples:
        payload["n_samples"] = n_samples
    torch.save(payload, path)


def test_merge_with_lr1_momentum0_reproduces_weighted_average(tmp_path):
    pending_dir = tmp_path / "pending"
    pending_dir.mkdir()
    master_path = tmp_path / "master.pt"
    momentum_path = tmp_path / "momentum.pt"

    _write_master(master_path, 1.0)
    _write_worker(pending_dir / "run1_a.pt", 0.0, n_samples=100)
    _write_worker(pending_dir / "run1_b.pt", 2.0, n_samples=100)

    merge_round(
        pending_dir, master_path, momentum_path, lr=1.0, momentum=0.0, weighting="data"
    )

    merged = torch.load(master_path, weights_only=False)["model_state_dict"]
    assert merged["w"][0, 0].item() == pytest.approx(1.0)


def test_alphas_follow_sample_counts_not_val_loss(tmp_path):
    pending_dir = tmp_path / "pending"
    pending_dir.mkdir()
    master_path = tmp_path / "master.pt"
    momentum_path = tmp_path / "momentum.pt"

    _write_master(master_path, 1.0)
    _write_worker(pending_dir / "run1_a.pt", 0.5, n_samples=300, val_loss=9.0)
    _write_worker(pending_dir / "run1_b.pt", 0.5, n_samples=100, val_loss=0.1)

    result = merge_round(
        pending_dir, master_path, momentum_path, lr=0.7, momentum=0.9, weighting="data"
    )

    assert result["alphas"]["run1_a"] == pytest.approx(0.75)
    assert result["alphas"]["run1_b"] == pytest.approx(0.25)


def test_momentum_buffer_is_written_and_reloaded(tmp_path, capsys):
    pending_dir = tmp_path / "pending"
    pending_dir.mkdir()
    master_path = tmp_path / "master.pt"
    momentum_path = tmp_path / "momentum.pt"

    _write_master(master_path, 1.0)
    _write_worker(pending_dir / "run1_a.pt", 1.1, n_samples=100)

    merge_round(
        pending_dir, master_path, momentum_path, lr=0.7, momentum=0.9, weighting="data"
    )
    assert momentum_path.exists()
    # Round 1 has no prior momentum buffer -- expect a loud stderr warning.
    assert "momentum" in capsys.readouterr().err.lower()

    m1 = torch.load(master_path, weights_only=False)["model_state_dict"]["w"][
        0, 0
    ].item()

    # Round 2: same local delta again, momentum should accelerate the step
    # (mirrors mesh/outer_step.py's own momentum-acceleration test).
    merge_round(
        pending_dir, master_path, momentum_path, lr=0.7, momentum=0.9, weighting="data"
    )
    m2 = torch.load(master_path, weights_only=False)["model_state_dict"]["w"][
        0, 0
    ].item()

    assert m2 != pytest.approx(m1)


def test_checkpoint_without_n_samples_is_rejected_loudly(tmp_path):
    pending_dir = tmp_path / "pending"
    pending_dir.mkdir()
    master_path = tmp_path / "master.pt"
    momentum_path = tmp_path / "momentum.pt"

    _write_master(master_path, 1.0)
    _write_worker(pending_dir / "run1_a.pt", 0.5, omit_n_samples=True)

    with pytest.raises(ValueError, match="n_samples"):
        merge_round(
            pending_dir,
            master_path,
            momentum_path,
            lr=0.7,
            momentum=0.9,
            weighting="data",
        )


def test_empty_pending_dir_raises(tmp_path):
    pending_dir = tmp_path / "pending"
    pending_dir.mkdir()
    master_path = tmp_path / "master.pt"
    momentum_path = tmp_path / "momentum.pt"
    _write_master(master_path, 1.0)

    with pytest.raises(ValueError, match="no pending"):
        merge_round(
            pending_dir,
            master_path,
            momentum_path,
            lr=0.7,
            momentum=0.9,
            weighting="data",
        )


def test_load_worker_checkpoints_upcasts_fp16_to_fp32(tmp_path):
    pending_dir = tmp_path / "pending"
    pending_dir.mkdir()
    torch.save(
        {
            "model_state_dict": {"w": torch.ones(2, 2).half()},
            "n_samples": 100,
        },
        pending_dir / "run1_a.pt",
    )

    checkpoints = _load_worker_checkpoints(pending_dir)

    assert checkpoints["run1_a"]["model_state_dict"]["w"].dtype == torch.float32


def test_inverse_loss_weighting_is_available_as_an_option(tmp_path):
    pending_dir = tmp_path / "pending"
    pending_dir.mkdir()
    master_path = tmp_path / "master.pt"
    momentum_path = tmp_path / "momentum.pt"

    _write_master(master_path, 1.0)
    _write_worker(pending_dir / "run1_a.pt", 0.5, n_samples=100, val_loss=1.0)
    _write_worker(pending_dir / "run1_b.pt", 0.5, n_samples=100, val_loss=3.0)

    result = merge_round(
        pending_dir,
        master_path,
        momentum_path,
        lr=0.7,
        momentum=0.9,
        weighting="inverse_loss",
    )

    assert result["alphas"]["run1_a"] > result["alphas"]["run1_b"]
