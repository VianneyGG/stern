from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import torch

from video_encoding.mesh.outer_step import BOOTSTRAP_MASTER_VERSION
from video_encoding.scripts.run_outer_step import (
    _load_worker_checkpoints,
    _quantize_broadcast_with_error_feedback,
    main,
    merge_round,
)

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
    _write_worker(pending_dir / "run1_0_a.pt", 0.5, n_samples=100)

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "video_encoding.scripts.run_outer_step",
            "--pending-dir",
            str(pending_dir),
            "--run-id",
            "run1",
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


def _write_master(path: Path, v: float, *, master_version: int | None = None) -> None:
    torch.save({"model_state_dict": _sd(v), "master_version": master_version}, path)


def _write_worker(
    path: Path,
    v: float,
    *,
    n_samples: int | None = 100,
    n_steps: int | None = None,
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
    if n_steps is not None:
        payload["n_steps"] = n_steps
    torch.save(payload, path)


def _write_worker_delta(
    path: Path, v: float, *, n_samples: int = 100, master_version: int | None = None
) -> None:
    """Phase 3.2 push shape: delta_state_dict instead of model_state_dict."""
    torch.save(
        {
            "delta_state_dict": _sd(v),
            "n_samples": n_samples,
            "master_version": master_version,
        },
        path,
    )


def test_merge_with_lr1_momentum0_reproduces_weighted_average(tmp_path):
    pending_dir = tmp_path / "pending"
    pending_dir.mkdir()
    master_path = tmp_path / "master.pt"
    momentum_path = tmp_path / "momentum.pt"

    _write_master(master_path, 1.0)
    _write_worker(pending_dir / "run1_0_a.pt", 0.0, n_samples=100)
    _write_worker(pending_dir / "run1_0_b.pt", 2.0, n_samples=100)

    merge_round(
        pending_dir,
        master_path,
        momentum_path,
        run_id="run1",
        lr=1.0,
        momentum=0.0,
        weighting="data",
    )

    merged = torch.load(master_path, weights_only=False)["model_state_dict"]
    assert merged["w"][0, 0].item() == pytest.approx(1.0)


def test_merge_round_reads_delta_state_dict_directly_when_present(tmp_path):
    """Phase 3.2: a worker that already pushed master-worker doesn't need
    merge_round to recompute it -- at lr=1, momentum=0 the outer step is
    exactly master - delta, same equivalence the full-weight path relies on."""
    pending_dir = tmp_path / "pending"
    pending_dir.mkdir()
    master_path = tmp_path / "master.pt"
    momentum_path = tmp_path / "momentum.pt"

    _write_master(master_path, 1.0)
    _write_worker_delta(pending_dir / "run1_0_a.pt", 1.0, n_samples=100)

    merge_round(
        pending_dir,
        master_path,
        momentum_path,
        run_id="run1",
        lr=1.0,
        momentum=0.0,
    )

    merged = torch.load(master_path, weights_only=False)["model_state_dict"]
    assert merged["w"][0, 0].item() == pytest.approx(0.0)


def test_merge_round_mixes_delta_and_full_weight_checkpoints_in_one_round(tmp_path):
    """A round-0 straggler merged late still pushes full weights (no master
    to have diffed against yet) alongside nodes already on the delta path --
    both must contribute to the same merge."""
    pending_dir = tmp_path / "pending"
    pending_dir.mkdir()
    master_path = tmp_path / "master.pt"
    momentum_path = tmp_path / "momentum.pt"

    _write_master(master_path, 1.0)
    _write_worker_delta(pending_dir / "run1_0_a.pt", 1.0, n_samples=100)
    _write_worker(pending_dir / "run1_0_b.pt", 0.0, n_samples=100)  # equivalent delta

    result = merge_round(
        pending_dir,
        master_path,
        momentum_path,
        run_id="run1",
        lr=1.0,
        momentum=0.0,
    )

    assert result["n_workers"] == 2
    merged = torch.load(master_path, weights_only=False)["model_state_dict"]
    assert merged["w"][0, 0].item() == pytest.approx(0.0)


def test_merge_round_drops_a_checkpoint_pushed_against_a_stale_master(tmp_path):
    pending_dir = tmp_path / "pending"
    pending_dir.mkdir()
    master_path = tmp_path / "master.pt"
    momentum_path = tmp_path / "momentum.pt"

    _write_master(master_path, 1.0, master_version=5)
    _write_worker_delta(
        pending_dir / "run1_0_a.pt", 1.0, n_samples=100, master_version=5
    )
    # Stale: computed against an older master, must be dropped, not merged in.
    _write_worker_delta(
        pending_dir / "run1_0_b.pt", 999.0, n_samples=100, master_version=4
    )

    result = merge_round(
        pending_dir,
        master_path,
        momentum_path,
        run_id="run1",
        lr=1.0,
        momentum=0.0,
    )

    assert set(result["alphas"]) == {"a"}
    merged = torch.load(master_path, weights_only=False)["model_state_dict"]
    assert merged["w"][0, 0].item() == pytest.approx(0.0)


def test_alphas_follow_sample_counts_not_val_loss(tmp_path):
    pending_dir = tmp_path / "pending"
    pending_dir.mkdir()
    master_path = tmp_path / "master.pt"
    momentum_path = tmp_path / "momentum.pt"

    _write_master(master_path, 1.0)
    _write_worker(pending_dir / "run1_0_a.pt", 0.5, n_samples=300, val_loss=9.0)
    _write_worker(pending_dir / "run1_0_b.pt", 0.5, n_samples=100, val_loss=0.1)

    result = merge_round(
        pending_dir,
        master_path,
        momentum_path,
        run_id="run1",
        lr=0.7,
        momentum=0.9,
        weighting="data",
    )

    assert result["alphas"]["a"] == pytest.approx(0.75)
    assert result["alphas"]["b"] == pytest.approx(0.25)


def test_momentum_buffer_is_written_and_reloaded(tmp_path, capsys):
    pending_dir = tmp_path / "pending"
    pending_dir.mkdir()
    master_path = tmp_path / "master.pt"
    momentum_path = tmp_path / "momentum.pt"

    _write_master(master_path, 1.0)
    _write_worker(pending_dir / "run1_0_a.pt", 1.1, n_samples=100)

    merge_round(
        pending_dir,
        master_path,
        momentum_path,
        run_id="run1",
        lr=0.7,
        momentum=0.9,
        weighting="data",
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
        pending_dir,
        master_path,
        momentum_path,
        run_id="run1",
        lr=0.7,
        momentum=0.9,
        weighting="data",
    )
    m2 = torch.load(master_path, weights_only=False)["model_state_dict"]["w"][
        0, 0
    ].item()

    assert m2 != pytest.approx(m1)


def test_checkpoint_with_neither_state_dict_key_is_rejected_loudly(tmp_path):
    pending_dir = tmp_path / "pending"
    pending_dir.mkdir()
    master_path = tmp_path / "master.pt"
    momentum_path = tmp_path / "momentum.pt"

    _write_master(master_path, 1.0)
    torch.save({"n_samples": 100}, pending_dir / "run1_0_a.pt")

    with pytest.raises(ValueError, match="exactly one of"):
        merge_round(
            pending_dir,
            master_path,
            momentum_path,
            run_id="run1",
            lr=0.7,
            momentum=0.9,
        )


def test_checkpoint_with_both_state_dict_keys_is_rejected_loudly(tmp_path):
    pending_dir = tmp_path / "pending"
    pending_dir.mkdir()
    master_path = tmp_path / "master.pt"
    momentum_path = tmp_path / "momentum.pt"

    _write_master(master_path, 1.0)
    torch.save(
        {"model_state_dict": _sd(0.5), "delta_state_dict": _sd(0.5), "n_samples": 100},
        pending_dir / "run1_0_a.pt",
    )

    with pytest.raises(ValueError, match="exactly one of"):
        merge_round(
            pending_dir,
            master_path,
            momentum_path,
            run_id="run1",
            lr=0.7,
            momentum=0.9,
        )


def test_checkpoint_without_n_samples_is_rejected_loudly(tmp_path):
    pending_dir = tmp_path / "pending"
    pending_dir.mkdir()
    master_path = tmp_path / "master.pt"
    momentum_path = tmp_path / "momentum.pt"

    _write_master(master_path, 1.0)
    _write_worker(pending_dir / "run1_0_a.pt", 0.5, omit_n_samples=True)

    with pytest.raises(ValueError, match="n_samples"):
        merge_round(
            pending_dir,
            master_path,
            momentum_path,
            run_id="run1",
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
            run_id="run1",
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
        pending_dir / "run1_0_a.pt",
    )

    checkpoints = _load_worker_checkpoints(pending_dir, "run1")

    assert checkpoints["a"]["model_state_dict"]["w"].dtype == torch.float32


def test_inverse_loss_weighting_is_available_as_an_option(tmp_path):
    pending_dir = tmp_path / "pending"
    pending_dir.mkdir()
    master_path = tmp_path / "master.pt"
    momentum_path = tmp_path / "momentum.pt"

    _write_master(master_path, 1.0)
    _write_worker(pending_dir / "run1_0_a.pt", 0.5, n_samples=100, val_loss=1.0)
    _write_worker(pending_dir / "run1_0_b.pt", 0.5, n_samples=100, val_loss=3.0)

    result = merge_round(
        pending_dir,
        master_path,
        momentum_path,
        run_id="run1",
        lr=0.7,
        momentum=0.9,
        weighting="inverse_loss",
    )

    assert result["alphas"]["a"] > result["alphas"]["b"]


def test_inverse_loss_weighting_rejects_nan_val_loss(tmp_path):
    """evaluate_val_loss (mesh_round.py) legitimately returns NaN when every
    val window in a round is non-finite. `val_loss <= 0` does not catch NaN
    (NaN comparisons are always False), so without an explicit isfinite
    check a NaN val_loss silently propagates through 1.0/val_loss into
    every node's alpha weight -- corrupting the whole merged master rather
    than failing loud like every other malformed-checkpoint case here.
    """
    pending_dir = tmp_path / "pending"
    pending_dir.mkdir()
    master_path = tmp_path / "master.pt"
    momentum_path = tmp_path / "momentum.pt"

    _write_master(master_path, 1.0)
    _write_worker(pending_dir / "run1_0_a.pt", 0.5, n_samples=100, val_loss=1.0)
    _write_worker(
        pending_dir / "run1_0_b.pt", 0.5, n_samples=100, val_loss=float("nan")
    )

    with pytest.raises(ValueError, match="non-finite val_loss"):
        merge_round(
            pending_dir,
            master_path,
            momentum_path,
            run_id="run1",
            lr=0.7,
            momentum=0.9,
            weighting="inverse_loss",
        )


def test_token_weighting_prefers_fewer_steps_per_sample_over_equal_data_weights(
    tmp_path,
):
    """mesh/quorum.py's token_weights ("quantity x quality", Decoupled DiLoCo
    2.2) must actually change the merge outcome vs weighting="data": two
    nodes with equal n_samples but different n_steps are NOT equal under
    token weighting -- the one that needed fewer steps to process the same
    data (a healthier, better-fed worker) gets more say."""
    pending_dir = tmp_path / "pending"
    pending_dir.mkdir()
    master_path = tmp_path / "master.pt"
    momentum_path = tmp_path / "momentum.pt"

    _write_master(master_path, 1.0)
    _write_worker(pending_dir / "run1_0_a.pt", 0.5, n_samples=100, n_steps=50)
    _write_worker(pending_dir / "run1_0_b.pt", 0.5, n_samples=100, n_steps=200)

    result = merge_round(
        pending_dir,
        master_path,
        momentum_path,
        run_id="run1",
        lr=0.7,
        momentum=0.9,
        weighting="token",
    )

    assert result["alphas"]["a"] > result["alphas"]["b"]


def test_token_weighting_rejects_missing_n_steps(tmp_path):
    """n_steps is optional for weighting='data'/'inverse_loss' (they never
    read it), but weighting='token' needs it -- must fail loud rather than
    silently falling back to an unweighted or wrong merge, matching the
    n_samples/val_loss validation pattern above."""
    pending_dir = tmp_path / "pending"
    pending_dir.mkdir()
    master_path = tmp_path / "master.pt"
    momentum_path = tmp_path / "momentum.pt"

    _write_master(master_path, 1.0)
    _write_worker(pending_dir / "run1_0_a.pt", 0.5, n_samples=100)  # no n_steps

    with pytest.raises(ValueError, match="n_steps"):
        merge_round(
            pending_dir,
            master_path,
            momentum_path,
            run_id="run1",
            lr=0.7,
            momentum=0.9,
            weighting="token",
        )


def test_merge_ignores_other_runs_checkpoints_in_the_shared_pending_dir(tmp_path):
    """pending/ is one flat directory shared by every concurrent run
    (mesh_tick.py's _do_merging passes it unscoped), so a merge that globs
    "*.pt" averages sibling runs -- and checkpoints left behind by finished
    runs -- into this run's master. Every other glob in mesh_tick.py is
    already run-scoped; this is the one path that was not.
    """
    pending_dir = tmp_path / "pending"
    pending_dir.mkdir()
    master_path = tmp_path / "master.pt"
    momentum_path = tmp_path / "momentum.pt"

    _write_master(master_path, 1.0)
    _write_worker(pending_dir / "runA_0_a.pt", 0.0, n_samples=100)
    _write_worker(pending_dir / "runA_0_b.pt", 2.0, n_samples=100)
    # A concurrent run and a finished run's leftovers, same directory.
    _write_worker(pending_dir / "runB_0_a.pt", 99.0, n_samples=100)
    _write_worker(pending_dir / "dead-run_0_z.pt", -99.0, n_samples=100)

    result = merge_round(
        pending_dir,
        master_path,
        momentum_path,
        run_id="runA",
        lr=1.0,
        momentum=0.0,
        weighting="data",
    )

    assert result["n_workers"] == 2
    assert set(result["alphas"]) == {"a", "b"}
    # lr=1, momentum=0 over deltas 1.0 and -1.0 -> master unchanged at 1.0.
    # If runB/dead-run leaked in, this lands nowhere near 1.0.
    merged = torch.load(master_path, weights_only=False)["model_state_dict"]
    assert merged["w"][0, 0].item() == pytest.approx(1.0)


def test_merge_raises_when_only_other_runs_checkpoints_are_present(tmp_path):
    pending_dir = tmp_path / "pending"
    pending_dir.mkdir()
    master_path = tmp_path / "master.pt"
    momentum_path = tmp_path / "momentum.pt"

    _write_master(master_path, 1.0)
    _write_worker(pending_dir / "runB_0_a.pt", 0.5, n_samples=100)

    with pytest.raises(ValueError, match="no pending"):
        merge_round(
            pending_dir,
            master_path,
            momentum_path,
            run_id="runA",
            lr=0.7,
            momentum=0.9,
            weighting="data",
        )


def test_run_id_scoping_does_not_match_hyphen_suffixed_sibling_runs(tmp_path):
    """ "pool-fish" must not pick up "pool-fish-5r"'s checkpoints -- the
    literal "_" in the glob is what separates them (same reasoning as
    mesh_tick.py's _clean_stale_run_artifacts docstring)."""
    pending_dir = tmp_path / "pending"
    pending_dir.mkdir()
    master_path = tmp_path / "master.pt"
    momentum_path = tmp_path / "momentum.pt"

    _write_master(master_path, 1.0)
    _write_worker(pending_dir / "pool-fish_0_a.pt", 0.5, n_samples=100)
    _write_worker(pending_dir / "pool-fish-5r_0_a.pt", 9.0, n_samples=100)

    result = merge_round(
        pending_dir,
        master_path,
        momentum_path,
        run_id="pool-fish",
        lr=0.7,
        momentum=0.9,
        weighting="data",
    )

    assert set(result["alphas"]) == {"a"}


def test_broadcast_dtype_fp32_is_the_default_and_keeps_master_fp32(tmp_path):
    pending_dir = tmp_path / "pending"
    pending_dir.mkdir()
    master_path = tmp_path / "master.pt"
    _write_master(master_path, 1.0)
    _write_worker(pending_dir / "run1_0_a.pt", 0.5, n_samples=100)

    merge_round(
        pending_dir,
        master_path,
        tmp_path / "momentum.pt",
        run_id="run1",
        lr=0.7,
        momentum=0.9,
    )

    sd = torch.load(master_path, weights_only=False)["model_state_dict"]
    assert sd["w"].dtype == torch.float32


def test_broadcast_dtype_fp16_halves_master_and_still_merges(tmp_path):
    pending_dir = tmp_path / "pending"
    pending_dir.mkdir()
    master_path = tmp_path / "master.pt"
    _write_master(master_path, 1.0)
    _write_worker(pending_dir / "run1_1_a.pt", 0.5, n_samples=100)

    merge_round(
        pending_dir,
        master_path,
        tmp_path / "momentum.pt",
        run_id="run1",
        lr=0.7,
        momentum=0.9,
        broadcast_dtype="fp16",
        residual_path=tmp_path / "residual.pt",
    )

    sd = torch.load(master_path, weights_only=False)["model_state_dict"]
    assert sd["w"].dtype == torch.float16
    # A second merge must read the fp16 master back and still do fp32 math.
    merge_round(
        pending_dir,
        master_path,
        tmp_path / "momentum.pt",
        run_id="run1",
        lr=0.7,
        momentum=0.9,
        round_num=1,
        broadcast_dtype="fp16",
        residual_path=tmp_path / "residual.pt",
    )
    assert torch.isfinite(
        torch.load(master_path, weights_only=False)["model_state_dict"]["w"]
    ).all()


def test_merge_round_stamps_master_version_from_round_num(tmp_path):
    """Phase 3.2: the version a worker's pushed delta was computed against
    has to be checkable against the master it would merge into -- merge_round
    is the only writer of master.pt, so it stamps the round it just wrote."""
    pending_dir = tmp_path / "pending"
    pending_dir.mkdir()
    master_path = tmp_path / "master.pt"
    _write_master(master_path, 1.0)
    _write_worker(pending_dir / "run1_3_a.pt", 0.5, n_samples=100)

    merge_round(
        pending_dir,
        master_path,
        tmp_path / "momentum.pt",
        run_id="run1",
        lr=0.7,
        momentum=0.9,
        round_num=3,
        allow_missing_momentum=True,
    )

    assert torch.load(master_path, weights_only=False)["master_version"] == 3


def test_merge_round_stamps_none_master_version_when_round_num_is_none(tmp_path):
    pending_dir = tmp_path / "pending"
    pending_dir.mkdir()
    master_path = tmp_path / "master.pt"
    _write_master(master_path, 1.0)
    _write_worker(pending_dir / "run1_0_a.pt", 0.5, n_samples=100)

    merge_round(
        pending_dir,
        master_path,
        tmp_path / "momentum.pt",
        run_id="run1",
        lr=0.7,
        momentum=0.9,
    )

    assert torch.load(master_path, weights_only=False)["master_version"] is None


def test_error_feedback_removes_the_fp16_rounding_bias_over_rounds(tmp_path):
    """Without feedback every round rounds the same direction, so the
    broadcast is biased by up to half an fp16 ulp forever. With feedback the
    residual dithers the broadcast so its running mean converges on the true
    fp32 value -- which is the whole reason this is safe to enable across a
    30-round run."""
    residual_path = tmp_path / "residual.pt"
    # Sits 30% of the way between two fp16 grid points near 1.0 (ulp ~9.77e-4).
    true_value = 1.0 + 0.3 * (2**-10)
    master_sd = {"w": torch.full((4, 4), true_value, dtype=torch.float32)}

    plain = master_sd["w"].half().float()
    plain_err = (plain - master_sd["w"]).abs().mean().item()

    total = torch.zeros_like(master_sd["w"])
    n_rounds = 100
    for _ in range(n_rounds):
        bc = _quantize_broadcast_with_error_feedback(master_sd, residual_path)
        total += bc["w"].float()
    feedback_err = (total / n_rounds - master_sd["w"]).abs().mean().item()

    assert plain_err > 2e-4  # the bias a plain cast would lock in
    assert feedback_err < plain_err / 10


def test_error_feedback_passes_non_float_buffers_through_untouched(tmp_path):
    master_sd = {
        "w": torch.full((2, 2), 0.5),
        "step": torch.tensor([7], dtype=torch.int64),
    }
    bc = _quantize_broadcast_with_error_feedback(master_sd, tmp_path / "r.pt")
    assert bc["step"].dtype == torch.int64
    assert bc["step"].item() == 7
    assert bc["w"].dtype == torch.float16


def test_error_feedback_survives_a_missing_residual_file(tmp_path):
    master_sd = {"w": torch.full((2, 2), 0.5)}
    bc = _quantize_broadcast_with_error_feedback(master_sd, tmp_path / "absent.pt")
    assert bc["w"].dtype == torch.float16


def test_error_feedback_drops_a_stale_residual_of_the_wrong_shape(tmp_path):
    residual_path = tmp_path / "residual.pt"
    torch.save({"residual": {"w": torch.zeros(9, 9)}}, residual_path)
    master_sd = {"w": torch.full((2, 2), 0.5)}

    bc = _quantize_broadcast_with_error_feedback(master_sd, residual_path)

    assert bc["w"].shape == (2, 2)


def test_missing_momentum_past_round_zero_is_fatal(tmp_path):
    pending_dir = tmp_path / "pending"
    pending_dir.mkdir()
    master_path = tmp_path / "master.pt"
    _write_master(master_path, 1.0)
    _write_worker(pending_dir / "run1_3_a.pt", 0.5, n_samples=100)

    with pytest.raises(ValueError, match="momentum"):
        merge_round(
            pending_dir,
            master_path,
            tmp_path / "gone.pt",
            run_id="run1",
            lr=0.7,
            momentum=0.9,
            round_num=3,
        )


def test_missing_momentum_at_round_zero_still_only_warns(tmp_path, capsys):
    pending_dir = tmp_path / "pending"
    pending_dir.mkdir()
    master_path = tmp_path / "master.pt"
    _write_master(master_path, 1.0)
    _write_worker(pending_dir / "run1_0_a.pt", 0.5, n_samples=100)

    merge_round(
        pending_dir,
        master_path,
        tmp_path / "momentum.pt",
        run_id="run1",
        lr=0.7,
        momentum=0.9,
        round_num=0,
    )

    assert "momentum" in capsys.readouterr().err.lower()


def test_missing_momentum_past_round_zero_can_be_overridden(tmp_path):
    pending_dir = tmp_path / "pending"
    pending_dir.mkdir()
    master_path = tmp_path / "master.pt"
    _write_master(master_path, 1.0)
    _write_worker(pending_dir / "run1_3_a.pt", 0.5, n_samples=100)

    result = merge_round(
        pending_dir,
        master_path,
        tmp_path / "momentum.pt",
        run_id="run1",
        lr=0.7,
        momentum=0.9,
        round_num=3,
        allow_missing_momentum=True,
    )

    assert result["n_workers"] == 1


def _cli_argv(pending_dir, master_path, momentum_path, run_id, *, extra=()):
    return [
        "--pending-dir",
        str(pending_dir),
        "--run-id",
        run_id,
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
        *extra,
    ]


def test_metrics_out_writes_one_line_with_contributors_matching_alphas(tmp_path):
    """0c: per-round merge data (contributor set, alpha weights) otherwise
    exists only in watcher stdout -- unworkable to scrape across an 8-run
    sweep."""
    pending_dir = tmp_path / "pending"
    pending_dir.mkdir()
    master_path = tmp_path / "master.pt"
    momentum_path = tmp_path / "momentum.pt"
    metrics_out = tmp_path / "results" / "run1_rounds.jsonl"

    _write_master(master_path, 1.0)
    _write_worker(pending_dir / "run1_7_a.pt", 0.0, n_samples=100)
    _write_worker(pending_dir / "run1_7_b.pt", 2.0, n_samples=100)

    rc = main(
        _cli_argv(
            pending_dir,
            master_path,
            momentum_path,
            "run1",
            extra=[
                "--round",
                "7",
                "--allow-missing-momentum",
                "--metrics-out",
                str(metrics_out),
            ],
        )
    )

    assert rc == 0
    lines = metrics_out.read_text().splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert record["run_id"] == "run1"
    assert record["round"] == 7
    assert record["n_contributors"] == 2
    assert record["contributors"] == sorted(record["alphas"])
    assert set(record["alphas"]) == {"a", "b"}
    assert record["timeout_forced"] is False
    assert isinstance(record["ts"], float)


def test_metrics_out_omitted_creates_no_file(tmp_path):
    """Back-compat: existing callers that never pass --metrics-out must see
    no behaviour change."""
    pending_dir = tmp_path / "pending"
    pending_dir.mkdir()
    master_path = tmp_path / "master.pt"
    momentum_path = tmp_path / "momentum.pt"
    results_dir = tmp_path / "results"

    _write_master(master_path, 1.0)
    _write_worker(pending_dir / "run1_0_a.pt", 0.5, n_samples=100)

    rc = main(_cli_argv(pending_dir, master_path, momentum_path, "run1"))

    assert rc == 0
    assert not results_dir.exists()


def test_metrics_out_records_timeout_forced_flag(tmp_path):
    pending_dir = tmp_path / "pending"
    pending_dir.mkdir()
    master_path = tmp_path / "master.pt"
    momentum_path = tmp_path / "momentum.pt"
    metrics_out = tmp_path / "run1_rounds.jsonl"

    _write_master(master_path, 1.0)
    _write_worker(pending_dir / "run1_1_a.pt", 0.5, n_samples=100)

    rc = main(
        _cli_argv(
            pending_dir,
            master_path,
            momentum_path,
            "run1",
            extra=[
                "--round",
                "1",
                "--allow-missing-momentum",
                "--metrics-out",
                str(metrics_out),
                "--timeout-forced",
            ],
        )
    )

    assert rc == 0
    record = json.loads(metrics_out.read_text().splitlines()[0])
    assert record["timeout_forced"] is True


class _RecordingWriter:
    def __init__(self):
        self.calls = []

    def add_scalar(self, tag, value, step):
        self.calls.append((tag, value, step))

    def flush(self):
        pass

    def close(self):
        pass


def test_tb_dir_writes_merge_contributor_scalars_and_replays_jsonl(
    tmp_path, monkeypatch
):
    pending_dir = tmp_path / "mesh_shared" / "checkpoints" / "pending"
    pending_dir.mkdir(parents=True)
    master_path = tmp_path / "master.pt"
    momentum_path = tmp_path / "momentum.pt"
    tb_dir = tmp_path / "tb"

    _write_master(master_path, 1.0)
    _write_worker(pending_dir / "run1_2_a.pt", 0.0, n_samples=100)
    _write_worker(pending_dir / "run1_2_b.pt", 2.0, n_samples=100)

    replay_calls = []
    monkeypatch.setattr(
        "video_encoding.scripts.run_outer_step.write_mesh_round_scalars",
        lambda writer, mesh_shared, run_id, last_round: (
            replay_calls.append((mesh_shared, run_id, last_round)) or 0
        ),
    )
    recording_writer = _RecordingWriter()
    monkeypatch.setattr(
        "video_encoding.scripts.run_outer_step.make_writer",
        lambda log_dir, rank=0: recording_writer,
    )

    rc = main(
        _cli_argv(
            pending_dir,
            master_path,
            momentum_path,
            "run1",
            extra=[
                "--round",
                "2",
                "--allow-missing-momentum",
                "--tb-dir",
                str(tb_dir),
            ],
        )
    )

    assert rc == 0
    assert replay_calls == [(pending_dir.parent.parent, "run1", 0)]
    tags = {c[0]: c for c in recording_writer.calls}
    assert tags["merge/n_contributors"] == ("merge/n_contributors", 2, 2)
    assert tags["merge/timeout_forced"] == ("merge/timeout_forced", 0.0, 2)


def test_tb_dir_writes_per_node_alpha_scalars(tmp_path, monkeypatch):
    pending_dir = tmp_path / "mesh_shared" / "checkpoints" / "pending"
    pending_dir.mkdir(parents=True)
    master_path = tmp_path / "master.pt"
    momentum_path = tmp_path / "momentum.pt"
    tb_dir = tmp_path / "tb"

    _write_master(master_path, 1.0)
    _write_worker(pending_dir / "run1_2_a.pt", 0.0, n_samples=100)
    _write_worker(pending_dir / "run1_2_b.pt", 2.0, n_samples=300)

    monkeypatch.setattr(
        "video_encoding.scripts.run_outer_step.write_mesh_round_scalars",
        lambda writer, mesh_shared, run_id, last_round: 0,
    )
    recording_writer = _RecordingWriter()
    monkeypatch.setattr(
        "video_encoding.scripts.run_outer_step.make_writer",
        lambda log_dir, rank=0: recording_writer,
    )

    rc = main(
        _cli_argv(
            pending_dir,
            master_path,
            momentum_path,
            "run1",
            extra=[
                "--round",
                "2",
                "--allow-missing-momentum",
                "--tb-dir",
                str(tb_dir),
            ],
        )
    )

    assert rc == 0
    tags = {c[0]: c[1] for c in recording_writer.calls}
    # data weighting: a's alpha=100/400=0.25, b's alpha=300/400=0.75
    assert tags["merge/alpha/a"] == 0.25
    assert tags["merge/alpha/b"] == 0.75


def test_tb_dir_persists_last_round_sidecar_after_replay(tmp_path, monkeypatch):
    pending_dir = tmp_path / "mesh_shared" / "checkpoints" / "pending"
    pending_dir.mkdir(parents=True)
    master_path = tmp_path / "master.pt"
    momentum_path = tmp_path / "momentum.pt"
    tb_dir = tmp_path / "tb"

    _write_master(master_path, 1.0)
    _write_worker(pending_dir / "run1_0_a.pt", 0.5, n_samples=100)

    monkeypatch.setattr(
        "video_encoding.scripts.run_outer_step.write_mesh_round_scalars",
        lambda writer, mesh_shared, run_id, last_round: 3,
    )

    rc = main(
        _cli_argv(
            pending_dir,
            master_path,
            momentum_path,
            "run1",
            extra=[
                "--round",
                "0",
                "--allow-missing-momentum",
                "--tb-dir",
                str(tb_dir),
            ],
        )
    )

    assert rc == 0
    assert (tb_dir.parent / "tb_last_round.txt").read_text().strip() == "3"


def test_tb_dir_reads_existing_sidecar_as_starting_last_round(tmp_path, monkeypatch):
    pending_dir = tmp_path / "mesh_shared" / "checkpoints" / "pending"
    pending_dir.mkdir(parents=True)
    master_path = tmp_path / "master.pt"
    momentum_path = tmp_path / "momentum.pt"
    tb_dir = tmp_path / "tb"
    tb_dir.parent.mkdir(parents=True, exist_ok=True)
    (tb_dir.parent / "tb_last_round.txt").write_text("5")

    _write_master(master_path, 1.0)
    _write_worker(pending_dir / "run1_0_a.pt", 0.5, n_samples=100)

    seen = []
    monkeypatch.setattr(
        "video_encoding.scripts.run_outer_step.write_mesh_round_scalars",
        lambda writer, mesh_shared, run_id, last_round: (
            seen.append(last_round) or last_round
        ),
    )

    rc = main(
        _cli_argv(
            pending_dir,
            master_path,
            momentum_path,
            "run1",
            extra=[
                "--round",
                "0",
                "--allow-missing-momentum",
                "--tb-dir",
                str(tb_dir),
            ],
        )
    )

    assert rc == 0
    assert seen == [5]


def test_tb_dir_write_failure_does_not_fail_the_merge(tmp_path, monkeypatch, capsys):
    pending_dir = tmp_path / "mesh_shared" / "checkpoints" / "pending"
    pending_dir.mkdir(parents=True)
    master_path = tmp_path / "master.pt"
    momentum_path = tmp_path / "momentum.pt"
    tb_dir = tmp_path / "tb"

    _write_master(master_path, 1.0)
    _write_worker(pending_dir / "run1_0_a.pt", 0.5, n_samples=100)

    def _raise(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("video_encoding.scripts.run_outer_step.make_writer", _raise)

    rc = main(
        _cli_argv(
            pending_dir,
            master_path,
            momentum_path,
            "run1",
            extra=[
                "--round",
                "0",
                "--allow-missing-momentum",
                "--tb-dir",
                str(tb_dir),
            ],
        )
    )

    assert rc == 0
    assert "run_outer_step" in capsys.readouterr().err
    merged = torch.load(master_path, weights_only=False)["model_state_dict"]
    # lr=0.7, momentum=0.9 over a nonzero delta -- master must have moved,
    # proving the merge itself completed despite the TB write raising.
    assert merged["w"][0, 0].item() != pytest.approx(1.0)


def test_tb_dir_omitted_creates_no_tb_directory(tmp_path):
    pending_dir = tmp_path / "mesh_shared" / "checkpoints" / "pending"
    pending_dir.mkdir(parents=True)
    master_path = tmp_path / "master.pt"
    momentum_path = tmp_path / "momentum.pt"
    tb_dir = tmp_path / "tb"

    _write_master(master_path, 1.0)
    _write_worker(pending_dir / "run1_0_a.pt", 0.5, n_samples=100)

    rc = main(_cli_argv(pending_dir, master_path, momentum_path, "run1"))

    assert rc == 0
    assert not tb_dir.exists()


# --- NaN checkpoints must never reach the master ---------------------------


def test_merge_drops_a_worker_whose_weights_are_nan(tmp_path):
    """The 2026-08-20 fault's silent half.

    A node assigned shards it did not hold trained zero samples, so its
    train_loss was nan -- while its val_loss stayed a healthy real number
    (val shards are replicated to every node and always resolved locally).
    The only finiteness check in the merge looked at val_loss, so the nan
    weights averaged straight into master.pt and poisoned every node.
    """
    pending = tmp_path / "pending"
    pending.mkdir()
    _write_master(tmp_path / "master.pt", 0.0)
    _write_worker(pending / "RUN_1_good.pt", 2.0, n_samples=100, val_loss=1.0)
    _write_worker(pending / "RUN_1_nan.pt", float("nan"), n_samples=100, val_loss=1.0)

    result = merge_round(
        pending,
        tmp_path / "master.pt",
        tmp_path / "momentum.pt",
        run_id="RUN",
        lr=1.0,
        momentum=0.0,
        round_num=1,
        allow_missing_momentum=True,
    )

    assert result["n_workers"] == 1
    assert "RUN_1_nan" not in result["alphas"]
    master = torch.load(tmp_path / "master.pt", weights_only=False)
    for tensor in master["model_state_dict"].values():
        assert torch.isfinite(tensor).all()


def test_merge_refuses_a_round_where_every_worker_is_nan(tmp_path):
    """Dropping the last healthy worker would silently merge nothing and
    advance the round, which reads as progress. Fail the round instead."""
    pending = tmp_path / "pending"
    pending.mkdir()
    _write_master(tmp_path / "master.pt", 0.0)
    _write_worker(pending / "RUN_1_a.pt", float("nan"), n_samples=100)
    _write_worker(pending / "RUN_1_b.pt", float("inf"), n_samples=100)

    with pytest.raises(ValueError, match="non-finite"):
        merge_round(
            pending,
            tmp_path / "master.pt",
            tmp_path / "momentum.pt",
            run_id="RUN",
            lr=1.0,
            momentum=0.0,
            round_num=1,
            allow_missing_momentum=True,
        )


def test_a_killed_merge_that_already_stamped_master_is_a_no_op_on_retry(tmp_path):
    """The 2026-08-28 crash loop: mesh_tick.py's `subprocess.run(...,
    timeout=300)` killed the merge *after* `_atomic_save(master_path)` had
    landed but before the process exited 0, so the watcher never unlinked
    pending/ and retried the same round forever. On every retry master.pt is
    already stamped with this round while the pending deltas still carry the
    previous version, so filter_stale_deltas drops all of them and the merge
    died in the weighting function on an empty dict (`total weight is zero
    (stats={})`, 722 times across two runs).

    A merge whose master write already landed has, by definition, already
    been applied: report it as a no-op success so the caller clears pending/
    and advances instead of retrying a round that can never succeed again.
    """
    pending_dir = tmp_path / "pending"
    pending_dir.mkdir()
    master_path = tmp_path / "master.pt"

    _write_master(master_path, 1.0, master_version=27)
    for node in ("a", "b"):
        _write_worker_delta(
            pending_dir / f"run1_27_{node}.pt", 1.0, n_samples=100, master_version=26
        )

    result = merge_round(
        pending_dir,
        master_path,
        tmp_path / "momentum.pt",
        run_id="run1",
        lr=0.7,
        momentum=0.9,
        round_num=27,
    )

    assert result["already_merged"] is True
    assert result["n_workers"] == 0
    assert result["alphas"] == {}
    # The already-merged master must not be touched a second time.
    payload = torch.load(master_path, weights_only=False)
    assert payload["master_version"] == 27
    assert payload["model_state_dict"]["w"][0, 0].item() == pytest.approx(1.0)


def test_already_merged_round_exits_zero_and_writes_no_metrics(tmp_path):
    """mesh_tick.py unlinks pending/ only on returncode 0, so the recovery
    above is only worth anything if the CLI reports success. The metrics
    JSONL must stay clean of the no-op: a 0-contributor record there would
    read as a real round that merged nothing."""
    pending_dir = tmp_path / "pending"
    pending_dir.mkdir()
    master_path = tmp_path / "master.pt"
    metrics_out = tmp_path / "rounds.jsonl"

    _write_master(master_path, 1.0, master_version=4)
    _write_worker_delta(
        pending_dir / "run1_4_a.pt", 1.0, n_samples=100, master_version=3
    )

    rc = main(
        [
            "--pending-dir",
            str(pending_dir),
            "--run-id",
            "run1",
            "--master",
            str(master_path),
            "--momentum",
            str(tmp_path / "momentum.pt"),
            "--round",
            "4",
            "--metrics-out",
            str(metrics_out),
        ]
    )

    assert rc == 0
    assert not metrics_out.exists()


def test_bootstrap_master_version_does_not_short_circuit_round_zero(tmp_path):
    """_bootstrap_master (watcher/mesh_tick.py) seeds master.pt before round
    0's own merge runs, so its sentinel version must be one no round_num can
    ever equal -- otherwise the already-merged recovery above would skip the
    first real merge of every run."""
    pending_dir = tmp_path / "pending"
    pending_dir.mkdir()
    master_path = tmp_path / "master.pt"

    _write_master(master_path, 1.0, master_version=BOOTSTRAP_MASTER_VERSION)
    _write_worker(pending_dir / "run1_0_a.pt", 0.5, n_samples=100)

    result = merge_round(
        pending_dir,
        master_path,
        tmp_path / "momentum.pt",
        run_id="run1",
        lr=1.0,
        momentum=0.0,
        round_num=0,
    )

    assert result["n_workers"] == 1
    assert torch.load(master_path, weights_only=False)["master_version"] == 0


def test_all_stale_checkpoints_report_the_version_mismatch(tmp_path):
    """Not the already-merged case (master is ahead of neither this round nor
    the deltas by one) -- this is a genuinely unmergeable set, and it has to
    say so instead of falling through to `total weight is zero (n_samples={})`
    from inside the weighting function."""
    pending_dir = tmp_path / "pending"
    pending_dir.mkdir()
    master_path = tmp_path / "master.pt"

    _write_master(master_path, 1.0, master_version=27)
    _write_worker_delta(
        pending_dir / "run1_28_a.pt", 1.0, n_samples=100, master_version=25
    )

    with pytest.raises(ValueError, match="stale master"):
        merge_round(
            pending_dir,
            master_path,
            tmp_path / "momentum.pt",
            run_id="run1",
            lr=0.7,
            momentum=0.9,
            round_num=28,
            allow_missing_momentum=True,
        )
