"""Plan 6 DiLoCo checks 1-5: pure CPU state_dict arithmetic over one captured round.

Every fixture is a synthetic state_dict written in Plan 1's snapshot layout
(`master_round<N>.pt`, `pending_round<N>/{run_id}_{N}_{node}.pt`,
`outer_momentum_round<N>.pt`, one `manifest.jsonl` record per round), so the
loader is exercised against the same files `snapshot_mesh_checkpoints.sh`
produces rather than against in-memory dicts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from stern.mesh.outer_step import average_deltas
from stern.models.videomae.mae import VideoMAEForPreTraining
from stern.evaluation.diagnostics.diloco import (
    MergeInputs,
    _verdict_meta,
    agreement_matrix,
    check,
    effective_outer_lr,
    effective_replicas,
    history_only,
    load_merge_inputs,
    merge_magnitude,
    momentum_alignment,
    per_block_divergence,
    quantization_noise_floor,
    synthetic_controls,
)
from stern.evaluation.diagnostics import diloco as diag_diloco

MODEL_CONF = Path(__file__).resolve().parents[2] / "configs" / "model" / "videomae_base.yaml"


def _write_snapshot(
    tmp_path,
    run_id,
    round,
    nodes,
    *,
    master,
    master_version=3,
    momentum=None,
    gaps=(),
):
    """Write one captured round the way Plan 1's snapshot script lays it out.

    `nodes` maps node name -> the raw pending payload dict, saved verbatim so a
    test can omit keys the loader must refuse on.
    """
    torch.save(
        {"model_state_dict": master, "master_version": master_version},
        tmp_path / f"master_round{round}.pt",
    )
    pending_dir = tmp_path / f"pending_round{round}"
    pending_dir.mkdir()
    pending = []
    for node, payload in nodes.items():
        torch.save(payload, pending_dir / f"{run_id}_{round}_{node}.pt")
        kind = next(
            (k for k in ("delta_state_dict", "model_state_dict") if k in payload),
            "unknown",
        )
        pending.append(
            {
                "node": node,
                "master_version": payload.get("master_version"),
                "payload_kind": kind,
                "sha256": "0" * 64,
            }
        )
    momentum_name = None
    if momentum is not None:
        momentum_name = f"outer_momentum_round{round}.pt"
        torch.save({"momentum_buf": momentum}, tmp_path / momentum_name)
    record = {
        "run_id": run_id,
        "round": round,
        "captured_at": "2026-09-30T00:00:00+00:00",
        "expected_nodes": sorted(nodes),
        "master": f"master_round{round}.pt",
        "master_sha256": "0" * 64,
        "pending": sorted(pending, key=lambda entry: entry["node"]),
        "momentum": momentum_name,
        "broadcast_residual": None,
        "push_residuals": {},
        "gaps": list(gaps),
    }
    with (tmp_path / "manifest.jsonl").open("a") as handle:
        handle.write(json.dumps(record) + "\n")
    return tmp_path


def _model_payload(sd, *, master_version=3, n_samples=100):
    return {
        "model_state_dict": sd,
        "master_version": master_version,
        "n_samples": n_samples,
        "n_steps": 10,
        "val_loss": 0.5,
    }


def _delta_payload(sd, *, master_version=3, n_samples=100):
    return {
        "delta_state_dict": sd,
        "master_version": master_version,
        "n_samples": n_samples,
        "n_steps": 10,
        "val_loss": 0.5,
    }


def _master():
    torch.manual_seed(0)
    return {"w": torch.randn(4, 4), "b": torch.randn(4)}


def _worker(master, scale=0.1, seed=1):
    gen = torch.Generator().manual_seed(seed)
    return {
        k: v + scale * torch.randn(v.shape, generator=gen) for k, v in master.items()
    }


# --- load_merge_inputs -------------------------------------------------------


def test_merge_inputs_model_and_delta_payloads_yield_the_same_delta(tmp_path):
    master = _master()
    w = _worker(master)
    _write_snapshot(
        tmp_path,
        "run-x",
        4,
        {
            "A": _delta_payload({k: master[k] - w[k] for k in master}),
            "B": _model_payload(w),
        },
        master=master,
    )
    inputs = load_merge_inputs(tmp_path, 4)
    torch.testing.assert_close(inputs.worker_deltas["A"], inputs.worker_deltas["B"])
    assert inputs.payload_kinds == {"A": "delta_state_dict", "B": "model_state_dict"}


def test_merge_inputs_trust_the_file_over_the_manifest_payload_kind(tmp_path):
    master = _master()
    w = _worker(master)
    _write_snapshot(
        tmp_path,
        "run-x",
        4,
        {"A": _delta_payload({k: master[k] - w[k] for k in master})},
        master=master,
    )
    manifest = tmp_path / "manifest.jsonl"
    record = json.loads(manifest.read_text())
    record["pending"][0]["payload_kind"] = "model_state_dict"
    manifest.write_text(json.dumps(record) + "\n")
    inputs = load_merge_inputs(tmp_path, 4)
    assert inputs.payload_kinds == {"A": "delta_state_dict"}
    # Diffing the delta against the master as if it were weights would give
    # master - (master - w) = w, not the delta.
    torch.testing.assert_close(
        inputs.worker_deltas["A"], {k: master[k] - w[k] for k in master}
    )


def test_merge_inputs_drop_and_record_stale_deltas(tmp_path):
    master = _master()
    _write_snapshot(
        tmp_path,
        "run-x",
        4,
        {
            "fresh": _model_payload(_worker(master)),
            "stale": _model_payload(_worker(master, seed=2), master_version=1),
        },
        master=master,
    )
    inputs = load_merge_inputs(tmp_path, 4)
    assert "stale" not in inputs.worker_deltas
    assert inputs.dropped_stale == ("stale",)
    assert inputs.alphas == {"fresh": 1.0}


def test_merge_inputs_unwrap_the_momentum_payload(tmp_path):
    master = _master()
    t = torch.arange(16.0).reshape(4, 4)
    with_momentum = tmp_path / "with"
    with_momentum.mkdir()
    _write_snapshot(
        with_momentum,
        "run-x",
        4,
        {"A": _model_payload(_worker(master))},
        master=master,
        momentum={"w": t},
    )
    assert torch.equal(load_merge_inputs(with_momentum, 4).momentum_buf["w"], t)

    without = tmp_path / "without"
    without.mkdir()
    _write_snapshot(
        without, "run-x", 4, {"A": _model_payload(_worker(master))}, master=master
    )
    assert load_merge_inputs(without, 4).momentum_buf is None


def test_merge_inputs_parse_a_node_name_containing_an_underscore(tmp_path):
    master = _master()
    _write_snapshot(
        tmp_path, "run-x", 4, {"node_a": _model_payload(_worker(master))}, master=master
    )
    assert (tmp_path / "pending_round4" / "run-x_4_node_a.pt").exists()
    assert list(load_merge_inputs(tmp_path, 4).worker_deltas) == ["node_a"]


def test_merge_inputs_refuse_a_payload_with_neither_state_key(tmp_path):
    master = _master()
    _write_snapshot(
        tmp_path,
        "run-x",
        4,
        {"A": {"master_version": 3, "n_samples": 100}},
        master=master,
    )
    with pytest.raises(ValueError, match=r"run-x_4_A\.pt"):
        load_merge_inputs(tmp_path, 4)


def test_merge_inputs_refuse_a_payload_missing_n_samples(tmp_path):
    master = _master()
    payload = _model_payload(_worker(master))
    del payload["n_samples"]
    _write_snapshot(tmp_path, "run-x", 4, {"A": payload}, master=master)
    with pytest.raises(ValueError, match="n_samples"):
        load_merge_inputs(tmp_path, 4)


# --- quantization_noise_floor ------------------------------------------------


def test_floor_is_positive_on_exactly_representable_fp16_deltas():
    gen = torch.Generator().manual_seed(0)
    deltas = {
        name: {"w": torch.randn(64, 64, generator=gen).half().float()}
        for name in ("a", "b")
    }
    floor = quantization_noise_floor(deltas)
    assert floor != 0.0
    # Rounding at fp16's 2**-11 relative precision cannot move a cosine far.
    assert floor < 1e-3


def test_floor_is_zero_for_fewer_than_two_deltas():
    assert quantization_noise_floor({"a": {"w": torch.randn(64, 64)}}) == 0.0


def test_floor_is_deterministic():
    gen = torch.Generator().manual_seed(0)
    deltas = {name: {"w": torch.randn(64, 64, generator=gen)} for name in ("a", "b")}
    assert quantization_noise_floor(deltas) == quantization_noise_floor(deltas)


# --- agreement_matrix / effective_replicas -----------------------------------


def _vec(*values):
    return {"w": torch.tensor(values, dtype=torch.float32)}


def test_agreement_identical_workers():
    d = _vec(1.0, -2.0, 3.0)
    deltas = {n: d for n in ("a", "b", "c")}
    matrix = agreement_matrix(deltas)
    assert matrix.shape == (3, 3)
    np.testing.assert_allclose(matrix, np.ones((3, 3)), atol=1e-5)
    m_eff, _ = effective_replicas(deltas, {n: 1 / 3 for n in deltas})
    assert m_eff == pytest.approx(1.0, abs=1e-4)


def test_agreement_orthogonal_workers():
    deltas = {f"n{i}": {"w": torch.eye(4)[i]} for i in range(4)}
    matrix = agreement_matrix(deltas)
    np.testing.assert_allclose(matrix, np.eye(4), atol=1e-5)
    m_eff, _ = effective_replicas(deltas, {n: 0.25 for n in deltas})
    assert m_eff == pytest.approx(4.0, abs=1e-3)


def test_agreement_anti_aligned_pair():
    d = _vec(1.0, 2.0, -0.5)
    deltas = {"a": d, "b": {"w": -d["w"]}}
    assert agreement_matrix(deltas)[0, 1] == pytest.approx(-1.0, abs=1e-5)
    merged = average_deltas(deltas, {"a": 0.5, "b": 0.5})
    assert float(merged["w"].norm()) == pytest.approx(0.0, abs=1e-6)


def test_m_eff_is_alpha_weighted():
    deltas = {"a": _vec(1.0, 0.0), "b": _vec(0.0, 1.0)}
    even, _ = effective_replicas(deltas, {"a": 0.5, "b": 0.5})
    skewed, _ = effective_replicas(deltas, {"a": 0.9, "b": 0.1})
    assert even == pytest.approx(2.0, abs=1e-4)
    assert skewed == pytest.approx(1.2195, abs=1e-3)


def test_m_eff_weights_the_norm_denominator_by_alpha():
    # Unequal norms make the denominator's weighting visible: sum a_i||d_i|| is
    # 1.9 here, while an unweighted mean of norms would be 1.5.
    deltas = {"a": _vec(2.0, 0.0), "b": _vec(0.0, 1.0)}
    m_eff, _ = effective_replicas(deltas, {"a": 0.9, "b": 0.1})
    assert m_eff == pytest.approx(1.9**2 / 3.25, abs=1e-4)


def test_norm_spread_exposes_a_dominant_worker():
    deltas = {"a": _vec(10.0, 0.0), "b": _vec(0.0, 1.0), "c": _vec(0.0, 1.0)}
    m_eff, norm_spread = effective_replicas(deltas, {n: 1 / 3 for n in deltas})
    assert m_eff < 2.0
    assert norm_spread > 1.0


def test_agreement_block_filter_excludes_decoder_blocks():
    decoder = torch.tensor([1.0, 2.0, 3.0])
    deltas = {
        n: {
            "encoder.blocks.0.attn.weight": torch.zeros(3),
            "decoder_blocks.0.attn.weight": decoder.clone(),
        }
        for n in ("a", "b")
    }
    # A bare "blocks.0." filter matches the decoder key too and would give 1.0.
    assert agreement_matrix(deltas, block=0)[0, 1] == 0.0


# --- per_block_divergence ----------------------------------------------------


def _tiny_mae_cfg():
    cfg = OmegaConf.load(MODEL_CONF)
    cfg.merge_with(
        {
            "depth": 3,
            "embed_dim": 32,
            "num_heads": 4,
            "decoder_embed_dim": 16,
            "decoder_num_heads": 2,
            "decoder_depth": 1,
            "num_frames": 4,
            "image_size": 32,
        }
    )
    return cfg


def test_per_block_divergence_partitions_on_encoder_depth():
    torch.manual_seed(0)
    master = VideoMAEForPreTraining(_tiny_mae_cfg()).state_dict()
    assert any(k.startswith("decoder_blocks.") for k in master)
    deltas = {n: _worker(master, scale=0.01, seed=s) for s, n in enumerate(("a", "b"))}
    deltas = {n: {k: master[k] - w[k] for k in master} for n, w in deltas.items()}
    result = per_block_divergence(deltas, master, {"a": 0.5, "b": 0.5})
    assert set(result) == {0, 1, 2}
    assert set(result[0]) == {"rho", "m_eff", "rel_norm", "mean_cos"}


def test_per_block_divergence_values_on_hand_built_blocks():
    master = {
        "encoder.blocks.0.w": torch.tensor([3.0, 4.0]),
        "encoder.blocks.1.w": torch.tensor([1.0, 0.0]),
        "decoder_blocks.0.w": torch.tensor([1.0, 1.0]),
    }
    deltas = {
        "a": {
            "encoder.blocks.0.w": torch.tensor([1.0, 0.0]),
            "encoder.blocks.1.w": torch.tensor([1.0, 0.0]),
            "decoder_blocks.0.w": torch.tensor([5.0, 5.0]),
        },
        "b": {
            "encoder.blocks.0.w": torch.tensor([0.0, 1.0]),
            "encoder.blocks.1.w": torch.tensor([2.0, 0.0]),
            "decoder_blocks.0.w": torch.tensor([-5.0, 5.0]),
        },
    }
    result = per_block_divergence(deltas, master, {"a": 0.5, "b": 0.5})
    # Block 0: orthogonal unit deltas against a norm-5 master.
    assert result[0]["rho"] == pytest.approx(2**-0.5, abs=1e-6)
    assert result[0]["m_eff"] == pytest.approx(2.0, abs=1e-5)
    assert result[0]["rel_norm"] == pytest.approx(0.2, abs=1e-6)
    assert result[0]["mean_cos"] == pytest.approx(0.0, abs=1e-6)
    # Block 1: parallel deltas of norms 1 and 2 against a unit master.
    assert result[1]["rho"] == pytest.approx(1.0, abs=1e-6)
    assert result[1]["m_eff"] == pytest.approx(1.0, abs=1e-5)
    assert result[1]["rel_norm"] == pytest.approx(1.5, abs=1e-6)
    assert result[1]["mean_cos"] == pytest.approx(1.0, abs=1e-6)


# --- effective_outer_lr / merge_magnitude / momentum_alignment ---------------


def _merge_inputs(master, deltas, momentum_buf=None):
    names = sorted(deltas)
    return MergeInputs(
        master=master,
        master_version=3,
        worker_deltas=deltas,
        alphas={n: 1 / len(names) for n in names},
        momentum_buf=momentum_buf,
        val_losses={},
        dropped_stale=(),
        quant_noise_floor=0.0,
        n_samples={n: 100 for n in names},
        n_steps={n: 10 for n in names},
        payload_kinds={n: "model_state_dict" for n in names},
        gaps=(),
    )


def _encoder_sd(seed, scale=1.0):
    gen = torch.Generator().manual_seed(seed)
    return {
        f"encoder.blocks.{b}.{p}": scale * torch.randn(8, generator=gen)
        for b in range(2)
        for p in ("attn.weight", "mlp.weight")
    }


def test_effective_outer_lr_at_the_cluster_configuration():
    assert effective_outer_lr(0.7, 0.9) == pytest.approx(7.0, abs=1e-6)


def test_merge_magnitude_degenerate_step_equals_the_merged_delta():
    master = _encoder_sd(0)
    deltas = {"a": _encoder_sd(1, 0.1), "b": _encoder_sd(2, 0.1)}
    # Unrelated and large: a hand-rolled `lr*delta + m` step would add it in,
    # while Nesterov at momentum=0 discards the old buffer entirely.
    momentum_buf = _encoder_sd(3, 5.0)
    result = merge_magnitude(
        _merge_inputs(master, deltas, momentum_buf), outer_lr=1.0, outer_momentum=0.0
    )
    merged = average_deltas(deltas, {"a": 0.5, "b": 0.5})
    assert set(result) == {0, 1}
    for block in (0, 1):
        keys = [k for k in master if k.startswith(f"encoder.blocks.{block}.")]
        merged_norm = torch.cat([merged[k] for k in keys]).norm()
        drift = sum(torch.cat([d[k] for k in keys]).norm() for d in deltas.values()) / 2
        assert result[block]["step_over_drift"] == pytest.approx(
            float(merged_norm / drift), abs=1e-5
        )
        master_norm = torch.cat([master[k] for k in keys]).norm()
        assert result[block]["step_over_master"] == pytest.approx(
            float(merged_norm / master_norm), abs=1e-5
        )


def test_merge_magnitude_includes_the_nesterov_look_ahead():
    master = {"encoder.blocks.0.w": torch.tensor([10.0, 0.0])}
    deltas = {n: {"encoder.blocks.0.w": torch.tensor([1.0, 0.0])} for n in ("a", "b")}
    momentum_buf = {"encoder.blocks.0.w": torch.tensor([0.0, 2.0])}
    result = merge_magnitude(
        _merge_inputs(master, deltas, momentum_buf), outer_lr=0.7, outer_momentum=0.9
    )
    # buf = 0.9*[0, 2] + [1, 0] = [1, 1.8]; step = 0.7*([1, 0] + 0.9*buf)
    # = [1.33, 1.134]. Heavy-ball (lr*g + m*old_buf) would give [0.7, 1.8].
    step_norm = (1.33**2 + 1.134**2) ** 0.5
    assert result[0]["step_over_drift"] == pytest.approx(step_norm, abs=1e-5)
    assert result[0]["step_over_master"] == pytest.approx(step_norm / 10, abs=1e-6)


def test_momentum_alignment_is_zero_for_a_zero_buffer():
    merged = _encoder_sd(1)
    zeros = {k: torch.zeros_like(v) for k, v in merged.items()}
    assert momentum_alignment(zeros, merged) == 0.0
    assert momentum_alignment(None, merged) == 0.0


def test_momentum_alignment_is_one_for_an_identical_buffer():
    merged = _encoder_sd(1)
    assert momentum_alignment(dict(merged), merged) == pytest.approx(1.0, abs=1e-5)


# --- synthetic_controls ------------------------------------------------------


def test_synthetic_controls_bracket_ceiling_above_floor():
    gen = torch.Generator().manual_seed(0)
    master = {"w": torch.randn(256, generator=gen)}
    single = {"w": master["w"] - 0.01 * torch.randn(256, generator=gen)}
    controls = synthetic_controls(single, master, n_workers=4)
    assert set(controls) == {"split", "noise"}
    split, noise = controls["split"], controls["noise"]
    assert len(split.worker_deltas) == len(noise.worker_deltas) == 4
    split_m_eff, _ = effective_replicas(split.worker_deltas, split.alphas)
    noise_m_eff, _ = effective_replicas(noise.worker_deltas, noise.alphas)
    assert split_m_eff < 1.5
    assert noise_m_eff > 3.0
    off = ~np.eye(4, dtype=bool)
    assert (
        agreement_matrix(split.worker_deltas)[off].mean()
        > agreement_matrix(noise.worker_deltas)[off].mean()
    )


# --- history_only ------------------------------------------------------------


def test_history_only_counts_contributors_and_loss_dispersion(tmp_path):
    for node, loss, abandoned in (
        ("a", 0.50, False),
        ("b", 0.51, False),
        ("c", 0.52, False),
        ("d", float("nan"), True),
    ):
        record = {
            "round": 1,
            "val_loss": loss,
            "n_samples": 100,
            "n_steps": 10,
            "abandoned": abandoned,
        }
        (tmp_path / f"{node}.jsonl").write_text(json.dumps(record) + "\n")
    result = history_only(tmp_path)
    assert set(result) == {1}
    assert result[1].metrics["n_contributors"].mean == 3
    assert result[1].metrics["loss_dispersion"].mean == pytest.approx(0.008165, abs=1e-5)
    assert result[1].meta["n_abandoned"] == 1


def test_history_only_on_an_empty_directory(tmp_path):
    assert history_only(tmp_path) == {}


# --- check -------------------------------------------------------------------

CHECK_KEYS = {
    "agreement_matrix",
    "effective_replicas",
    "per_block_divergence",
    "merge_magnitude",
    "momentum_alignment",
    "merge_regression",
    "counterfactual_sweep",
}


def _check_inputs(tmp_path, nodes):
    master = _encoder_sd(0)
    _write_snapshot(
        tmp_path,
        "run-x",
        4,
        nodes(master),
        master=master,
        momentum=_encoder_sd(9, 0.1),
        gaps=("pending missing for node z",),
    )
    return {
        "diloco_snapshot_dir": tmp_path,
        "diloco_round": 4,
        "diloco_outer_lr": 0.7,
        "diloco_outer_momentum": 0.9,
    }


def _two_model_nodes(master):
    return {
        n: _model_payload(_worker(master, seed=s))
        for s, n in enumerate(("a", "b"), start=1)
    }


def test_check_attaches_controls_to_absolute_reports(tmp_path):
    inputs = _check_inputs(tmp_path, _two_model_nodes)
    reports = check(None, inputs)
    assert set(reports) == CHECK_KEYS
    for name in ("agreement_matrix", "effective_replicas"):
        meta = reports[name].meta
        assert meta["synthetic_split_m_eff"] < meta["synthetic_noise_m_eff"]
        assert meta["control_source_node"] == "a"
        assert meta["quant_noise_floor"] > 0.0
    for report in reports.values():
        assert report.meta["gaps"] == ["pending missing for node z"]
        assert report.meta["dropped_stale"] == []
    agreement = reports["agreement_matrix"].arrays["agreement"]
    assert agreement.shape == (2, 2)
    # Off-diagonal only: including the unit diagonal would give (1 + c) / 2.
    assert reports["agreement_matrix"].metrics["mean_cos"].mean == pytest.approx(
        agreement[0, 1], abs=1e-9
    )
    assert abs(agreement[0, 1]) < 0.99
    assert reports["momentum_alignment"].metrics["momentum_alignment"].deterministic
    again = check(None, inputs)
    for name in CHECK_KEYS:
        assert again[name].metrics == reports[name].metrics
        assert again[name].meta == reports[name].meta


def test_check_refuses_momentum_alignment_under_error_feedback(tmp_path):
    def nodes(master):
        w = _worker(master, seed=2)
        return {
            "a": _model_payload(_worker(master, seed=1)),
            "b": _delta_payload({k: master[k] - w[k] for k in master}),
        }

    reports = check(None, _check_inputs(tmp_path, nodes))
    assert reports["momentum_alignment"].metrics == {}
    assert (
        reports["momentum_alignment"].meta["skipped"]
        == "momentum alignment unresolvable under error feedback"
    )
    assert "m_eff" in reports["effective_replicas"].metrics


def test_verdict_meta_downgrades_only_below_a_nonzero_floor():
    assert _verdict_meta(0.01, 0.05) == {
        "verdict": "inconclusive: below quantization noise floor"
    }
    assert _verdict_meta(0.5, 0.05) == {}
    assert _verdict_meta(0.01, 0.0) == {}
    assert _verdict_meta(-0.01, 0.05) == {
        "verdict": "inconclusive: below quantization noise floor"
    }
    assert _verdict_meta(0.05, 0.05) == {}


def test_check_skips_everything_without_a_snapshot(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("check read from disk with no snapshot supplied")

    monkeypatch.setattr(diag_diloco, "load_merge_inputs", forbidden)
    monkeypatch.setattr(diag_diloco.torch, "load", forbidden)
    reports = check(None, {})
    assert set(reports) == CHECK_KEYS
    for report in reports.values():
        assert report.metrics == {}
        assert report.meta["skipped"] == "no diloco snapshot supplied"


# --- mae_loss_fn ---------------------------------------------------------------


def _tiny_model_and_windows(seed=0):
    cfg = _tiny_mae_cfg()
    # 0.9 masks every token of the tiny 2x2x2 grid, leaving one possible mask.
    cfg.mask_ratio = 0.5
    torch.manual_seed(seed)
    model = VideoMAEForPreTraining(cfg)
    gen = torch.Generator().manual_seed(seed)
    # Fresh init zeroes the prediction head, which makes the loss independent
    # of the mask; random weights make the mask observable.
    sd = {
        k: 0.1 * torch.randn(v.shape, generator=gen) if v.is_floating_point() else v
        for k, v in model.state_dict().items()
    }
    windows = torch.randn(
        2, cfg.in_channels, cfg.num_frames, cfg.image_size, cfg.image_size,
        generator=gen,
    )
    return cfg, sd, windows


def test_mae_loss_fn_returns_a_real_finite_loss():
    cfg, sd, windows = _tiny_model_and_windows()
    loss = diag_diloco.mae_loss_fn(cfg, windows, seed=0, device="cpu")(sd)
    assert np.isfinite(loss)
    assert loss > 0


def test_mae_loss_fn_same_seed_same_mask_other_seed_other_mask():
    cfg, sd, windows = _tiny_model_and_windows()
    first = diag_diloco.mae_loss_fn(cfg, windows, seed=0, device="cpu")
    second = diag_diloco.mae_loss_fn(cfg, windows, seed=0, device="cpu")
    other = diag_diloco.mae_loss_fn(cfg, windows, seed=1, device="cpu")
    assert first(sd) == second(sd)
    # Scoring twice must not advance a mask: every state dict sees one mask.
    assert first(sd) == first(sd)
    assert other(sd) != first(sd)


def test_mae_loss_fn_tolerates_only_the_empty_register_key():
    cfg, sd, windows = _tiny_model_and_windows()
    loss_fn = diag_diloco.mae_loss_fn(cfg, windows, seed=0, device="cpu")
    without_registers = {k: v for k, v in sd.items() if k != "encoder.registers.tokens"}
    assert loss_fn(without_registers) == loss_fn(sd)
    broken = {k: v for k, v in sd.items() if k != "encoder.blocks.0.attn.qkv.weight"}
    with pytest.raises(ValueError, match="encoder.blocks.0.attn.qkv.weight"):
        loss_fn(broken)


# --- merge_regression -----------------------------------------------------------


def _scalar_inputs(contributors, *, n_samples=None, val_losses=None, momentum_buf=None):
    """Master w=1.0; contributor `name` has weight `contributors[name]`, so each
    model in the regression is identifiable by its single scalar."""
    master = {"w": torch.tensor([1.0])}
    names = sorted(contributors)
    n_samples = n_samples or {n: 100 for n in names}
    return MergeInputs(
        master=master,
        master_version=3,
        worker_deltas={n: {"w": master["w"] - contributors[n]} for n in names},
        alphas=diag_diloco.data_weights(n_samples),
        momentum_buf=momentum_buf,
        val_losses=val_losses or {},
        dropped_stale=(),
        quant_noise_floor=0.0,
        n_samples=n_samples,
        n_steps={n: 10 for n in names},
        payload_kinds={n: "model_state_dict" for n in names},
        gaps=(),
    )


def _lookup_loss(by_weight, *, post, seen=None):
    """Score a state dict by its scalar: known weights map to their loss,
    anything else is the replayed post-merge master."""

    def loss_fn(sd):
        if seen is not None:
            seen.append(sd)
        w = round(float(sd["w"]), 6)
        return by_weight.get(w, post)

    return loss_fn


def test_merge_regression_sign_convention_all_positive_when_merge_hurts():
    inputs = _scalar_inputs({"a": 1.5, "b": 2.0})
    loss_fn = _lookup_loss({1.0: 0.4, 1.5: 0.5, 2.0: 0.6}, post=0.9)
    result = diag_diloco.merge_regression(
        inputs, loss_fn, outer_lr=0.7, outer_momentum=0.9
    )
    assert result["vs_alpha_mean"] == pytest.approx(0.35, abs=1e-9)
    assert result["vs_best_worker"] == pytest.approx(0.4, abs=1e-9)
    assert result["vs_prev_master"] == pytest.approx(0.5, abs=1e-9)


def test_merge_regression_baselines_disagree_on_a_merge_between_its_workers():
    inputs = _scalar_inputs({"a": 1.5, "b": 2.0})
    loss_fn = _lookup_loss({1.0: 0.4, 1.5: 0.5, 2.0: 0.6}, post=0.52)
    result = diag_diloco.merge_regression(
        inputs, loss_fn, outer_lr=0.7, outer_momentum=0.9
    )
    assert result["vs_alpha_mean"] < 0 < result["vs_best_worker"]


def test_merge_regression_ignores_logged_val_losses():
    inputs = _scalar_inputs({"a": 1.5, "b": 2.0}, val_losses={"a": 0.01, "b": 0.01})
    loss_fn = _lookup_loss({1.0: 0.4, 1.5: 0.5, 2.0: 0.6}, post=0.9)
    result = diag_diloco.merge_regression(
        inputs, loss_fn, outer_lr=0.7, outer_momentum=0.9
    )
    assert result["vs_alpha_mean"] == pytest.approx(0.9 - 0.55, abs=1e-9)


def test_merge_regression_replays_the_production_outer_step():
    momentum = {"w": torch.tensor([0.3])}
    inputs = _scalar_inputs({"a": 1.5, "b": 2.0}, momentum_buf=momentum)
    seen = []
    diag_diloco.merge_regression(
        inputs,
        _lookup_loss({1.0: 0.4, 1.5: 0.5, 2.0: 0.6}, post=0.9, seen=seen),
        outer_lr=0.7,
        outer_momentum=0.9,
    )
    expected, _ = diag_diloco.nesterov_outer_step(
        inputs.master,
        average_deltas(inputs.worker_deltas, inputs.alphas),
        momentum,
        lr=0.7,
        momentum=0.9,
    )
    replayed = [sd for sd in seen if round(float(sd["w"]), 6) not in (1.0, 1.5, 2.0)]
    assert len(replayed) == 1
    assert torch.equal(replayed[0]["w"], expected["w"])


# --- counterfactual_sweep ---------------------------------------------------------

_GRID = [(0.1, 0.0), (0.7, 0.9), (1.0, 0.0), (1.0, 0.9)]


def _weight_loss(sd):
    # The loss IS the scalar weight, so every row's post_loss exposes the
    # replayed master exactly.
    return float(sd["w"])


def _sweep(inputs=None, loss_fn=_weight_loss, **overrides):
    kwargs = {
        "grid": _GRID,
        "rules": ["nesterov", "average"],
        "weightings": ["data"],
        "configured": (0.7, 0.9),
    }
    kwargs.update(overrides)
    inputs = inputs or _scalar_inputs({"a": 1.5, "b": 2.0})
    return diag_diloco.counterfactual_sweep(inputs, loss_fn, **kwargs)


def test_sweep_grid_must_reach_below_the_configured_lr():
    with pytest.raises(ValueError, match="below"):
        _sweep(grid=[(0.7, 0.9), (1.0, 0.9)])


def test_sweep_requires_the_configured_cell():
    with pytest.raises(ValueError, match="configured"):
        _sweep(grid=[(0.1, 0.0), (0.3, 0.9)])


def test_sweep_rejects_an_unknown_rule():
    with pytest.raises(ValueError, match="iso_c"):
        _sweep(rules=["iso_c"])


def test_sweep_average_equals_nesterov_at_lr1_momentum0():
    inputs = _scalar_inputs({"a": 1.5, "b": 2.0}, momentum_buf={"w": torch.tensor([0.3])})
    rows = _sweep(inputs)
    average = next(r for r in rows if r["rule"] == "average")
    plain = next(
        r for r in rows
        if r["rule"] == "nesterov" and (r["outer_lr"], r["outer_momentum"]) == (1.0, 0.0)
    )
    assert (average["outer_lr"], average["outer_momentum"]) == (1.0, 0.0)
    assert average["post_loss"] == plain["post_loss"] == pytest.approx(1.75)
    # The momentum buffer is live elsewhere in the grid, so equality above is
    # not the trivial "no momentum anywhere" case.
    configured = next(
        r for r in rows
        if r["rule"] == "nesterov" and (r["outer_lr"], r["outer_momentum"]) == (0.7, 0.9)
    )
    assert configured["post_loss"] != plain["post_loss"]


def test_sweep_emits_average_once_per_weighting():
    rows = _sweep()
    assert len(rows) == 5
    for row in rows:
        assert {
            "rule", "weighting", "outer_lr", "outer_momentum", "post_loss",
            "vs_alpha_mean", "vs_best_worker", "vs_prev_master",
        } <= set(row)


def test_sweep_uses_real_per_node_sample_counts():
    inputs = _scalar_inputs({"a": 1.5, "b": 2.0}, n_samples={"a": 900, "b": 100})
    for row in _sweep(inputs):
        assert row["vs_alpha_mean"] == pytest.approx(
            row["post_loss"] - (0.9 * 1.5 + 0.1 * 2.0), abs=1e-6
        )


def test_sweep_omits_a_weighting_production_would_refuse():
    # _inverse_loss_weights raises on a non-positive val_loss; the sweep must
    # drop that weighting, not take the whole check down.
    inputs = _scalar_inputs({"a": 1.5, "b": 2.0}, val_losses={"a": 0.5, "b": 0.0})
    rows = _sweep(inputs, weightings=["data", "inverse_loss"])
    assert {r["weighting"] for r in rows} == {"data"}
    alphas, reason = diag_diloco._sweep_alphas(inputs, "inverse_loss")
    assert alphas is None
    assert "val_loss" in reason


def test_sweep_omits_a_weighting_with_incomplete_inputs():
    inputs = _scalar_inputs({"a": 1.5, "b": 2.0})
    inputs = diag_diloco.MergeInputs(**{**inputs.__dict__, "n_steps": {"a": 10}})
    rows = _sweep(inputs, weightings=["data", "token"])
    assert {r["weighting"] for r in rows} == {"data"}


def test_sweep_scores_the_baselines_once():
    calls = []

    def counting(sd):
        calls.append(sd)
        return _weight_loss(sd)

    # Two weightings, so scoring the baselines per weighting would show up.
    _sweep(loss_fn=counting, weightings=["data", "token"])
    assert len(calls) == 2 + 1 + 2 * 5


# --- held-out guard and forward-pass checks --------------------------------------


def _assignments(tmp_path, mapping):
    path = tmp_path / "shard_assignments_run-x.json"
    path.write_text(json.dumps(mapping))
    return path


def test_held_out_rejects_an_overlapping_eval_shard(tmp_path):
    path = _assignments(tmp_path, {"a": ["shard-001.tar"], "b": ["shard-002.tar"]})
    with pytest.raises(ValueError, match="'a'"):
        diag_diloco.assert_held_out(tmp_path / "val" / "shard-001.tar", path, ["a", "b"])


def test_held_out_rejects_a_node_missing_from_assignments(tmp_path):
    path = _assignments(tmp_path, {"a": ["shard-001.tar"]})
    with pytest.raises(ValueError, match="'c'"):
        diag_diloco.assert_held_out(tmp_path / "val" / "shard-val.tar", path, ["a", "c"])


def test_held_out_rejects_a_shard_outside_val(tmp_path):
    path = _assignments(tmp_path, {"a": ["shard-001.tar"]})
    with pytest.raises(ValueError, match="val"):
        diag_diloco.assert_held_out(tmp_path / "train" / "shard-900.tar", path, ["a"])


def test_held_out_accepts_a_disjoint_val_shard(tmp_path):
    path = _assignments(tmp_path, {"a": ["shard-001.tar"], "b": ["shard-002.tar"]})
    shard = tmp_path / "val" / "shard-val.tar"
    assert diag_diloco.assert_held_out(shard, path, ["a", "b"]) is None


def test_check_without_eval_shard_runs_12a_and_skips_12b(tmp_path):
    reports = check(None, _check_inputs(tmp_path, _two_model_nodes))
    assert set(reports) == CHECK_KEYS
    assert "mean_cos" in reports["agreement_matrix"].metrics
    for name in ("merge_regression", "counterfactual_sweep"):
        assert reports[name].metrics == {}
        assert reports[name].meta["skipped"] == "no diloco eval shard supplied"


def _forward_pass_inputs(tmp_path, *, overlap=False):
    from tests.evaluation.test_diag_common import make_fake_shard

    cfg, master, _ = _tiny_model_and_windows()
    nodes = {
        n: _model_payload(
            {k: v + 0.05 * torch.randn(v.shape, generator=torch.Generator().manual_seed(s))
             if v.is_floating_point() else v for k, v in master.items()},
            n_samples=100 * s,
        )
        for s, n in enumerate(("a", "b"), start=1)
    }
    snap = tmp_path / "snap"
    snap.mkdir()
    _write_snapshot(snap, "run-x", 4, nodes, master=master)
    val_dir = tmp_path / "val"
    val_dir.mkdir()
    shard = make_fake_shard(val_dir, {"c0": 8, "c1": 8})
    assigned = shard.name if overlap else "shard-train-000.tar"
    return {
        "diloco_snapshot_dir": snap,
        "diloco_round": 4,
        "diloco_outer_lr": 0.7,
        "diloco_outer_momentum": 0.9,
        "diloco_eval_shard": shard,
        "diloco_assignments": _assignments(tmp_path, {"a": [assigned], "b": ["x.tar"]}),
        "diloco_n_replicates": 3,
        "model_cfg": cfg,
        "n_windows": 2,
        "seed": 0,
        "device": "cpu",
    }


def test_check_forward_pass_reports_on_a_tiny_model(tmp_path):
    inputs = _forward_pass_inputs(tmp_path)
    reports = check(None, inputs)

    regression = reports["merge_regression"]
    assert set(regression.metrics) == {"vs_alpha_mean", "vs_best_worker", "vs_prev_master"}
    for value in regression.metrics.values():
        assert value.n == 3
        assert value.deterministic is False
    assert set(regression.meta["verdicts"]) == set(regression.metrics)
    assert set(regression.meta["verdicts"].values()) <= {"HURT", "HELPED", "INCONCLUSIVE"}
    assert "masking seeds" in regression.meta["sd_source"]

    # The emitted numbers are merge_regression's, on the same windows and masks.
    merge = diag_diloco.load_merge_inputs(inputs["diloco_snapshot_dir"], 4)
    windows = diag_diloco.paired_inputs(
        inputs["diloco_eval_shard"], n_windows=2, num_frames=4, image_size=32, seed=0
    )[0]
    direct = [
        diag_diloco.merge_regression(
            merge,
            diag_diloco.mae_loss_fn(inputs["model_cfg"], windows, seed=s, device="cpu"),
            outer_lr=0.7,
            outer_momentum=0.9,
        )
        for s in range(3)
    ]
    for key in ("vs_alpha_mean", "vs_best_worker", "vs_prev_master"):
        assert regression.metrics[key].mean == pytest.approx(
            float(np.mean([d[key] for d in direct])), abs=1e-9
        )

    sweep = reports["counterfactual_sweep"]
    assert set(sweep.metrics) == {
        "configured_post_loss", "best_post_loss", "best_minus_configured",
    }
    assert sweep.metrics["best_minus_configured"].mean <= 0.0
    configured = [
        c for c in sweep.meta["grid"]
        if (c["rule"], c["weighting"], c["outer_lr"], c["outer_momentum"])
        == ("nesterov", "data", 0.7, 0.9)
    ]
    assert len(configured) == 1
    assert min(c["outer_lr"] for c in sweep.meta["grid"]) < 0.7
    assert sweep.meta["verdict"] in {"CONFIG_CHANGE", "INCONCLUSIVE"}
    assert sweep.meta["weightings_unavailable"] == {}
    assert set(sweep.meta["best_cell"]) == {"rule", "weighting", "outer_lr", "outer_momentum"}


def test_check_raises_on_an_overlapping_eval_shard(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("forward pass ran before the held-out guard")

    monkeypatch.setattr(diag_diloco, "mae_loss_fn", forbidden)
    with pytest.raises(ValueError, match="'a'"):
        check(None, _forward_pass_inputs(tmp_path, overlap=True))


def test_merge_inputs_drop_and_record_nonfinite_workers(tmp_path):
    master = _master()

    def nodes(master):
        bad = _worker(master, seed=3)
        bad["w"][0, 0] = float("nan")
        return {
            "a": _model_payload(_worker(master, seed=1)),
            "b": _model_payload(_worker(master, seed=2)),
            "c": _model_payload(bad),
        }

    _write_snapshot(tmp_path, "run-x", 4, nodes(master), master=master)
    merge = diag_diloco.load_merge_inputs(tmp_path, 4)
    assert sorted(merge.worker_deltas) == ["a", "b"]
    assert merge.dropped_nonfinite == ("c",)
    assert set(merge.alphas) == {"a", "b"}


def test_check_replays_the_run_configured_weighting(tmp_path):
    import dataclasses

    inputs = _forward_pass_inputs(tmp_path)
    inputs["diloco_weighting"] = "token"
    reports = check(None, inputs)
    sweep = reports["counterfactual_sweep"]
    assert sweep.meta["configured_cell"] == {
        "rule": "nesterov", "weighting": "token", "outer_lr": 0.7, "outer_momentum": 0.9,
    }
    merge = diag_diloco.load_merge_inputs(inputs["diloco_snapshot_dir"], 4)
    token_alphas, _ = diag_diloco._sweep_alphas(merge, "token")
    assert token_alphas != merge.alphas  # else this test could not tell them apart
    windows = diag_diloco.paired_inputs(
        inputs["diloco_eval_shard"], n_windows=2, num_frames=4, image_size=32, seed=0
    )[0]
    direct = [
        diag_diloco.merge_regression(
            dataclasses.replace(merge, alphas=token_alphas),
            diag_diloco.mae_loss_fn(inputs["model_cfg"], windows, seed=s, device="cpu"),
            outer_lr=0.7,
            outer_momentum=0.9,
        )["vs_alpha_mean"]
        for s in range(3)
    ]
    assert reports["merge_regression"].metrics["vs_alpha_mean"].mean == pytest.approx(
        float(np.mean(direct)), abs=1e-9
    )


@pytest.mark.parametrize(
    "trigger, expected_reason",
    [
        pytest.param("forward_pass_oom", "RuntimeError: CUDA out of memory", id="forward_pass_oom"),
        pytest.param("unreplayable_weighting", "no_such_weighting", id="unreplayable_weighting"),
    ],
)
def test_12b_is_skipped_with_its_reason_and_12a_survives(
    tmp_path, monkeypatch, trigger, expected_reason
):
    inputs = _forward_pass_inputs(tmp_path)
    if trigger == "forward_pass_oom":

        def oom(*args, **kwargs):
            raise RuntimeError("CUDA out of memory")

        monkeypatch.setattr(diag_diloco, "mae_loss_fn", oom)
    else:
        inputs["diloco_weighting"] = "no_such_weighting"
    reports = check(None, inputs)
    assert "mean_cos" in reports["agreement_matrix"].metrics
    for name in ("merge_regression", "counterfactual_sweep"):
        assert reports[name].metrics == {}
        assert expected_reason in reports[name].meta["skipped"]


def test_check_reports_a_custom_skip_reason_without_a_snapshot():
    reports = check(None, {"diloco_skip_reason": "no outer hyperparameters"})
    assert {r.meta["skipped"] for r in reports.values()} == {"no outer hyperparameters"}


def test_fewer_than_three_replicates_is_refused_before_any_forward_pass(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("forward pass ran with too few replicates")

    monkeypatch.setattr(diag_diloco, "mae_loss_fn", forbidden)
    inputs = _forward_pass_inputs(tmp_path)
    inputs["diloco_n_replicates"] = 1
    reports = check(None, inputs)
    assert "diloco_n_replicates must be >= 3" in reports["merge_regression"].meta["skipped"]
