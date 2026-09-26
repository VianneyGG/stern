"""Tests for the shared SSL-diagnostic harness (`scripts/_diag_common.py`).

Every later diagnostic plan imports this module, so the invariants pinned here
are what make "the metric moved" a defensible claim rather than a reading of
one seed: a metric is a distribution or it is explicitly deterministic, and a
difference counts only when it clears a replication band that actually exists.
"""

from __future__ import annotations

import io
import json
import math
import pickle
import tarfile

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from video_encoding.mesh.pending_files import pending_filename
from video_encoding.models.videomaev2.encoder import VideoMAEEncoder
from video_encoding.models.videomaev2.mae import VideoMAEForPreTraining
from video_encoding.scripts._diag_common import (
    LADDER,
    LADDER_REVISION,
    DiagReport,
    MetricValue,
    Rung,
    RungUnavailable,
    Tier,
    _public_videomae_sinusoid_pos_embed,
    forward_with_hidden_states,
    ladder_tau,
    load_bundle,
    load_ladder_rung,
    load_solo_bundle,
    paired_inputs,
    random_init_bundle,
    remap_public_videomae,
    replicate,
    run_hyperparams,
)


def sampled(mean: float, sd: float, n: int = 3) -> MetricValue:
    return MetricValue(
        mean=mean, sd=sd, n=n, seeds=tuple(range(n)), deterministic=False
    )


def test_metric_value_separated_from_pins_both_sides_of_band(tmp_path):
    """Separation is decided by the pooled band, not by the raw difference."""
    band = 2.0 * math.sqrt(0.1**2 + 0.1**2)
    assert band == pytest.approx(0.28284271247461906)
    a = sampled(1.0, 0.1)
    just_inside = sampled(1.0 + band * 0.9, 0.1)
    just_outside = sampled(1.0 + band * 1.1, 0.1)

    # Pin the actual gap on each side of the band, not just the verdict --
    # a wrong band constant inside separated_from would still pass a
    # bool-only check as long as both fixtures moved with it.
    assert abs(a.mean - just_inside.mean) == pytest.approx(0.25455844122715715)
    assert abs(a.mean - just_outside.mean) == pytest.approx(0.31112698372208096)

    assert a.separated_from(just_inside) is False
    assert a.separated_from(just_outside) is True
    # Symmetric: which operand is the receiver must not change the verdict.
    assert just_outside.separated_from(a) is True
    assert just_inside.separated_from(a) is False


def test_separated_from_raises_when_pooled_band_is_zero():
    """Zero observed variance is a bug report, not universal significance.

    A sampled function handed three distinct seeds that ignores them produces
    sd=0, which would make every non-zero gap "separated" and hand Tier A to an
    unreplicated metric. Surface it as an error naming the metric instead.
    """
    a = sampled(1.0, 0.0)
    b = sampled(2.0, 0.0)
    with pytest.raises(ValueError, match="variance"):
        a.separated_from(b)


def test_separated_from_raises_on_a_deterministic_operand():
    det = MetricValue.deterministic_value(1.0)
    with pytest.raises(ValueError, match="deterministic"):
        det.separated_from(sampled(2.0, 0.1))
    with pytest.raises(ValueError, match="deterministic"):
        sampled(2.0, 0.1).separated_from(det)


def test_sampled_metric_value_without_sd_raises():
    """A sampled value with no error bar must fail at construction.

    Deferring it to separated_from surfaces the problem far from its cause.
    """
    with pytest.raises(ValueError, match="sd"):
        MetricValue(mean=1.0, sd=None, n=3, seeds=(0, 1, 2), deterministic=False)


def test_sampled_metric_value_below_three_samples_raises():
    with pytest.raises(ValueError, match="n"):
        MetricValue(mean=1.0, sd=0.1, n=2, seeds=(0, 1), deterministic=False)


def test_deterministic_metric_value_carrying_sampling_fields_raises():
    with pytest.raises(
        ValueError, match="deterministic MetricValue carries no sampling"
    ):
        MetricValue(mean=1.0, sd=1.0, n=None, seeds=(), deterministic=True)
    with pytest.raises(
        ValueError, match="deterministic MetricValue carries no sampling"
    ):
        MetricValue(mean=1.0, sd=None, n=3, seeds=(), deterministic=True)


def test_deterministic_value_constructs_the_only_valid_shape():
    det = MetricValue.deterministic_value(2.5)
    assert det.deterministic is True
    assert det.mean == 2.5
    assert det.sd is None
    assert det.n is None
    assert det.seeds == ()


def test_replicate_returns_three_distinct_seeds_and_nonzero_sd():
    value = replicate(lambda seed: float(seed % 97), n=3, seed=7)
    assert value.n == 3
    assert len(set(value.seeds)) == 3
    assert value.sd is not None and value.sd > 0.0
    assert not value.deterministic


def test_replicate_seeds_are_deterministic_given_the_base_seed():
    first = replicate(lambda seed: float(seed % 97), n=3, seed=7)
    second = replicate(lambda seed: float(seed % 97), n=3, seed=7)
    assert first.seeds == second.seeds
    assert first.mean == second.mean

    other = replicate(lambda seed: float(seed % 97), n=3, seed=8)
    assert other.seeds != first.seeds


def test_replicate_of_a_seed_ignoring_function_is_caught_downstream():
    """replicate cannot reject a seed-ignoring function -- separated_from does.

    Distinct seeds are necessary but not sufficient: a function that receives
    three seeds and ignores them still yields sd=0, which is why the zero-band
    check lives in separated_from rather than only in replicate.
    """
    value = replicate(lambda seed: 1.0, n=3, seed=7)
    assert len(set(value.seeds)) == 3
    assert value.sd == 0.0
    with pytest.raises(ValueError, match="variance"):
        value.separated_from(replicate(lambda seed: 2.0, n=3, seed=9))


# --- fixtures for the loading / sampling half of the harness ---------------


def _tiny_model_cfg(**overrides):
    """The smallest config the encoder and MAE head both accept."""
    cfg = {
        "num_frames": 4,
        "image_size": 32,
        "in_channels": 3,
        "tubelet_size": 2,
        "patch_size": 16,
        "embed_dim": 16,
        "depth": 2,
        "num_heads": 2,
        "mlp_ratio": 2.0,
        "dropout": 0.0,
        "drop_path_rate": 0.0,
        "decoder_depth": 1,
        "decoder_embed_dim": 8,
        "decoder_num_heads": 2,
        "mask_ratio": 0.5,
        "decoder_mask_ratio": 0.0,
    }
    cfg.update(overrides)
    return OmegaConf.create(cfg)


def _jpeg_bytes(color: tuple[int, int, int], size: int = 32) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), color=color).save(buf, format="JPEG")
    return buf.getvalue()


def make_fake_shard(tmp_path, clip_frame_counts: dict[str, int]):
    """A WebDataset shard in the real `<clip>.pickle` + manifest.json form.

    Mirrors test_export_vit_internals.py's private `_write_shard` rather than
    importing it -- a shared private test helper would couple the two modules
    for fifteen lines.
    """
    shard_path = tmp_path / "shard-000000.tar"
    manifest = {}
    with tarfile.open(shard_path, "w") as tar:
        for clip_key, n_frames in clip_frame_counts.items():
            frames = [_jpeg_bytes((i * 7 % 256, 0, 0)) for i in range(n_frames)]
            payload = pickle.dumps(frames)
            info = tarfile.TarInfo(name=f"{clip_key}.pickle")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
            manifest[clip_key] = {
                "shard": shard_path.name,
                "num_frames": n_frames,
                "size": len(payload),
            }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return shard_path


def write_checkpoint(mesh_shared, run_id: str, model_cfg, *, filename="master.pt"):
    """Save a real VideoMAEForPreTraining state dict, as the mesh writes it."""
    torch.manual_seed(0)
    model = VideoMAEForPreTraining(model_cfg)
    path = mesh_shared / "checkpoints" / run_id / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, path)
    return model


# --- load_bundle -----------------------------------------------------------


def test_load_bundle_full_model_false_matches_fixture_encoder_keys(tmp_path):
    cfg = _tiny_model_cfg()
    saved = write_checkpoint(tmp_path, "lev-run", cfg)

    bundle = load_bundle("lev-run", tmp_path, cfg, full_model=False)

    assert bundle.run_id == "lev-run"
    assert bundle.model is None
    expected = saved.encoder.state_dict()
    got = bundle.encoder.state_dict()
    assert set(got) == set(expected)
    for key, tensor in expected.items():
        assert torch.allclose(got[key], tensor), key


def test_load_bundle_full_model_true_encoder_matches_false_path(tmp_path):
    """Both paths must produce the same encoder, or every tier means two things."""
    cfg = _tiny_model_cfg()
    write_checkpoint(tmp_path, "lev-run", cfg)

    lean = load_bundle("lev-run", tmp_path, cfg, full_model=False)
    full = load_bundle("lev-run", tmp_path, cfg, full_model=True)

    assert full.model is not None
    full_sd = full.model.encoder.state_dict()
    for key, tensor in lean.encoder.state_dict().items():
        assert torch.allclose(full_sd[key], tensor), key


def test_load_bundle_allows_missing_zero_element_registers(tmp_path):
    """Pre-RegisterTokens checkpoints lack a parameter that holds no values."""
    cfg = _tiny_model_cfg(num_registers=0)
    write_checkpoint(tmp_path, "lev-old", cfg)
    path = tmp_path / "checkpoints" / "lev-old" / "master.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    del payload["model_state_dict"]["encoder.registers.tokens"]
    torch.save(payload, path)

    bundle = load_bundle("lev-old", tmp_path, cfg, full_model=False)
    assert bundle.encoder.registers.tokens.numel() == 0


def test_load_bundle_raises_on_differing_register_count(tmp_path):
    """A different register count changes the sequence every block attends over."""
    write_checkpoint(tmp_path, "lev-reg", _tiny_model_cfg(num_registers=4))

    with pytest.raises(ValueError, match="mismatch"):
        load_bundle("lev-reg", tmp_path, _tiny_model_cfg(num_registers=2))


def test_load_bundle_raises_on_patch_embed_channel_mismatch_both_paths(tmp_path):
    """Never inherit load_encoder_state's zero-inflation of the first conv.

    That inflation is right for fine-tuning -- a zero channel is a no-op that
    trains up -- and wrong here: it rewrites the first convolution and reports a
    clean load, so every delta_from_init and block-0 branch-norm reading would
    describe a model the checkpoint never contained.
    """
    write_checkpoint(tmp_path, "lev-ch", _tiny_model_cfg(in_channels=3))
    target = _tiny_model_cfg(in_channels=6)

    for full_model in (False, True):
        with pytest.raises(ValueError, match="channel"):
            load_bundle("lev-ch", tmp_path, target, full_model=full_model)


def test_load_bundle_does_not_reserve_random_init_as_a_run_id(tmp_path):
    """A run genuinely named "random_init" loads its checkpoint, not a stub."""
    cfg = _tiny_model_cfg()
    saved = write_checkpoint(tmp_path, "random_init", cfg)

    bundle = load_bundle("random_init", tmp_path, cfg, full_model=False)

    first = "patch_embed.proj.weight"
    loaded_diff = (
        (bundle.encoder.state_dict()[first] - saved.encoder.state_dict()[first])
        .abs()
        .max()
        .item()
    )
    # Loaded from the checkpoint: bit-identical to what was saved, not just close.
    assert loaded_diff == pytest.approx(0.0, abs=0.0)

    # And the untrained control is reachable by its own entry point -- a
    # freshly initialized model, not the checkpoint under a stub name.
    control = random_init_bundle(cfg)
    control_diff = (
        (control.encoder.state_dict()[first] - saved.encoder.state_dict()[first])
        .abs()
        .max()
        .item()
    )
    assert control_diff > 1e-3


# --- load_solo_bundle (rung 2's merge=false route) --------------------------


def write_pending_checkpoint(mesh_shared, run_id, model_cfg, *, round_num, node):
    """Save what a `mesh.merge=false` worker actually pushes.

    A solo run never produces `checkpoints/<run_id>/master.pt` -- that file is
    written by the merge path, which `merge=false` skips by definition. What it
    does push is `checkpoints/pending/{run_id}_{round}_{node}.pt`
    (mesh_round.push_checkpoint_and_wait), which is what rung 2 must load.
    """
    torch.manual_seed(0)
    model = VideoMAEForPreTraining(model_cfg)
    path = (
        mesh_shared
        / "checkpoints"
        / "pending"
        / pending_filename(run_id, round_num, node)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, path)
    return model


def test_load_solo_bundle_reads_the_pending_checkpoint_a_merge_false_run_writes(
    tmp_path,
):
    cfg = _tiny_model_cfg()
    saved = write_pending_checkpoint(
        tmp_path, "ab-single_node-overfit50", cfg, round_num=3, node="node055"
    )

    bundle = load_solo_bundle(
        "ab-single_node-overfit50",
        tmp_path,
        cfg,
        node="node055",
        round=3,
        full_model=False,
    )

    assert bundle.run_id == "ab-single_node-overfit50"
    assert bundle.round == 3
    first = "patch_embed.proj.weight"
    diff = (
        (bundle.encoder.state_dict()[first] - saved.encoder.state_dict()[first])
        .abs()
        .max()
        .item()
    )
    assert diff == pytest.approx(0.0, abs=0.0)


def test_load_solo_bundle_does_not_fall_back_to_master(tmp_path):
    """A missing pending file must raise, never silently read master.pt.

    Rung 2 is defined as one specific solo run's weights. Falling back to a
    master.pt that some *other* (merged) run left in the tree would put a
    different model on the rung and report it as the overfit control -- the
    silent-fallback failure the ladder exists to rule out.
    """
    cfg = _tiny_model_cfg()
    # A master.pt exists for the same run id, but no pending file does.
    write_checkpoint(tmp_path, "solo-run", cfg)

    with pytest.raises(FileNotFoundError, match="no solo checkpoint"):
        load_solo_bundle("solo-run", tmp_path, cfg, node="node055", round=3)


def test_load_bundle_and_load_solo_bundle_agree_on_identical_weights(tmp_path):
    """The two entry points differ only in which path they read.

    Same weights saved under both layouts must produce the same encoder, so a
    metric cannot depend on which route a rung was loaded through.
    """
    cfg = _tiny_model_cfg()
    write_checkpoint(tmp_path, "same", cfg)
    write_pending_checkpoint(tmp_path, "same", cfg, round_num=1, node="node055")

    merged = load_bundle("same", tmp_path, cfg)
    solo = load_solo_bundle("same", tmp_path, cfg, node="node055", round=1)

    merged_sd = merged.encoder.state_dict()
    solo_sd = solo.encoder.state_dict()
    assert set(solo_sd) == set(merged_sd)
    compared = 0
    for key, tensor in merged_sd.items():
        # registers.tokens is zero-element when num_registers=0; .max() on an
        # empty tensor raises, and shape equality is the whole claim there.
        assert solo_sd[key].shape == tensor.shape, key
        if tensor.numel() == 0:
            continue
        gap = (solo_sd[key] - tensor).abs().max().item()
        assert gap == pytest.approx(0.0, abs=0.0), key
        compared += 1
    # Guard against the loop vacuously passing if the state dict ever empties.
    assert compared >= 20


# --- paired_inputs ---------------------------------------------------------


def test_paired_inputs_deterministic_and_matching_shape(tmp_path):
    shard = make_fake_shard(tmp_path, {f"clip{i}": 8 for i in range(6)})

    a_real, a_rand = paired_inputs(
        shard, n_windows=2, num_frames=4, image_size=32, seed=3
    )
    b_real, b_rand = paired_inputs(
        shard, n_windows=2, num_frames=4, image_size=32, seed=3
    )

    assert tuple(a_real.shape) == (2, 3, 4, 32, 32)
    assert tuple(a_rand.shape) == (2, 3, 4, 32, 32)

    # Same seed, same bytes -- pin the gap at exactly zero rather than only
    # asserting a bool, so a near-miss (e.g. a reseeded RNG that happens to
    # land close) is reported as the numeric value it actually is.
    assert (a_real - b_real).abs().max().item() == pytest.approx(0.0, abs=0.0)
    assert (a_rand - b_rand).abs().max().item() == pytest.approx(0.0, abs=0.0)


def test_paired_inputs_real_and_random_are_not_close(tmp_path):
    shard = make_fake_shard(tmp_path, {f"clip{i}": 8 for i in range(6)})
    real, rand = paired_inputs(shard, n_windows=2, num_frames=4, image_size=32, seed=3)

    # The random half is a standard normal by construction -- pin that, so a
    # bug that returned the real batch twice (which would still be "not
    # allclose" to nothing) cannot pass.
    assert rand.mean().item() == pytest.approx(0.0, abs=0.05)
    assert rand.std().item() == pytest.approx(1.0, abs=0.05)

    # The real half is normalized frames: a far narrower spread than N(0,1),
    # so the two are separated by much more than any tolerance.
    assert (real - rand).abs().mean().item() > 1.0
    assert not torch.allclose(real, rand, atol=1e-3)


# --- run_hyperparams -------------------------------------------------------


def test_run_hyperparams_reads_runs_state_path(tmp_path):
    configs = tmp_path / "mesh_run_configs.json"
    configs.write_text(
        json.dumps(
            [
                {"run_id": "lev-a", "outer_lr": 0.7, "rounds": 10},
                {"run_id": "lev-b", "outer_lr": 0.45, "rounds": 30},
            ]
        )
    )
    assert run_hyperparams("lev-b", configs) == {
        "run_id": "lev-b",
        "outer_lr": 0.45,
        "rounds": 30,
    }


def test_run_hyperparams_unknown_run_returns_empty_dict_not_raise(tmp_path):
    """Hyperparameters annotate a report; their absence must not kill the run."""
    configs = tmp_path / "mesh_run_configs.json"
    configs.write_text(json.dumps([{"run_id": "lev-a", "outer_lr": 0.7}]))
    assert run_hyperparams("nonexistent-run", configs) == {}

    assert run_hyperparams("lev-a", tmp_path / "absent.json") == {}
    unparseable = tmp_path / "broken.json"
    unparseable.write_text("{not json")
    assert run_hyperparams("lev-a", unparseable) == {}


# --- DiagReport ------------------------------------------------------------


def test_diag_report_roundtrips_metrics_and_arrays(tmp_path):
    report = DiagReport(
        metrics={
            "sampled": sampled(1.5, 0.2),
            "det": MetricValue.deterministic_value(9.0),
        },
        arrays={"per_block": np.arange(6, dtype=np.float32).reshape(2, 3)},
        meta={"LADDER_REVISION": LADDER_REVISION, "run_id": "lev-a"},
    )
    json_path, npz_path = tmp_path / "diagnostics.json", tmp_path / "diagnostics.npz"
    report.to_json(json_path)
    report.write_npz(npz_path)

    back = DiagReport.from_json(json_path)
    for name, original in report.metrics.items():
        assert back.metrics[name] == original

    # Pin what actually came back, not just "it equals what went in" -- a
    # to_json/from_json pair that dropped the same field on both sides would
    # satisfy the equality above while losing the value.
    assert back.metrics["sampled"].mean == 1.5
    assert back.metrics["sampled"].sd == 0.2
    assert back.metrics["sampled"].n == 3
    assert back.metrics["sampled"].deterministic is False
    assert back.metrics["det"].mean == 9.0
    assert back.metrics["det"].deterministic is True
    assert back.metrics["det"].sd is None
    assert back.meta["LADDER_REVISION"] == LADDER_REVISION

    recovered = np.load(npz_path)["per_block"]
    assert recovered.tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
    assert np.array_equal(recovered, report.arrays["per_block"])


def test_diag_report_json_marks_deterministic_metrics(tmp_path):
    """Without the flag, {"sd": null, "n": null} reads as a malformed sample."""
    report = DiagReport(
        metrics={"det": MetricValue.deterministic_value(9.0)},
        arrays={},
        meta={"LADDER_REVISION": LADDER_REVISION},
    )
    path = tmp_path / "diagnostics.json"
    report.to_json(path)
    assert json.loads(path.read_text())["metrics"]["det"]["deterministic"] is True


def test_diag_report_from_json_rejects_n_below_three(tmp_path):
    """The n>=3-or-deterministic invariant holds on read, not just on write."""
    path = tmp_path / "diagnostics.json"
    path.write_text(
        json.dumps(
            {
                "meta": {"LADDER_REVISION": LADDER_REVISION},
                "metrics": {
                    "stale": {
                        "mean": 1.0,
                        "sd": 0.1,
                        "n": 1,
                        "seeds": [0],
                        "deterministic": False,
                    }
                },
            }
        )
    )
    with pytest.raises(ValueError, match="n"):
        DiagReport.from_json(path)


def test_diag_report_without_ladder_revision_raises():
    """A tier that cannot be attributed to a ladder version is unreadable later."""
    with pytest.raises(ValueError, match="LADDER_REVISION"):
        DiagReport(metrics={}, arrays={}, meta={"run_id": "lev-a"})


# --- forward_with_hidden_states -------------------------------------------


def test_forward_with_hidden_states_matches_encoder_forward():
    """The one mirrored traversal is pinned against the real forward, exactly.

    export_vit_internals.py's hand-mirrored copy of ViTBlock.forward drifted
    from the model it claimed to describe; a bit-for-bit parity test is what
    keeps this one honest.
    """
    cfg = _tiny_model_cfg(num_registers=2)
    torch.manual_seed(0)
    encoder = VideoMAEEncoder(cfg).eval()
    x = torch.randn(2, 3, 4, 32, 32)

    hidden = forward_with_hidden_states(encoder, x)

    assert len(hidden) == 3  # cfg.depth (2) + 1 -- pre-block input plus one per block
    reconstructed = encoder.norm(encoder.registers.strip(hidden[-1])).mean(dim=1)
    with torch.no_grad():
        assert torch.equal(reconstructed, encoder(x))


def test_forward_with_hidden_states_returns_unnormed_stream():
    """Raw pre-norm residuals, registers attached -- what branch ratios need.

    forward_tokens(return_layers=...) norms and register-strips every
    intermediate, which destroys exactly the magnitudes being measured.
    """
    cfg = _tiny_model_cfg(num_registers=2)
    torch.manual_seed(0)
    encoder = VideoMAEEncoder(cfg).eval()
    x = torch.randn(1, 3, 4, 32, 32)

    hidden = forward_with_hidden_states(encoder, x)

    n_patches = encoder.pos_embed.shape[1]
    for state in hidden:
        assert state.shape[1] == n_patches + encoder.registers.num_registers
    last = hidden[-1]
    assert not torch.allclose(last, encoder.norm(last), atol=1e-6)


def test_forward_with_hidden_states_returns_detached_tensors():
    """depth+1 activations must not drag the autograd graph along with them."""
    cfg = _tiny_model_cfg()
    encoder = VideoMAEEncoder(cfg).eval()
    assert any(p.requires_grad for p in encoder.parameters())

    hidden = forward_with_hidden_states(encoder, torch.randn(1, 3, 4, 32, 32))

    for state in hidden:
        assert not state.requires_grad
        state.numpy()
    # no_grad, never inference_mode: Plan 3 trains attentive probes on these.
    assert (hidden[-1] * torch.ones(1, requires_grad=True)).requires_grad


# --- the measurement ladder ------------------------------------------------


def _rung_values(*means, sd=0.01, rungs=None):
    """Sampled MetricValues in ladder order, all sharing one sd."""
    rungs = rungs or LADDER[: len(means)]
    return {rung: sampled(mean, sd) for rung, mean in zip(rungs, means)}


def test_ladder_is_strictly_ordered_five_rungs():
    """The five-rung size is load-bearing: 0.8/0.5 are derived from it.

    At five rungs tau = 1 - inv/5, so 0.8 admits one inversion and 0.5 admits
    two. Shrinking the ladder without re-deriving them leaves both unjustified.
    """
    assert LADDER == (
        Rung.RANDOM_INIT,
        Rung.ROUND,
        Rung.OVERFIT,
        Rung.PUBLIC_VIDEOMAEV2,
        Rung.SUPERVISED_VA,
    )
    assert len(set(LADDER)) == len(LADDER) == 5
    assert LADDER_REVISION


def test_mutilation_is_not_a_ladder_member():
    """Rung -1 is a self-test control a check asserts against directly.

    Folding it into the rank ladder would inflate every metric's tau with a
    difference no metric is allowed to miss.
    """
    assert Rung.MUTILATION not in LADDER


def test_ladder_tau_perfect_order_well_separated_is_tier_a():
    result = ladder_tau(_rung_values(0.0, 1.0, 2.0, 3.0, 4.0))
    assert result.tau == pytest.approx(1.0)
    assert result.tier is Tier.A
    assert result.present_rungs == 5


def test_ladder_tau_inverted_order_is_tier_c_not_a():
    result = ladder_tau(_rung_values(4.0, 3.0, 2.0, 1.0, 0.0))
    assert result.tau == pytest.approx(-1.0)
    assert result.tier is Tier.C


def test_ladder_tau_correct_order_all_gaps_inside_band_is_tau_zero_tier_c():
    """Correct order on unresolvable differences is not evidence.

    Every pair is tied, so tau-b's denominator has no untied pairs: tau is 0.0,
    which is the honest number -- the metric ranked nothing it could resolve.
    """
    result = ladder_tau(_rung_values(0.0, 0.5, 1.0, 1.5, 2.0, sd=1.0))
    assert result.tau == 0.0
    assert result.tier is Tier.C


def test_ladder_tau_ties_are_band_based_not_exact_equality():
    """A plateau differing in the sixth decimal must not score as clean order.

    Under an exact-equality tie rule tau-b's tie-awareness is dead code: real
    measurements are never bitwise equal, so a plateau would read as a perfect
    ordering (or an inversion, on the sign of that noise).
    """
    result = ladder_tau(
        _rung_values(1.0, 1.000001, 1.000002, 1.000003, 1.000004, sd=0.1)
    )
    assert result.tau == 0.0
    assert result.tier is Tier.C


def test_ladder_tau_deterministic_rungs_compared_exactly():
    """A deterministic value has no sampling error, so any difference is real.

    This is what lets a weights-only metric reach a high tier at all.
    """
    values = {
        rung: MetricValue.deterministic_value(mean)
        for rung, mean in zip(LADDER[:4], (1.0, 2.0, 3.0, 4.0))
    }
    result = ladder_tau(values)
    assert result.tau == pytest.approx(1.0)
    assert result.tier is Tier.A

    tied = {
        rung: MetricValue.deterministic_value(mean)
        for rung, mean in zip(LADDER[:4], (1.0, 1.0, 3.0, 4.0))
    }
    assert ladder_tau(tied).tau < 1.0


def test_ladder_tau_raises_on_mixed_deterministic_and_sampled():
    values = dict(_rung_values(0.0, 1.0, 2.0, 3.0))
    values[Rung.OVERFIT] = MetricValue.deterministic_value(2.0)
    with pytest.raises(ValueError, match="deterministic"):
        ladder_tau(values)


def test_ladder_tau_correct_order_majority_separated_is_tier_b_not_a():
    """Tier A needs every adjacent gap separated, not merely a good tau."""
    result = ladder_tau(_rung_values(0.0, 1.0, 2.0, 2.01, 3.0))
    assert result.tau >= 0.8
    assert result.tier is Tier.B


def test_tier_assignment_conjunctive_at_both_a_and_b():
    """Neither half suffices alone: rank without resolution, or the reverse."""
    # Good tau, one unresolvable gap -> not A.
    assert ladder_tau(_rung_values(0.0, 1.0, 2.0, 2.01, 3.0)).tier is not Tier.A
    # Every gap separated, but the order is wrong -> not B.
    every_gap_separated_wrong_order = ladder_tau(_rung_values(4.0, 3.0, 2.0, 1.0, 0.0))
    assert every_gap_separated_wrong_order.tier is Tier.C


def test_ladder_tau_ties_count_as_agreement_not_inversion():
    """The plateau reading must score above the wrong-order reading."""
    plateau = ladder_tau(_rung_values(0.0, 0.5, 1.0, 5.0, sd=1.0))
    wrong_order = ladder_tau(_rung_values(4.0, 3.0, 2.0, 1.0))
    assert plateau.tau > wrong_order.tau


def test_ladder_tau_two_present_rungs_is_tier_c_however_separated():
    """On two rungs tau is +/-1 by construction, so rank carries no evidence."""
    values = {
        Rung.RANDOM_INIT: sampled(0.0, 0.01),
        Rung.SUPERVISED_VA: sampled(10.0, 0.01),
    }
    result = ladder_tau(values)
    assert result.tau == pytest.approx(1.0)
    assert result.tier is Tier.C
    assert result.present_rungs == 2


def test_ladder_tau_four_present_rungs_can_reach_tier_a():
    result = ladder_tau(_rung_values(0.0, 1.0, 2.0, 3.0))
    assert result.present_rungs == 4
    assert result.tier is Tier.A


def test_ladder_tau_records_present_rung_count():
    """The ladder's own incompleteness belongs in the report, not laundered."""
    assert ladder_tau(_rung_values(0.0, 1.0, 2.0)).present_rungs == 3
    assert ladder_tau(_rung_values(0.0, 1.0, 2.0)).tier is Tier.C


# --- rung construction -----------------------------------------------------


def test_load_ladder_rung_random_init_needs_no_checkpoint(tmp_path):
    bundle = load_ladder_rung(Rung.RANDOM_INIT, _tiny_model_cfg())
    assert bundle.model is None
    assert bundle.encoder is not None


def test_load_ladder_rung_without_a_run_id_raises_rather_than_defaulting(tmp_path):
    with pytest.raises(ValueError, match="ladder_run_ids"):
        load_ladder_rung(Rung.ROUND, _tiny_model_cfg(), mesh_shared=tmp_path)


def test_load_ladder_rung_overfit_takes_a_solo_route_spec(tmp_path):
    """Rung 2 is a merge=false run, so its entry carries node and round.

    A bare run-id string cannot name a pending checkpoint (which is keyed by
    run/round/node), so the overfit rung's ladder_run_ids entry is a mapping.
    """
    cfg = _tiny_model_cfg()
    saved = write_pending_checkpoint(
        tmp_path, "overfit50", cfg, round_num=2, node="node055"
    )

    bundle = load_ladder_rung(
        Rung.OVERFIT,
        cfg,
        mesh_shared=tmp_path,
        ladder_run_ids={
            Rung.OVERFIT: {"run_id": "overfit50", "node": "node055", "round": 2}
        },
    )

    assert bundle.run_id == "overfit50"
    assert bundle.round == 2
    first = "patch_embed.proj.weight"
    gap = (
        (bundle.encoder.state_dict()[first] - saved.encoder.state_dict()[first])
        .abs()
        .max()
        .item()
    )
    assert gap == pytest.approx(0.0, abs=0.0)


def test_load_ladder_rung_solo_spec_missing_a_field_raises(tmp_path):
    """A half-specified route must not silently fall back to the master path."""
    with pytest.raises(ValueError, match="node"):
        load_ladder_rung(
            Rung.OVERFIT,
            _tiny_model_cfg(),
            mesh_shared=tmp_path,
            ladder_run_ids={Rung.OVERFIT: {"run_id": "overfit50", "round": 2}},
        )


def test_load_ladder_rung_supervised_va_names_its_blocking_dependency(tmp_path):
    with pytest.raises(NotImplementedError, match="VEATIC"):
        load_ladder_rung(Rung.SUPERVISED_VA, _tiny_model_cfg())


def test_remap_public_videomae_raises_on_unmatched_key():
    with pytest.raises(KeyError, match="no_such_key"):
        remap_public_videomae({"no_such_key": torch.zeros(3, 3)}, {})


def test_remap_public_videomae_raises_on_shape_mismatch():
    """Never a reshaped or truncated load -- strict=False is not used here."""
    with pytest.raises((KeyError, ValueError), match="does not match"):
        remap_public_videomae(
            {"patch_embed.proj.weight": torch.zeros(4, 4)},
            {"patch_embed.proj.weight": torch.zeros(8, 8)},
        )


# --- rung 3: public VideoMAEv2 ----------------------------------------------


def _synthetic_public_videomae_state_dict(
    *, depth: int, embed_dim: int, num_heads: int, mlp_hidden: int, patch_shape: tuple
) -> dict:
    """Mimics mae-b/pytorch_model.bin's real key set and naming (task 1.0's
    findings): `backbone.`-prefixed, `q_bias`/`v_bias` split, `mlp.fc1`/`fc2`,
    `fc_norm`, a `cls_head`, and -- crucially -- no `pos_embed` key at all.
    """
    sd = {
        "backbone.patch_embed.proj.weight": torch.randn(embed_dim, *patch_shape),
        "backbone.patch_embed.proj.bias": torch.randn(embed_dim),
        "backbone.fc_norm.weight": torch.randn(embed_dim),
        "backbone.fc_norm.bias": torch.randn(embed_dim),
        "cls_head.fc_cls.weight": torch.randn(101, embed_dim),
        "cls_head.fc_cls.bias": torch.randn(101),
    }
    for i in range(depth):
        sd.update(
            {
                f"backbone.blocks.{i}.norm1.weight": torch.randn(embed_dim),
                f"backbone.blocks.{i}.norm1.bias": torch.randn(embed_dim),
                f"backbone.blocks.{i}.attn.qkv.weight": torch.randn(
                    3 * embed_dim, embed_dim
                ),
                f"backbone.blocks.{i}.attn.q_bias": torch.randn(embed_dim),
                f"backbone.blocks.{i}.attn.v_bias": torch.randn(embed_dim),
                f"backbone.blocks.{i}.attn.proj.weight": torch.randn(
                    embed_dim, embed_dim
                ),
                f"backbone.blocks.{i}.attn.proj.bias": torch.randn(embed_dim),
                f"backbone.blocks.{i}.norm2.weight": torch.randn(embed_dim),
                f"backbone.blocks.{i}.norm2.bias": torch.randn(embed_dim),
                f"backbone.blocks.{i}.mlp.fc1.weight": torch.randn(
                    mlp_hidden, embed_dim
                ),
                f"backbone.blocks.{i}.mlp.fc1.bias": torch.randn(mlp_hidden),
                f"backbone.blocks.{i}.mlp.fc2.weight": torch.randn(
                    embed_dim, mlp_hidden
                ),
                f"backbone.blocks.{i}.mlp.fc2.bias": torch.randn(embed_dim),
            }
        )
    return sd


def test_public_videomaev2_rung_reports_unavailable_on_depth_mismatch(tmp_path):
    """A synthetic fixture, not the real 691MB file -- runs without node090 access."""
    sd = _synthetic_public_videomae_state_dict(
        depth=3, embed_dim=16, num_heads=2, mlp_hidden=32, patch_shape=(3, 2, 16, 16)
    )
    ckpt_path = tmp_path / "fake_mae_b.pth"
    torch.save({"model": sd}, ckpt_path)

    cfg = _tiny_model_cfg()  # depth=2, embed_dim=16 -- mismatches depth=3 above
    with pytest.raises(RungUnavailable, match="depth"):
        load_ladder_rung(
            Rung.PUBLIC_VIDEOMAEV2, cfg, public_videomaev2_ckpt_path=ckpt_path
        )


def test_public_videomaev2_rung_loads_cleanly_when_geometry_matches(tmp_path):
    cfg = _tiny_model_cfg()  # depth=2, embed_dim=16
    sd = _synthetic_public_videomae_state_dict(
        depth=int(cfg.depth),
        embed_dim=int(cfg.embed_dim),
        num_heads=int(cfg.num_heads),
        mlp_hidden=int(cfg.embed_dim * cfg.mlp_ratio),
        patch_shape=(3, 2, 16, 16),
    )
    ckpt_path = tmp_path / "fake_mae_b.pth"
    torch.save({"model": sd}, ckpt_path)

    bundle = load_ladder_rung(
        Rung.PUBLIC_VIDEOMAEV2, cfg, public_videomaev2_ckpt_path=ckpt_path
    )

    assert bundle.run_id == "public_videomaev2"
    target_sd = bundle.encoder.state_dict()
    assert torch.allclose(
        target_sd["patch_embed.proj.weight"], sd["backbone.patch_embed.proj.weight"]
    )
    assert torch.allclose(target_sd["norm.weight"], sd["backbone.fc_norm.weight"])
    assert torch.allclose(
        target_sd["blocks.0.mlp.net.0.weight"], sd["backbone.blocks.0.mlp.fc1.weight"]
    )
    assert torch.allclose(
        target_sd["blocks.0.mlp.net.3.weight"], sd["backbone.blocks.0.mlp.fc2.weight"]
    )
    expected_qkv_bias = torch.cat(
        (
            sd["backbone.blocks.0.attn.q_bias"],
            torch.zeros_like(sd["backbone.blocks.0.attn.v_bias"]),
            sd["backbone.blocks.0.attn.v_bias"],
        )
    )
    assert torch.allclose(target_sd["blocks.0.attn.qkv.bias"], expected_qkv_bias)


def test_public_videomaev2_pos_embed_is_computed_not_left_random(tmp_path):
    """The checkpoint has no pos_embed key; the loader must not silently keep
    the encoder's random init there."""
    cfg = _tiny_model_cfg()
    sd = _synthetic_public_videomae_state_dict(
        depth=int(cfg.depth),
        embed_dim=int(cfg.embed_dim),
        num_heads=int(cfg.num_heads),
        mlp_hidden=int(cfg.embed_dim * cfg.mlp_ratio),
        patch_shape=(3, 2, 16, 16),
    )
    ckpt_path = tmp_path / "fake_mae_b.pth"
    torch.save({"model": sd}, ckpt_path)

    torch.manual_seed(0)
    baseline = VideoMAEEncoder(cfg).pos_embed.detach().clone()

    bundle = load_ladder_rung(
        Rung.PUBLIC_VIDEOMAEV2, cfg, public_videomaev2_ckpt_path=ckpt_path
    )

    assert not torch.allclose(bundle.encoder.pos_embed, baseline)
    expected = _public_videomae_sinusoid_pos_embed(
        bundle.encoder.pos_embed.shape[1], int(cfg.embed_dim)
    )
    assert torch.allclose(bundle.encoder.pos_embed, expected)

    # Independent of the production formula: position 0's angle is 0 for
    # every hidden index (numerator is position=0), so sin(0)=0 on the even
    # slots and cos(0)=1 on the odd slots -- true regardless of d_hid or how
    # `_public_videomae_sinusoid_pos_embed` computes the rest of the table.
    expected_row0 = torch.tensor([0.0, 1.0] * (int(cfg.embed_dim) // 2))
    assert torch.allclose(bundle.encoder.pos_embed[0, 0], expected_row0)


def test_public_videomaev2_rung_raises_when_no_local_path_given():
    with pytest.raises(RungUnavailable, match="PUBLIC_VIDEOMAEV2_LOCAL_PATH"):
        load_ladder_rung(Rung.PUBLIC_VIDEOMAEV2, _tiny_model_cfg())
