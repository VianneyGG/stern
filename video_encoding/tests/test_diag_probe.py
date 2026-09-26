"""Tests for `scripts/diag_probe.py`, the frozen attentive probe."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from data.labels import write_label_csv
from data.manifest import ManifestRecord, read_manifest, write_manifest
from video_encoding.models.videomaev2.encoder import VideoMAEEncoder
from video_encoding.models.videomaev2.regressor import LinearVAHead
from video_encoding.scripts._diag_common import DiagBundle, DiagReport, MetricValue
from video_encoding.scripts.diag_probe import (
    AttentiveProbe,
    _fit_and_score_probe,
    _identity_disjoint_split,
    check,
    probe_by_depth,
    probe_ccc,
    standardise_on_fit,
)


def _random_init_bundle() -> DiagBundle:
    encoder = VideoMAEEncoder(_tiny_model_cfg()).eval()
    return DiagBundle(encoder=encoder, model=None, run_id="", round=None)


pytestmark = pytest.mark.integration

NUM_FRAMES = 4
IMAGE_SIZE = 32


def _tiny_model_cfg(**overrides):
    cfg = {
        "num_frames": NUM_FRAMES,
        "image_size": IMAGE_SIZE,
        "in_channels": 3,
        "tubelet_size": 2,
        "patch_size": 16,
        "embed_dim": 16,
        "depth": 2,
        "num_heads": 2,
        "mlp_ratio": 2.0,
        "dropout": 0.0,
        "drop_path_rate": 0.0,
    }
    cfg.update(overrides)
    return OmegaConf.create(cfg)


def _write_indexed_frames(clip_dir: Path, n: int, size: int = IMAGE_SIZE) -> None:
    clip_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        value = (i * 17) % 256
        img = np.full((size, size, 3), value, dtype=np.uint8)
        Image.fromarray(img).save(clip_dir / f"{i:04d}.png")


# 40 frames per clip, not 16: a window under `PROBE_FRAME_STRIDE` spans
# `(num_frames - 1) * stride + 1` frames, and a fixture short enough to yield
# one or two near-identical windows per clip makes the probe predict a constant
# -- which reads as a legitimate CCC of 0.0 and quietly defeats every test here
# that asserts seed sensitivity or a non-zero band.
def _write_probe_manifest(
    root: Path, *, n_identities: int, clips_per_identity: int = 1, n_frames: int = 16
) -> Path:
    """Writes `n_identities` identities each with `clips_per_identity` clips
    of `n_frames` frames, plus a matching probe-role manifest.csv."""
    records = []
    for i in range(n_identities):
        identity = f"id{i}"
        for c in range(clips_per_identity):
            clip = f"{identity}/clip{c}"
            _write_indexed_frames(root / clip, n_frames)
            label_path = f"{clip}.csv"
            # A ramp across frame position (not a constant/alternating value)
            # so windows at different starts have distinct, non-degenerate
            # labels -- CCC against a zero-variance target is trivially 0.
            write_label_csv(
                root / label_path,
                [
                    (-1.0 + 2.0 * t / (n_frames - 1), 1.0 - 2.0 * t / (n_frames - 1))
                    for t in range(n_frames)
                ],
            )
            records.append(
                ManifestRecord(
                    identity_id=identity,
                    clip_path=clip,
                    duration_s=float(n_frames),
                    role="probe",
                    label_path=label_path,
                )
            )
    manifest_path = root / "manifest.csv"
    write_manifest(manifest_path, records)
    return manifest_path


def _tiny_bundle() -> DiagBundle:
    encoder = VideoMAEEncoder(_tiny_model_cfg()).eval()
    return DiagBundle(encoder=encoder, model=None, run_id="test", round=None)


# --- AttentiveProbe ----------------------------------------------------------


def test_attentive_probe_output_shape_and_range():
    probe = AttentiveProbe(embed_dim=768, num_heads=16, n_blocks=3)
    tokens = torch.randn(4, 197, 768)

    out = probe(tokens)

    assert out.shape == (4, 2)
    assert torch.all(out >= -1.0) and torch.all(out <= 1.0)


def test_final_block_is_linear_va_head():
    probe = AttentiveProbe(embed_dim=768, num_heads=16, n_blocks=3)
    assert isinstance(probe.head, LinearVAHead)
    assert probe.head.embed_dim == 768


def test_num_heads_reaches_the_readout():
    probe = AttentiveProbe(embed_dim=768, num_heads=16, n_blocks=3)
    assert probe.head.num_heads == 16  # LinearVAHead's own default is 12


def test_probe_capacity_depends_only_on_construction_args():
    """`AttentiveProbe` takes no depth/rung argument at all -- its parameter
    count is a pure function of (embed_dim, num_heads, n_blocks), so the same
    probe is used unchanged at every depth and against every ladder rung."""
    probe = AttentiveProbe(768, 16, 3)
    count = sum(p.numel() for p in probe.parameters())
    assert count == 23_036_162


# --- identity-disjoint split --------------------------------------------------


def _records(identity_ids: list[str]) -> list[ManifestRecord]:
    return [
        ManifestRecord(
            identity_id=iid,
            clip_path=f"{iid}/clip0",
            duration_s=1.0,
            role="probe",
            label_path=None,
        )
        for iid in identity_ids
    ]


def test_probe_ccc_split_is_identity_disjoint():
    records = _records([f"id{i}" for i in range(10)])
    fit_ids, scored_ids = _identity_disjoint_split(records)
    assert len(fit_ids & scored_ids) == 0
    assert fit_ids | scored_ids == {r.identity_id for r in records}


def test_probe_ccc_split_is_identical_across_seeds():
    """The split is derived from identity_ids alone -- it must not vary with
    a caller-supplied seed, since both arms need the same held-out set."""
    records = _records([f"id{i}" for i in range(10)])
    split_a = _identity_disjoint_split(records)
    split_b = _identity_disjoint_split(records)
    assert split_a == split_b


def test_probe_ccc_windows_of_one_clip_do_not_straddle_split(tmp_path):
    """Every window drawn from one clip belongs to exactly one side of the
    split, because the split partitions identities, and `_probe_dataset`
    is built from `fit_records`/`scored_records`, which are disjoint by
    identity."""
    manifest_path = _write_probe_manifest(tmp_path, n_identities=10, n_frames=40)
    records = read_manifest(manifest_path, expected_role="probe")
    fit_ids, scored_ids = _identity_disjoint_split(records)
    fit_records = [r for r in records if r.identity_id in fit_ids]
    scored_records = [r for r in records if r.identity_id in scored_ids]
    fit_clips = {r.clip_path for r in fit_records}
    scored_clips = {r.clip_path for r in scored_records}
    assert len(fit_clips & scored_clips) == 0


# --- probe_ccc -----------------------------------------------------------------


def test_probe_ccc_returns_finite_score_on_tiny_fixture(tmp_path):
    manifest_path = _write_probe_manifest(tmp_path, n_identities=10, n_frames=40)
    bundle = _tiny_bundle()

    score = probe_ccc(bundle, manifest_path, layer=1, seed=0, epochs=1)

    assert isinstance(score, float)
    assert score == score  # not NaN
    assert score not in (float("inf"), float("-inf"))


def test_probe_ccc_same_seed_reproduces_score(tmp_path):
    manifest_path = _write_probe_manifest(tmp_path, n_identities=10, n_frames=40)
    bundle = _tiny_bundle()

    score_a = probe_ccc(bundle, manifest_path, layer=1, seed=0, epochs=1)
    score_b = probe_ccc(bundle, manifest_path, layer=1, seed=0, epochs=1)

    assert score_a == pytest.approx(score_b)


def test_probe_ccc_is_seed_sensitive(tmp_path):
    manifest_path = _write_probe_manifest(tmp_path, n_identities=10, n_frames=40)
    bundle = _tiny_bundle()

    score_a = probe_ccc(bundle, manifest_path, layer=1, seed=0, epochs=1)
    score_b = probe_ccc(bundle, manifest_path, layer=1, seed=1, epochs=1)

    assert score_a != pytest.approx(score_b)


def test_probe_ccc_leaves_global_rng_untouched(tmp_path):
    manifest_path = _write_probe_manifest(tmp_path, n_identities=10, n_frames=40)
    bundle = _tiny_bundle()

    torch.manual_seed(1234)
    before = torch.get_rng_state()

    probe_ccc(bundle, manifest_path, layer=1, seed=0, epochs=1)

    after = torch.get_rng_state()
    assert torch.equal(before, after)


# --- probe_by_depth -----------------------------------------------------------


def test_probe_by_depth_every_layer_has_both_arms(tmp_path):
    manifest_path = _write_probe_manifest(tmp_path, n_identities=10, n_frames=40)
    bundle = _tiny_bundle()
    control = _random_init_bundle()

    per_layer, _ = probe_by_depth(
        bundle, control, manifest_path, layers=[1, 2], n_seeds=3, epochs=1
    )

    assert set(per_layer.keys()) == {1, 2}
    for checkpoint_value, control_value in per_layer.values():
        assert isinstance(checkpoint_value, MetricValue)
        assert isinstance(control_value, MetricValue)


def test_probe_by_depth_final_block_reference_present(tmp_path):
    """The tiny fixture's encoder has depth=2 (`_tiny_model_cfg`); requesting
    only layer=1 must still yield a reference reading at the final block
    (layer=2), unrequested but always reported."""
    manifest_path = _write_probe_manifest(tmp_path, n_identities=10, n_frames=40)
    bundle = _tiny_bundle()
    control = _random_init_bundle()

    per_layer, final_block_reference = probe_by_depth(
        bundle, control, manifest_path, layers=[1], n_seeds=3, epochs=1
    )

    assert set(per_layer.keys()) == {1}
    assert isinstance(final_block_reference, MetricValue)


def test_probe_by_depth_random_init_vs_itself_is_null(tmp_path):
    """`separated_from` *raises*, not returns True/False, on a zero pooled
    band -- this is not a difference-vs-zero test, it is the null case the
    check is designed to report correctly."""
    manifest_path = _write_probe_manifest(tmp_path, n_identities=10, n_frames=40)
    bundle = _random_init_bundle()

    per_layer, _ = probe_by_depth(
        bundle, bundle, manifest_path, layers=[1, 2], n_seeds=3, epochs=1
    )

    for checkpoint_value, control_value in per_layer.values():
        assert checkpoint_value.separated_from(control_value, k=2.0) is False


def test_probe_by_depth_seed_variance_is_nonzero(tmp_path):
    manifest_path = _write_probe_manifest(tmp_path, n_identities=10, n_frames=40)
    bundle = _tiny_bundle()
    control = _random_init_bundle()

    per_layer, _ = probe_by_depth(
        bundle, control, manifest_path, layers=[1], n_seeds=3, epochs=5
    )

    checkpoint_value, control_value = per_layer[1]
    assert checkpoint_value.sd != 0.0
    assert control_value.sd != 0.0


def test_encoder_runs_once_per_bundle(tmp_path, monkeypatch):
    """Encoder forward calls are gated by dataset batch count alone -- adding
    layers or seeds must not grow it (design.md's "two passes total, not
    layers x arms x n_seeds x epochs")."""
    manifest_path = _write_probe_manifest(tmp_path, n_identities=10, n_frames=40)
    bundle = _tiny_bundle()
    control = _random_init_bundle()

    import video_encoding.scripts.diag_probe as diag_probe_module

    real_fn = diag_probe_module.forward_with_hidden_states
    call_count = {"n": 0}

    def _counting(encoder, pixel_values):
        call_count["n"] += 1
        return real_fn(encoder, pixel_values)

    monkeypatch.setattr(diag_probe_module, "forward_with_hidden_states", _counting)

    probe_by_depth(bundle, control, manifest_path, layers=[1], n_seeds=3, epochs=1)
    count_one_layer = call_count["n"]

    call_count["n"] = 0
    probe_by_depth(bundle, control, manifest_path, layers=[1, 2], n_seeds=3, epochs=1)
    count_two_layers = call_count["n"]

    call_count["n"] = 0
    probe_by_depth(bundle, control, manifest_path, layers=[1], n_seeds=6, epochs=1)
    count_more_seeds = call_count["n"]

    assert count_one_layer == count_two_layers == count_more_seeds


# --- check() ---------------------------------------------------------------


def _tiny_checkpoint_bundle() -> DiagBundle:
    encoder = VideoMAEEncoder(_tiny_model_cfg()).eval()
    return DiagBundle(encoder=encoder, model=None, run_id="ckpt", round=None)


def test_check_skips_cleanly_without_veatic_root(monkeypatch):
    monkeypatch.delenv("VEATIC_ROOT", raising=False)
    bundle = _tiny_checkpoint_bundle()

    reports = check(bundle, {"control_bundle": _random_init_bundle()})

    assert set(reports.keys()) == {"probe_by_depth"}
    report = reports["probe_by_depth"]
    assert isinstance(report, DiagReport)
    assert report.metrics == {}
    assert "skipped" in report.meta


def test_check_skips_cleanly_with_missing_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("VEATIC_ROOT", str(tmp_path))  # no manifest.csv written
    bundle = _tiny_checkpoint_bundle()

    reports = check(bundle, {"control_bundle": _random_init_bundle()})

    report = reports["probe_by_depth"]
    assert report.metrics == {}
    assert "skipped" in report.meta


def test_check_skips_cleanly_without_control_bundle(tmp_path, monkeypatch):
    _write_probe_manifest(tmp_path, n_identities=10, n_frames=40)
    monkeypatch.setenv("VEATIC_ROOT", str(tmp_path))
    bundle = _tiny_checkpoint_bundle()

    reports = check(bundle, {})

    report = reports["probe_by_depth"]
    assert report.metrics == {}
    assert "skipped" in report.meta


def test_check_runs_probe_by_depth_with_veatic_and_control_present(
    tmp_path, monkeypatch
):
    _write_probe_manifest(tmp_path, n_identities=10, n_frames=40)
    monkeypatch.setenv("VEATIC_ROOT", str(tmp_path))
    bundle = _tiny_checkpoint_bundle()
    control = _random_init_bundle()

    reports = check(bundle, {"control_bundle": control, "layers": [1], "n_seeds": 3})

    report = reports["probe_by_depth"]
    assert "skipped" not in report.meta
    assert report.meta["LADDER_REVISION"]
    assert "layer_1_checkpoint" in report.metrics
    assert "layer_1_control" in report.metrics
    assert "layer_1_diff" in report.metrics
    assert "final_block_reference" in report.metrics


def test_check_ignores_fit_budget_keys_in_inputs(tmp_path, monkeypatch):
    """`check()` SHALL NOT read a fit parameter out of `inputs` -- an epochs
    key there must not change the pinned PROBE_FIT_EPOCHS fit."""
    _write_probe_manifest(tmp_path, n_identities=10, n_frames=40)
    monkeypatch.setenv("VEATIC_ROOT", str(tmp_path))
    bundle = _tiny_checkpoint_bundle()
    control = _random_init_bundle()

    import video_encoding.scripts.diag_probe as diag_probe_module

    captured_epochs = []
    real_probe_by_depth = diag_probe_module.probe_by_depth

    def _capturing(*args, **kwargs):
        captured_epochs.append(kwargs["epochs"])
        return real_probe_by_depth(*args, **kwargs)

    monkeypatch.setattr(diag_probe_module, "probe_by_depth", _capturing)

    check(
        bundle,
        {"control_bundle": control, "layers": [1], "n_seeds": 3, "epochs": 1},
    )

    assert captured_epochs == [diag_probe_module.PROBE_FIT_EPOCHS]


# --- fit-split standardisation ------------------------------------------------


def test_standardise_on_fit_centres_and_scales_the_fit_split():
    fit = (torch.randn(40, 6, 8) * 37.0 + 12.0).half()
    scored = (torch.randn(10, 6, 8) * 37.0 + 12.0).half()
    standardise_on_fit(fit, scored)
    assert fit.float().mean().abs() < 0.05
    assert abs(fit.float().std().item() - 1.0) < 0.1


def test_standardise_on_fit_does_not_use_the_scored_splits_own_statistics():
    """The leak guard: a scored split shifted away from the fit split must stay
    shifted after the transform. Standardising it on its own mean would centre
    it at zero and quietly hand the probe information about the test set."""
    fit = (torch.zeros(40, 4, 3) + 5.0).half()
    fit[:, :, :] += (torch.randn(40, 4, 3) * 10.0).half()
    scored = fit[:10].clone() + 100.0
    standardise_on_fit(fit, scored)
    # (100 offset) / (fit sd ~10) ~ 10 sds away, nowhere near centred
    assert scored.float().mean().item() > 5.0


def test_standardise_on_fit_makes_the_reading_scale_invariant():
    """Two arms whose streams differ only by a global scale must standardise to
    the same features -- stream norm is the confound this removes."""
    base_fit = torch.randn(40, 4, 6)
    base_scored = torch.randn(10, 4, 6)
    small_fit, small_scored = base_fit.half(), base_scored.half()
    big_fit, big_scored = (base_fit * 400.0).half(), (base_scored * 400.0).half()
    standardise_on_fit(small_fit, small_scored)
    standardise_on_fit(big_fit, big_scored)
    assert torch.allclose(small_fit.float(), big_fit.float(), atol=2e-2)
    assert torch.allclose(small_scored.float(), big_scored.float(), atol=2e-2)


def _informative_tokens(n: int, dim: int, scale: float, seed: int):
    """Tokens whose feature-0 mean is the label, so a working probe must score
    above zero and a dead one cannot."""
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randn(n, 4, dim, generator=generator) * scale
    signal = torch.rand(n, generator=generator) * 2.0 - 1.0
    tokens[:, :, 0] = signal.unsqueeze(1) * scale
    labels = torch.stack([signal, signal], dim=1)
    return tokens.half(), labels


def test_standardisation_is_what_makes_a_realistic_stream_scale_learnable():
    """The 2026-09-23 failure, pinned. At a residual-stream scale this
    workstream actually produces, the tanh head saturates and the fit barely
    moves; the same tokens standardised recover the signal. On real AFEW-VA
    tokens the unstandardised score was exactly 0.0 for every arm and seed --
    a dead fit wearing the shape of a precise null."""
    raw_fit, fit_labels = _informative_tokens(48, 16, scale=500.0, seed=0)
    raw_scored, scored_labels = _informative_tokens(16, 16, scale=500.0, seed=1)
    std_fit, std_scored = raw_fit.clone(), raw_scored.clone()
    standardise_on_fit(std_fit, std_scored)

    def _score(fit_tokens, scored_tokens):
        return _fit_and_score_probe(
            fit_tokens,
            fit_labels,
            scored_tokens,
            scored_labels,
            seed=0,
            epochs=20,
            embed_dim=16,
        )

    raw_score = _score(raw_fit, raw_scored)
    standardised_score = _score(std_fit, std_scored)
    assert raw_score < 0.05
    assert standardised_score > 0.1
    assert standardised_score > raw_score + 0.1


def test_standardisation_revives_the_fit_at_that_same_scale():
    fit_tokens, fit_labels = _informative_tokens(48, 16, scale=500.0, seed=0)
    scored_tokens, scored_labels = _informative_tokens(16, 16, scale=500.0, seed=1)
    standardise_on_fit(fit_tokens, scored_tokens)
    score = _fit_and_score_probe(
        fit_tokens,
        fit_labels,
        scored_tokens,
        scored_labels,
        seed=0,
        epochs=20,
        embed_dim=16,
    )
    assert score > 0.1
