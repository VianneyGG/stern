import copy

import pytest
import torch
from omegaconf import OmegaConf

from video_encoding.models.videomaev2.encoder import VideoMAEEncoder
from video_encoding.scripts._diag_common import (
    DiagBundle,
    MUTILATION_BLOCK,
    MUTILATION_TARGET_KEY,
)
from video_encoding.scripts.diag_depth import (
    branch_norm_ratio,
    check,
    delta_from_init,
    layerscale_gamma,
    residual_branch_norm,
)

MODEL_CFG_PATH = "video_encoding/conf/model/videomae_base.yaml"


def _cfg(*, depth: int | None = None, layer_scale_init: float | None = None):
    """Size fixtures at MUTILATION_BLOCK + 2 by default (see tasks.md 1.3) so
    Section 6's mutilation-block index stays in range; individual tests may
    ask for a larger depth. layer_scale_init is never overridden here except
    by the one test that deliberately pins the enabled branch -- every other
    fixture uses the repo config's own (null) value."""
    cfg = OmegaConf.load(MODEL_CFG_PATH)
    cfg.depth = depth if depth is not None else MUTILATION_BLOCK + 2
    cfg.embed_dim = 32
    cfg.num_heads = 4
    cfg.num_frames = 4
    cfg.image_size = 32
    if layer_scale_init is not None:
        cfg.layer_scale_init = layer_scale_init
    return cfg


def _bundle(cfg, run_id: str = "r", round: int | None = None) -> DiagBundle:
    encoder = VideoMAEEncoder(cfg).eval()
    return DiagBundle(encoder=encoder, model=None, run_id=run_id, round=round)


# --- residual_branch_norm -------------------------------------------------


def test_residual_branch_norm_self_comparison_is_one():
    bundle = _bundle(_cfg())
    result = residual_branch_norm(bundle, bundle)
    for block_result in result.values():
        assert block_result["attn"].mean == pytest.approx(1.0)
        assert block_result["mlp"].mean == pytest.approx(1.0)


def test_residual_branch_norm_scaled_block_reports_the_scale_factor():
    cfg = _cfg(depth=3)
    ref_bundle = _bundle(cfg)
    mutated_bundle = copy.deepcopy(ref_bundle)
    with torch.no_grad():
        mutated_bundle.encoder.blocks[2].attn.proj.weight.mul_(10.0)

    result = residual_branch_norm(mutated_bundle, ref_bundle)

    assert result[2]["attn"].mean == pytest.approx(10.0, rel=1e-4)
    assert result[0]["attn"].mean == pytest.approx(1.0)


def test_residual_branch_norm_raises_on_depth_mismatch():
    bundle = _bundle(_cfg(depth=2))
    ref_bundle = _bundle(_cfg(depth=3))
    with pytest.raises(ValueError, match="zip"):
        residual_branch_norm(bundle, ref_bundle)


# --- layerscale_gamma ------------------------------------------------------


def test_layerscale_gamma_is_none_when_disabled():
    bundle = _bundle(_cfg())
    assert layerscale_gamma(bundle) is None


def test_layerscale_gamma_reports_rms_when_enabled():
    bundle = _bundle(_cfg(layer_scale_init=1e-5))
    result = layerscale_gamma(bundle)
    assert result is not None
    for value in result.values():
        assert value == pytest.approx(1e-5, rel=1e-3)


# --- check() ----------------------------------------------------------


def test_check_emits_skipped_reports_when_ref_bundle_absent():
    bundle = _bundle(_cfg())
    reports = check(bundle, {})
    assert reports["residual_branch_norm"].metrics == {}
    assert reports["residual_branch_norm"].meta["skipped"] == "no ref_bundle supplied"
    assert reports["layerscale_gamma"].metrics == {}
    assert reports["layerscale_gamma"].meta["skipped"] == "layerscale disabled"


def test_check_emits_populated_reports_when_ref_bundle_present():
    bundle = _bundle(_cfg())
    reports = check(bundle, {"ref_bundle": bundle})
    assert "skipped" not in reports["residual_branch_norm"].meta
    assert reports["residual_branch_norm"].metrics
    assert "skipped" in reports["layerscale_gamma"].meta  # repo default is disabled


def test_check_reports_carry_ladder_revision():
    bundle = _bundle(_cfg())
    reports = check(bundle, {})
    for report in reports.values():
        assert "LADDER_REVISION" in report.meta


# --- branch_norm_ratio ------------------------------------------------


def _pixel_values(cfg):
    return torch.randn(
        1,
        int(cfg.in_channels),
        int(cfg.num_frames),
        int(cfg.image_size),
        int(cfg.image_size),
    )


def test_branch_norm_ratio_reports_one_row_per_block():
    cfg = _cfg()
    bundle = _bundle(cfg)
    result = branch_norm_ratio(bundle, _pixel_values(cfg))
    assert set(result.keys()) == set(range(cfg.depth))
    for block_result in result.values():
        assert set(block_result.keys()) == {
            "attn",
            "mlp",
            "residual",
            "attn_ratio",
            "mlp_ratio",
        }


def test_branch_norm_ratio_ratios_match_their_definition():
    cfg = _cfg()
    bundle = _bundle(cfg)
    result = branch_norm_ratio(bundle, _pixel_values(cfg))
    for block_result in result.values():
        assert block_result["attn_ratio"].mean == pytest.approx(
            block_result["attn"].mean / block_result["residual"].mean
        )
        assert block_result["mlp_ratio"].mean == pytest.approx(
            block_result["mlp"].mean / block_result["residual"].mean
        )


def test_check_emits_branch_norm_ratio_report_with_and_without_real_input():
    cfg = _cfg()
    bundle = _bundle(cfg)

    without_input = check(bundle, {})
    assert without_input["branch_norm_ratio"].metrics == {}
    assert (
        without_input["branch_norm_ratio"].meta["skipped"] == "no input tensor supplied"
    )

    with_input = check(bundle, {"real": _pixel_values(cfg)})
    assert "skipped" not in with_input["branch_norm_ratio"].meta
    assert with_input["branch_norm_ratio"].metrics


# --- delta_from_init and the mutilation smoke test -------------------


def test_delta_from_init_self_comparison_is_zero():
    bundle = _bundle(_cfg())
    result = delta_from_init(bundle, bundle)
    for value in result.values():
        assert value == pytest.approx(0.0, abs=1e-8)


def test_delta_from_init_measures_l2_norm_of_perturbation():
    """Independently hand-computes the expected per-block L2 norm from the
    same perturbation delta_from_init is asked to measure, rather than
    reusing delta_from_init's own formula -- closes the gap the
    test-quality audit flagged: a delta_from_init that always returned 0.0
    passed every other test in this file."""
    cfg = _cfg(depth=3)
    ref_bundle = _bundle(cfg, run_id="r", round=0)
    mutated_bundle = copy.deepcopy(ref_bundle)
    mutated_bundle = DiagBundle(
        encoder=mutated_bundle.encoder, model=None, run_id="r", round=1
    )
    with torch.no_grad():
        mutated_bundle.encoder.blocks[1].attn.proj.weight.add_(1.0)

    expected = {}
    for i, (block, ref_block) in enumerate(
        zip(mutated_bundle.encoder.blocks, ref_bundle.encoder.blocks, strict=True)
    ):
        total = 0.0
        for (_, p), (_, ref_p) in zip(
            block.named_parameters(), ref_block.named_parameters(), strict=True
        ):
            total += (p - ref_p).norm().item() ** 2
        expected[i] = total**0.5

    result = delta_from_init(mutated_bundle, ref_bundle)

    for i in expected:
        assert result[i] == pytest.approx(expected[i], rel=1e-5)
    assert result[1] > 0.0
    assert result[0] == pytest.approx(0.0, abs=1e-8)
    assert result[2] == pytest.approx(0.0, abs=1e-8)


def test_delta_from_init_raises_on_unrelated_reference():
    bundle = _bundle(_cfg(), run_id="a")
    ref_bundle = _bundle(_cfg(), run_id="b")
    with pytest.raises(ValueError, match="init"):
        delta_from_init(bundle, ref_bundle)


def _perturb_every_block(encoder, sigma: float, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for block in encoder.blocks:
            for weight in (
                block.attn.proj.weight,
                block.mlp.net[0].weight,
                block.mlp.net[3].weight,
            ):
                weight.add_(torch.randn(weight.shape, generator=generator) * sigma)


def test_residual_branch_norm_detects_mutilation_delta_from_init_runs_without_raising():
    """The rung-(-1) self-test: residual_branch_norm's mlp ratio goes to
    exactly 0.0 at the mutilated block (scale-free, no threshold). This test
    does NOT assert delta_from_init separates the mutilated block -- it
    cannot, since it measures distance moved, which detects a block that
    never learned, not one that was damaged; see design.md's "delta_from_init
    cannot detect the mutilation" section and
    test_delta_from_init_measures_l2_norm_of_perturbation for its real
    value-level coverage."""
    cfg = _cfg()
    init_bundle = _bundle(cfg, run_id="mut", round=0)

    trained_bundle = copy.deepcopy(init_bundle)
    _perturb_every_block(trained_bundle.encoder, sigma=0.05, seed=0)
    trained_bundle = DiagBundle(
        encoder=trained_bundle.encoder, model=None, run_id="mut", round=1
    )

    mutilated_bundle = copy.deepcopy(trained_bundle)
    target = dict(mutilated_bundle.encoder.named_parameters())[MUTILATION_TARGET_KEY]
    with torch.no_grad():
        target.zero_()

    rbn = residual_branch_norm(mutilated_bundle, init_bundle)
    assert rbn[MUTILATION_BLOCK]["mlp"].mean == pytest.approx(0.0, abs=1e-8)
    assert rbn[MUTILATION_BLOCK]["attn"].mean != pytest.approx(0.0, abs=1e-8)
    for i, block_result in rbn.items():
        if i != MUTILATION_BLOCK:
            assert block_result["mlp"].mean != pytest.approx(0.0, abs=1e-8)

    # Companion assertion only: delta_from_init runs on the mutilated bundle
    # without raising. A separation claim (mutilated > others) is false --
    # see design.md -- and is deliberately not asserted here.
    dfi = delta_from_init(mutilated_bundle, init_bundle)
    assert set(dfi.keys()) == set(range(cfg.depth))


def test_check_emits_delta_from_init_report_with_and_without_ref_bundle():
    bundle = _bundle(_cfg())

    without_ref = check(bundle, {})
    assert without_ref["delta_from_init"].metrics == {}
    assert without_ref["delta_from_init"].meta["skipped"] == "no ref_bundle supplied"

    with_ref = check(bundle, {"ref_bundle": bundle})
    assert "skipped" not in with_ref["delta_from_init"].meta
    assert with_ref["delta_from_init"].metrics
