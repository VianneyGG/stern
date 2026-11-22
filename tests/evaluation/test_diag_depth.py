import copy

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from stern.models.videomae.encoder import VideoMAEEncoder
from stern.evaluation.diagnostics.common import (
    DiagBundle,
    MUTILATION_BLOCK,
    MUTILATION_TARGET_KEY,
    random_init_bundle,
)
from stern.evaluation.diagnostics.depth import (
    branch_norm_ratio,
    check,
    cka,
    cka_between,
    cka_matrix,
    delta_from_init,
    layerscale_gamma,
    residual_branch_norm,
)
from stern.evaluation.diagnostics.reconstruction import temporal_split

MODEL_CFG_PATH = "configs/model/videomae_base.yaml"


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


def test_mutilation_constants_are_pinned_by_the_spec():
    """specs/ssl-diag-depth-checks pins both values; fixtures size off
    MUTILATION_BLOCK, so a silent change would fail as an IndexError far
    from its cause. Change the requirement first, then this test."""
    assert MUTILATION_BLOCK == 0
    assert MUTILATION_TARGET_KEY == "blocks.0.mlp.net.3.weight"
    encoder = VideoMAEEncoder(_cfg(depth=MUTILATION_BLOCK + 1))
    assert MUTILATION_TARGET_KEY in dict(encoder.named_parameters())


def _trained_from(cfg, seed: int, *, init_seed: int) -> DiagBundle:
    """A run initialised from `seed`, lightly trained, declaring `init_seed`."""
    init = random_init_bundle(cfg, seed=seed)
    encoder = copy.deepcopy(init.encoder)
    _perturb_every_block(encoder, sigma=0.002, seed=1)
    return DiagBundle(
        encoder=encoder, model=None, run_id="r", round=None, init_seed=init_seed
    )


def test_delta_from_init_accepts_the_rebuilt_init_of_the_declared_seed():
    cfg = _cfg()
    trained = _trained_from(cfg, seed=7, init_seed=7)
    result = delta_from_init(trained, random_init_bundle(cfg, seed=7))
    assert all(v > 0.0 for v in result.values())


def test_delta_from_init_refuses_a_declared_seed_the_weights_contradict():
    """A mistyped seed rebuilds an independent init, about sqrt(2) away in
    relative terms: matching seeds alone must not be taken on trust."""
    cfg = _cfg()
    trained = _trained_from(cfg, seed=8, init_seed=7)
    with pytest.raises(ValueError, match="init"):
        delta_from_init(trained, random_init_bundle(cfg, seed=7))


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


def test_check_skips_only_delta_from_init_on_a_non_init_reference():
    """A random-init reference at round >= 2 is valid for residual_branch_norm
    but not for delta_from_init. check must skip delta_from_init alone, not
    raise and lose all six reports."""
    cfg = _cfg()
    bundle = _bundle(cfg, run_id="r", round=5)
    ref_bundle = _bundle(cfg, run_id="random_init", round=None)

    reports = check(bundle, {"ref_bundle": ref_bundle})

    assert reports.keys() == check(bundle, {}).keys()
    assert reports["delta_from_init"].metrics == {}
    assert reports["delta_from_init"].meta["skipped"].startswith("r: delta_from_init")
    assert len(reports["residual_branch_norm"].metrics) == 2 * cfg.depth


# --- cka ---------------------------------------------------------------


def test_cka_of_matrix_with_itself_is_one():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((50, 12))
    assert cka(x, x) == pytest.approx(1.0, abs=1e-6)


def test_cka_invariant_to_rotation_and_scaling():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((50, 12))
    y = rng.standard_normal((50, 12))
    q, _ = np.linalg.qr(rng.standard_normal((12, 12)))
    base = cka(x, y)
    assert cka(x @ q, y) == pytest.approx(base, rel=1e-4)
    assert cka(x * 3.0, y) == pytest.approx(base, rel=1e-4)


def test_cka_of_independent_gaussians_near_zero():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2000, 16))
    y = rng.standard_normal((2000, 16))
    assert cka(x, y) == pytest.approx(0.0, abs=0.1)


def test_cka_is_nan_on_zero_variance_input_not_zero():
    """A dead (constant) layer has no similarity structure to compare. 0.0
    would read as "maximally different", a strong false depth signal."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((50, 12))
    dead = np.ones((50, 12))
    assert cka(x, dead) == pytest.approx(float("nan"), nan_ok=True)
    assert cka(x, dead) != pytest.approx(0.0)


def test_cka_outlier_sensitivity_reproduced():
    """Davari et al.: a handful of huge-norm outlier tokens should move
    trimmed CKA less than untrimmed CKA, relative to the outlier-free
    baseline."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((60, 16))
    y = rng.standard_normal((60, 16))
    baseline = cka(x, y)

    x_outlier = x.copy()
    x_outlier[:5] *= 1000.0

    untrimmed = cka(x_outlier, y, trim_outlier_tokens=0)
    trimmed = cka(x_outlier, y, trim_outlier_tokens=5)

    assert abs(trimmed - baseline) < abs(untrimmed - baseline)


def _cfg(*, depth: int | None = None, layer_scale_init: float | None = None):
    cfg = OmegaConf.load(MODEL_CFG_PATH)
    cfg.depth = depth if depth is not None else MUTILATION_BLOCK + 2
    cfg.embed_dim = 32
    cfg.num_heads = 4
    cfg.num_frames = 4
    cfg.image_size = 32
    if layer_scale_init is not None:
        cfg.layer_scale_init = layer_scale_init
    return cfg


def _bundle_(cfg, run_id: str = "r", round: int | None = None) -> DiagBundle:
    encoder = VideoMAEEncoder(cfg).eval()
    return DiagBundle(encoder=encoder, model=None, run_id=run_id, round=round)


def test_cka_matrix_symmetric_unit_diagonal():
    cfg = _cfg(depth=3)
    bundle = _bundle_(cfg)
    pixel_values = torch.randn(2, 3, 4, 32, 32)

    result = cka_matrix(bundle, pixel_values)

    assert set(result.keys()) == {"full", "trimmed", "diff"}
    for key in ("full", "trimmed"):
        matrix = result[key]
        assert np.allclose(matrix, matrix.T, atol=1e-5)
        assert np.allclose(np.diag(matrix), 1.0, atol=1e-5)
    assert np.allclose(result["diff"], result["full"] - result["trimmed"], atol=1e-6)


def test_cka_between_depth_matched():
    cfg = _cfg(depth=3)
    bundle_a = _bundle_(cfg, run_id="a")
    bundle_b = _bundle_(cfg, run_id="b")
    pixel_values = torch.randn(2, 3, 4, 32, 32)

    result = cka_between(bundle_a, bundle_b, pixel_values)

    assert result.shape == (cfg.depth + 1,)
    assert np.all(np.isfinite(result))


# --- decode_from_depth ---------------------------------------------------


def _decode_bundle_and_input():
    cfg = _cfg(depth=3)
    bundle = _bundle_(cfg)
    pixel_values = torch.randn(2, int(cfg.in_channels), 4, 32, 32)
    return bundle, pixel_values, cfg


def test_fit_depth_head_signature_forces_a_keep_split():
    from stern.evaluation.diagnostics.depth import _fit_depth_head

    bundle, pixel_values, cfg = _decode_bundle_and_input()
    grid = bundle.encoder.token_grid
    keep, mask, _ = temporal_split(grid, pixel_values.device, keep_frac=0.5)

    import inspect

    sig = inspect.signature(_fit_depth_head)
    assert "keep" in sig.parameters
    assert sig.parameters["keep"].kind == inspect.Parameter.KEYWORD_ONLY

    keep_set = set(keep.tolist())
    mask_set = set(mask.tolist())
    assert keep_set.isdisjoint(mask_set)

    head = _fit_depth_head(bundle, pixel_values, 1, keep=keep, steps=5, seed=0)
    assert isinstance(head, torch.nn.Linear)


def test_decode_from_depth_fits_separate_head_per_depth():
    from stern.evaluation.diagnostics.depth import _fit_depth_head, decode_from_depth

    bundle, pixel_values, cfg = _decode_bundle_and_input()

    result = decode_from_depth(bundle, pixel_values, [1, 2], n_seeds=3)

    grid = bundle.encoder.token_grid
    keep, _mask, _ = temporal_split(grid, pixel_values.device, keep_frac=0.5)
    head1 = _fit_depth_head(bundle, pixel_values, 1, keep=keep, steps=5, seed=0)
    head2 = _fit_depth_head(bundle, pixel_values, 2, keep=keep, steps=5, seed=0)
    assert not torch.allclose(head1.weight, head2.weight)
    assert set(result.keys()) == {1, 2}


def test_decode_from_depth_returns_replicated_metric_value():
    from stern.evaluation.diagnostics.depth import decode_from_depth

    bundle, pixel_values, cfg = _decode_bundle_and_input()

    result = decode_from_depth(bundle, pixel_values, [1], n_seeds=3)

    value = result[1]
    assert value.n == 3
    assert len(set(value.seeds)) == 3
    assert value.deterministic is False
    assert np.isfinite(value.mean)


# --- check() wiring for cka_matrix / decode_from_depth ------------------


def test_check_wires_cka_matrix_and_decode_from_depth_when_input_present():
    cfg = _cfg(depth=3)
    bundle = _bundle_(cfg)
    pixel_values = torch.randn(2, int(cfg.in_channels), 4, 32, 32)

    reports = check(bundle, {"real": pixel_values})

    assert "skipped" not in reports["cka_matrix"].meta
    assert set(reports["cka_matrix"].arrays.keys()) == {"full", "trimmed", "diff"}

    assert "skipped" not in reports["decode_from_depth"].meta
    assert reports["decode_from_depth"].metrics
    for value in reports["decode_from_depth"].metrics.values():
        assert value.n == 3


def test_check_skips_cka_matrix_and_decode_from_depth_without_input():
    bundle = _bundle_(_cfg())

    reports = check(bundle, {})

    assert reports["cka_matrix"].arrays == {}
    assert reports["cka_matrix"].meta["skipped"] == "no input tensor supplied"
    assert reports["decode_from_depth"].metrics == {}
    assert reports["decode_from_depth"].meta["skipped"] == "no input tensor supplied"
