"""Tests for the E2b depth-collapse ablation knobs.

Three arms share one control: every knob added here defaults to the committed
behaviour, so arm A must be *bit-comparable* to the recipe already running.
That bit-equality is the first thing each group asserts -- a knob that quietly
perturbs the control makes the whole 3-pool comparison meaningless.

  arm A  control                      (all knobs at their defaults)
  arm B  model.visible_loss_weight    (+ distance weighting)
  arm C  model.deep_supervision_layers

See docs/research/2026-08-27-depth-collapse-3pool-prompt.md.
"""

import pytest
import torch
from omegaconf import DictConfig

from stern.models.videomae.mae import (
    VideoMAEForPreTraining,
    visible_isolation_weights,
)
from stern.models.videomae.masking import RandomTubeMasking

TOKEN_GRID = (2, 2, 2)  # T', H', W' for the tiny config below -> N = 8 tokens
# The isolation weighting needs more than two visible columns to be anything but
# uniform, so the arm-B weighted tests use image_size=64 -> a 4x4 spatial grid.
LARGE_GRID = (2, 4, 4)


def _cfg(**overrides) -> DictConfig:
    base = {
        "num_frames": 4,
        "image_size": 32,
        "in_channels": 3,
        "tubelet_size": 2,
        "patch_size": 16,
        "embed_dim": 16,
        "depth": 4,
        "num_heads": 2,
        "mlp_ratio": 2.0,
        "dropout": 0.0,
        "drop_path_rate": 0.0,
        "decoder_depth": 1,
        "decoder_embed_dim": 8,
        "decoder_num_heads": 2,
        "mask_ratio": 0.5,
    }
    base.update(overrides)
    return DictConfig(base)


def _batch_large(seed: int = 0):
    """As `_batch`, on the 4x4 spatial grid the isolation weighting needs."""
    torch.manual_seed(seed)
    pixel_values = torch.randn(2, 3, 4, 64, 64)
    keep_indices, mask_indices = RandomTubeMasking(mask_ratio=0.5)(
        pixel_values, LARGE_GRID, torch.device("cpu"), batch_size=2
    )
    return pixel_values, keep_indices, mask_indices


def _large_loss_for(cfg: DictConfig, *, seed: int = 0):
    torch.manual_seed(seed)
    model = VideoMAEForPreTraining(cfg)
    model.eval()
    pred, target, aux = model(*_batch_large(seed), return_aux=True)
    return model, model.compute_loss(pred, target, aux=aux)


def _batch(seed: int = 0):
    """A fixed (pixel_values, keep_indices, mask_indices) triple.

    Seeded so two models built from the same seed are bit-comparable: the
    control-equality assertions below depend on identical weights *and*
    identical masks.
    """
    torch.manual_seed(seed)
    pixel_values = torch.randn(2, 3, 4, 32, 32)
    masker = RandomTubeMasking(mask_ratio=0.5)
    keep_indices, mask_indices = masker(
        pixel_values, TOKEN_GRID, torch.device("cpu"), batch_size=2
    )
    return pixel_values, keep_indices, mask_indices


def _loss_for(cfg: DictConfig, *, seed: int = 0, detach_main: bool = False):
    torch.manual_seed(seed)
    model = VideoMAEForPreTraining(cfg)
    model.eval()
    pixel_values, keep_indices, mask_indices = _batch(seed)
    pred, target, aux = model(pixel_values, keep_indices, mask_indices, return_aux=True)
    if detach_main:
        pred = pred.detach()
    return model, model.compute_loss(pred, target, aux=aux)


# --- arm A: the control must not move ------------------------------------


def test_defaults_reproduce_the_two_tuple_forward():
    """The committed call site is `pred, target = model(...)`. Both training
    paths use it verbatim, so the default return arity is load-bearing."""
    model = VideoMAEForPreTraining(_cfg())
    model.eval()
    pixel_values, keep_indices, mask_indices = _batch()

    with torch.no_grad():
        out = model(pixel_values, keep_indices, mask_indices)

    assert isinstance(out, tuple) and len(out) == 2
    pred, target = out
    assert pred.shape == target.shape


def test_control_loss_is_the_pinned_value():
    """The anchor for every equality assertion below. Pinned rather than merely
    self-consistent: if a refactor shifted the control *and* the arms together,
    relative comparisons would all still pass and the ablation would be
    measuring a different model than the one on the cluster.

    Regenerate deliberately, never to make a red test green -- a change here
    means arm A is no longer bit-comparable to the committed recipe.
    """
    _, control = _loss_for(_cfg())

    assert control.item() == pytest.approx(1.3716167212, abs=1e-9)


def test_visible_loss_weight_zero_is_bit_equal_to_the_control():
    """Not merely close: arm B at w=0 must be the *same number* as arm A, or
    the arms differ by a floating-point accident rather than by the knob."""
    _, control = _loss_for(_cfg())
    _, armb_off = _loss_for(_cfg(visible_loss_weight=0.0))

    assert torch.equal(control, armb_off)
    assert armb_off.item() == pytest.approx(1.3716167212, abs=1e-9)


def test_empty_deep_supervision_is_bit_equal_to_the_control():
    _, control = _loss_for(_cfg())
    _, armc_off = _loss_for(_cfg(deep_supervision_layers=[]))

    assert torch.equal(control, armc_off)
    assert armc_off.item() == pytest.approx(1.3716167212, abs=1e-9)


def test_defaults_produce_no_aux_terms():
    model = VideoMAEForPreTraining(_cfg())
    model.eval()
    pixel_values, keep_indices, mask_indices = _batch()

    with torch.no_grad():
        _, _, aux = model(pixel_values, keep_indices, mask_indices, return_aux=True)

    assert aux.visible_pred is None
    assert aux.visible_target is None
    assert aux.deep_preds == []


# --- arm B: the visible-patch reconstruction term -------------------------


def test_visible_loss_weight_adds_the_expected_amount():
    """Pins the size of the term, not just that it moved. w=0.5 on this fixture
    adds 0.6823 to the control's 1.3716 -- so the visible MSE is ~1.365, the
    same order as the masked MSE. That is what makes the weight a real dial
    rather than a rounding error."""
    _, control = _loss_for(_cfg())
    _, armb = _loss_for(_cfg(visible_loss_weight=0.5))

    assert armb.item() == pytest.approx(2.0538916588, abs=1e-9)
    assert (armb - control).item() == pytest.approx(0.6822749376, abs=1e-8)


def test_visible_term_covers_exactly_the_kept_tokens():
    cfg = _cfg(visible_loss_weight=0.5)
    model = VideoMAEForPreTraining(cfg)
    model.eval()
    pixel_values, keep_indices, mask_indices = _batch()

    with torch.no_grad():
        _, _, aux = model(pixel_values, keep_indices, mask_indices, return_aux=True)

    assert aux.visible_pred.shape[1] == keep_indices.shape[1]
    assert aux.visible_target.shape == aux.visible_pred.shape


def test_visible_loss_gradient_reaches_encoder_block_zero():
    """The point of arm B: the visible term must train the *encoder*, not just
    the decoder. If it stopped at the decoder it would not touch the residual
    stream the ablation is about."""
    cfg = _cfg(visible_loss_weight=0.5)
    model, loss = _loss_for(cfg, detach_main=True)
    loss.backward()

    grad = model.encoder.blocks[0].mlp.net[0].weight.grad
    assert grad is not None
    assert torch.any(grad != 0)


def test_isolation_weighting_is_non_degenerate_at_the_production_mask_ratio():
    """**The test that earns arm B's weighted variant its pool.** The weighting
    has to discriminate at the ratio we actually train at, or the arm is a
    duplicate of plain arm B and burns 120 rounds proving nothing.

    Measured on the real (8, 14, 14) grid at ratio 0.9 under seed 0, distance to
    the nearest visible token outside the tube spans nine values:
        1.00:36%  1.41:18%  2.00:10%  2.24:18%  2.83:9%
        3.16:2%   3.61:4%   4.00:1%   4.47:2%

    Contrast the definition this replaced -- distance to the nearest *masked*
    token -- which is a single value (1.00, 100%) at 0.9 *and* at 0.75, and only
    spreads at 0.5. See visible_isolation_weights' docstring.

    The draw is seeded so the values below can be pinned exactly; the three
    assertions after them hold for every seed, and are the ones that carry the
    claim (checked at seeds 1-3: 10-11 distinct weights, min always 1-exp(-1)).
    """
    grid = (8, 14, 14)
    torch.manual_seed(0)
    keep, _ = RandomTubeMasking(mask_ratio=0.9)(
        torch.zeros(4, 3, 2, 2, 2), grid, torch.device("cpu"), 4
    )

    w = visible_isolation_weights(keep, grid, torch.device("cpu"))

    assert w.shape == (4, 160)
    assert torch.isfinite(w).all()
    assert len(torch.unique(torch.round(w * 1e4))) == 9
    assert float(w.min()) == pytest.approx(0.6321205497, abs=1e-9)
    assert float(w.max()) == pytest.approx(0.9885771275, abs=1e-9)

    # Seed-independent. The minimum is exactly 1-exp(-1): at ratio 0.9 some
    # visible token always has another visible token in an adjacent column, and
    # a weight below that would mean two visible tokens shared a column, which
    # tube masking forbids.
    assert float(w.min()) == pytest.approx(1.0 - float(torch.exp(torch.tensor(-1.0))))
    assert len(torch.unique(torch.round(w * 1e4))) >= 4, (
        "isolation weights collapsed to near-uniform at ratio 0.9 -- arm B's "
        "weighted variant is a duplicate of plain arm B and must not get a pool"
    )
    # Bounded in (0, 1), so the normalised weighted mean stays on the same
    # scale as the unweighted one and W keeps its meaning across both variants.
    assert 0.0 < float(w.min()) and float(w.max()) < 1.0


def test_isolation_weight_rises_with_distance_to_the_nearest_visible_token():
    """w_i = 1 - exp(-d_i), d_i = spatial distance to the nearest visible token
    outside token i's tube. An isolated patch -- nothing to copy from -- must
    outweigh one sitting next to a visible neighbour."""
    # Grid (1, 1, 8): an 8-token row, all one tubelet. Keep 0, 1 and 5:
    # token 0 and token 1 are adjacent (d=1); token 5 is four cells away (d=4).
    keep = torch.tensor([[0, 1, 5]])
    w = visible_isolation_weights(keep, (1, 1, 8), torch.device("cpu"))

    assert w.shape == (1, 3)
    assert w[0, 2] > w[0, 0] == w[0, 1]
    assert w[0, 0].item() == pytest.approx(1.0 - float(torch.exp(torch.tensor(-1.0))))
    assert w[0, 2].item() == pytest.approx(1.0 - float(torch.exp(torch.tensor(-4.0))))


def test_isolation_weighting_ignores_the_temporal_axis():
    """The bug that made the first definition a no-op at every ratio. Tube
    masking keeps a whole (h, w) column across all of T', so a token's own tube
    is always present at d=1 along T'. A 3D metric therefore reports 1.0 for
    every token forever. Distances must be computed in (H', W') only."""
    # Grid (4, 1, 2): one spatial row of 2, four temporal positions. Keep every
    # copy of column 0 and nothing else -- so in 3D each token has a tube-mate
    # at d=1, but spatially there is no other visible column at all.
    keep = torch.tensor([[0, 2, 4, 6]])
    w = visible_isolation_weights(keep, (4, 1, 2), torch.device("cpu"))

    assert torch.allclose(w, torch.ones_like(w)), (
        "temporal copies were counted as spatial neighbours; the weighting "
        "collapses to a constant exactly as the masked-distance version did"
    )


def test_isolation_weighting_changes_the_visible_term():
    """Non-degeneracy where it counts: in the loss, not just in the weights."""
    _, plain = _large_loss_for(_cfg(image_size=64, visible_loss_weight=0.5))
    _, weighted = _large_loss_for(
        _cfg(
            image_size=64,
            visible_loss_weight=0.5,
            visible_loss_isolation_weighting=True,
        )
    )

    assert not torch.equal(plain, weighted)


def test_isolation_weighting_is_inert_without_a_visible_term():
    """The weighting knob only scales the visible term; on its own it must not
    perturb the control, or arm A and arm C stop being comparable."""
    _, control = _loss_for(_cfg())
    _, weighting_only = _loss_for(_cfg(visible_loss_isolation_weighting=True))

    assert torch.equal(control, weighting_only)


# --- arm C: deep supervision ----------------------------------------------


def test_deep_supervision_adds_a_term_of_comparable_size():
    """The auxiliary term is averaged over layers and added at unit weight, so
    supervising one layer roughly doubles the loss (1.3716 -> 2.7434). Pinned
    so a change to that weighting convention has to be deliberate: it sets how
    hard arm C pulls against the main objective."""
    _, control = _loss_for(_cfg())
    _, armc = _loss_for(_cfg(deep_supervision_layers=[2]))

    assert armc.item() == pytest.approx(2.7433607578, abs=1e-9)
    assert armc.item() / control.item() == pytest.approx(2.0, abs=0.01)


def test_deep_supervision_gradient_reaches_the_supervised_block_only():
    """The whole point of arm C, and the test the prompt calls out as the one
    that must not be tautological: with the final-layer loss detached, the
    supervised block must still receive gradient -- and the blocks *after* it
    must receive none, proving the signal came from the auxiliary head and not
    by leaking through the main path.

    depth=4, supervising block index 2 (1-based layer 2 -> blocks[1]).
    """
    cfg = _cfg(deep_supervision_layers=[2])
    model, loss = _loss_for(cfg, detach_main=True)
    loss.backward()

    supervised = model.encoder.blocks[1].mlp.net[0].weight.grad
    downstream = model.encoder.blocks[3].mlp.net[0].weight.grad

    assert supervised is not None and torch.any(supervised != 0), (
        "supervised block got no gradient; the auxiliary head is not wired "
        "to the encoder"
    )
    assert downstream is None or torch.all(downstream == 0), (
        "a block after the supervised layer received gradient with the main "
        "loss detached; the deep term is leaking through the final path"
    )


def test_deep_supervision_emits_one_prediction_per_requested_layer():
    cfg = _cfg(deep_supervision_layers=[1, 3])
    model = VideoMAEForPreTraining(cfg)
    model.eval()
    pixel_values, keep_indices, mask_indices = _batch()

    with torch.no_grad():
        pred, _, aux = model(pixel_values, keep_indices, mask_indices, return_aux=True)

    assert len(aux.deep_preds) == 2
    for deep_pred in aux.deep_preds:
        assert deep_pred.shape == pred.shape


def test_deep_supervision_averages_over_layers_rather_than_summing():
    """`[1, 3]` on a 4-block encoder. The auxiliary term is the *mean* of the
    per-layer terms, so its magnitude does not grow with how many layers are
    supervised -- under the production `[4, 8, 12]` a sum would silently weigh
    the auxiliary objective three times as hard as the main one.

    Pinned against both conventions: `main + mean = 2.7434` but
    `main + sum = 4.1151`, so a `.mean()` -> `.sum()` regression moves the loss
    by 50% and cannot pass. The single-layer test above cannot see this
    difference, because there sum and mean are the same number.
    """
    cfg = _cfg(deep_supervision_layers=[1, 3])
    model, loss = _loss_for(cfg)

    assert loss.item() == pytest.approx(2.7433662415, abs=1e-9)
    assert loss.item() != pytest.approx(4.1151157618, abs=1e-3)


def test_deep_supervision_accepts_the_last_block():
    """`layer == depth` is the boundary the production config sits on: the
    prompt's `[4, 8, 12]` supervises block 12 of a 12-block encoder. An
    off-by-one in the validator (`layer < depth`) would reject exactly that
    config at construction time, on the cluster, after the pool is seeded.
    """
    cfg = _cfg(depth=4, deep_supervision_layers=[4])
    model = VideoMAEForPreTraining(cfg)
    model.eval()
    pixel_values, keep_indices, mask_indices = _batch()

    with torch.no_grad():
        pred, _, aux = model(pixel_values, keep_indices, mask_indices, return_aux=True)

    assert len(aux.deep_preds) == 1
    assert aux.deep_preds[0].shape == pred.shape


def test_deep_supervision_rejects_an_out_of_range_layer():
    with pytest.raises(ValueError, match="deep_supervision_layers"):
        VideoMAEForPreTraining(_cfg(depth=4, deep_supervision_layers=[9]))


def test_deep_supervision_rejects_a_non_positive_layer():
    """Layers are 1-based to match the prompt's [4, 8, 12]; 0 would silently
    mean 'the last block' under Python indexing."""
    with pytest.raises(ValueError, match="deep_supervision_layers"):
        VideoMAEForPreTraining(_cfg(depth=4, deep_supervision_layers=[0]))
