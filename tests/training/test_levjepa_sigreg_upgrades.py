"""Opt-in SIGReg upgrades: fp32 under autocast, seeded/orthogonal directions,
moment term, pooled SIGReg across micro-batches (train.sigreg_pool)."""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from stern.models.levjepa.losses import (
    SigregOptions,
    levjepa_loss,
    sample_directions,
    sigreg_from_sums,
    sigreg_loss,
    sigreg_sums,
    view_directions,
)
from stern.training import mesh_round
from stern.training.pretrain_levjepa import _levjepa_embed

LAM, M = 0.3, 64  # lam large so SIGReg's gradient is not swamped in the checks


def _reference_levjepa_loss(z_global, z_locals, lam=0.02, num_directions=1024):
    """Verbatim copy of the pre-upgrade losses.py (production 2026-10-06):
    the default path must stay bit-identical to it."""

    def ref_sigreg(z, num_directions=1024, num_knots=17, t_max=3.0):
        B, K = z.shape
        directions = torch.randn(K, num_directions, device=z.device, dtype=z.dtype)
        directions = directions / directions.norm(dim=0, keepdim=True)
        projections = z @ directions
        t = torch.linspace(0.0, t_max, num_knots, device=z.device, dtype=z.dtype)
        phase = projections.unsqueeze(-1) * t.view(1, 1, -1)
        real = torch.cos(phase).mean(dim=0)
        imag = torch.sin(phase).mean(dim=0)
        gaussian_char = torch.exp(-0.5 * t.pow(2))
        sq_error = (real - gaussian_char.unsqueeze(0)).pow(2) + imag.pow(2)
        dt = t_max / (num_knots - 1)
        weights = torch.full_like(t, 2 * dt)
        weights[[0, -1]] = dt
        return (sq_error @ (weights * gaussian_char) * B).mean()

    inv = (z_locals - z_global.unsqueeze(1)).pow(2).mean()
    views = torch.cat([z_global.unsqueeze(1), z_locals], dim=1)
    sig = torch.stack(
        [
            ref_sigreg(views[:, v], num_directions=num_directions)
            for v in range(views.shape[1])
        ]
    ).mean()
    return (1 - lam) * inv + lam * sig


class _StubEncoder(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.g = nn.Linear(3 * 2 * 4 * 4, 16)
        self.l = nn.Linear(3 * 2 * 2 * 2, 16)
        self.drop = nn.Dropout(p)

    def forward_cls(self, x, crop_box=None):
        lin = self.l if crop_box is not None else self.g
        return self.drop(lin(x.flatten(1)))


class _Stub(nn.Module):
    """encoder + BN-free projector: BatchNorm per micro-batch is the one known
    non-exactness of pooling, so the exactness tests leave it out."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.encoder = _StubEncoder(p)
        self.projector = nn.Linear(16, 8)


def _batch(B, V=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(B, 3, 2, 4, 4, generator=g),
        torch.randn(B, V, 3, 2, 2, 2, generator=g),
        torch.rand(B, V, 4, generator=g),
    )


def _split(batch, k):
    return [tuple(t.chunk(k)[i] for t in batch) for i in range(k)]


def _grads(model):
    return [p.grad.clone() for p in model.parameters()]


def _nullctx():
    return torch.autocast("cpu", enabled=False)


# (iii) default path is bit-identical to the pre-upgrade code ----------------


def test_default_levjepa_loss_is_bit_identical_to_pre_upgrade_code():
    zg = torch.randn(16, 8, requires_grad=True)
    zl = torch.randn(16, 3, 8, requires_grad=True)
    torch.manual_seed(5)
    ref = _reference_levjepa_loss(zg, zl, lam=LAM, num_directions=M)
    ref_g = torch.autograd.grad(ref, [zg, zl])
    torch.manual_seed(5)
    new, _ = levjepa_loss(zg, zl, lam=LAM, num_directions=M)
    new_g = torch.autograd.grad(new, [zg, zl])
    assert torch.equal(ref, new)
    assert all(torch.equal(a, b) for a, b in zip(ref_g, new_g))


def test_absent_config_keys_give_default_options():
    cfg = OmegaConf.create({"model": {}, "train": {}})
    assert mesh_round._sigreg_options(cfg) == SigregOptions()


# (ii) bf16 autocast ---------------------------------------------------------


def test_loss_under_bf16_autocast_is_computed_in_fp32():
    """Autocast would otherwise run the SIGReg matmul in bf16; the loss must
    equal the fp32 computation on the same (bf16-rounded) embeddings."""
    torch.manual_seed(0)
    zg = torch.randn(64, 8).bfloat16()
    zl = torch.randn(64, 3, 8).bfloat16()
    opts = SigregOptions(seed=1)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        amp_loss, amp_c = levjepa_loss(zg, zl, lam=LAM, num_directions=M, opts=opts)
    fp32_loss, fp32_c = levjepa_loss(
        zg.float(), zl.float(), lam=LAM, num_directions=M, opts=opts
    )
    assert amp_loss.dtype == amp_c["sigreg"].dtype == torch.float32
    torch.testing.assert_close(amp_loss, fp32_loss, rtol=1e-6, atol=1e-6)


# seeded / orthogonal directions --------------------------------------------


def test_seeded_directions_leave_global_rng_untouched_and_repeat():
    zg, zl = torch.randn(16, 8), torch.randn(16, 3, 8)
    opts = SigregOptions(seed=11)
    state = torch.get_rng_state()
    a, _ = levjepa_loss(zg, zl, num_directions=M, opts=opts, key=(2, 3))
    assert torch.equal(torch.get_rng_state(), state)
    b, _ = levjepa_loss(zg, zl, num_directions=M, opts=opts, key=(2, 3))
    c, _ = levjepa_loss(zg, zl, num_directions=M, opts=opts, key=(2, 4))
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_orthogonal_directions_are_orthonormal_within_each_block():
    opts = SigregOptions(seed=0, orthogonal=True)
    d = view_directions(opts, (0,), 1, 8, 20, torch.device("cpu"))[0]  # blocks 8,8,4
    assert d.shape == (8, 20)
    for lo, hi in [(0, 8), (8, 16), (16, 20)]:
        blk = d[:, lo:hi]
        torch.testing.assert_close(blk.T @ blk, torch.eye(hi - lo), atol=1e-5, rtol=0)


# moment term ----------------------------------------------------------------


def _loss_at_scale(scale, w, dirs, base):
    return sigreg_loss(base * scale, num_directions=M, directions=dirs, moment_weight=w).item()


def test_epps_pulley_goes_flat_when_embeddings_explode_and_moment_term_does_not():
    """Without the moment term the loss is the same at scale 20 and 200 (no
    expected restoring force); with it, the loss still rises with scale --
    but only logarithmically, not as the scale^4 a raw-variance term gives."""
    torch.manual_seed(0)
    base = torch.randn(256, 8)
    dirs = view_directions(SigregOptions(seed=0), (), 1, 8, M, torch.device("cpu"))[0]
    ep = [_loss_at_scale(s, 0.0, dirs, base) for s in (2, 20, 200)]
    assert ep[1] > 2 * ep[0]  # still informative at moderate scale
    assert abs(ep[2] - ep[1]) < 0.05 * ep[1]  # flat: no signal left
    mom = [_loss_at_scale(s, 1.0, dirs, base) - e for s, e in zip((2, 20, 200), ep)]
    assert mom[2] > mom[1] + 100  # still pulls back at 200x
    assert mom[2] < 10 * mom[1]  # but bounded growth (s^4 would be 1e4x)


def test_moment_term_is_small_at_the_gaussian_optimum():
    torch.manual_seed(0)
    dirs = view_directions(SigregOptions(seed=0), (), 1, 8, M, torch.device("cpu"))[0]
    z = torch.randn(4096, 8)
    extra = _loss_at_scale(1.0, 1.0, dirs, z) - _loss_at_scale(1.0, 0.0, dirs, z)
    assert 0.0 < extra < 5.0  # null floor ~1.5


def test_sigreg_from_sums_matches_sigreg_loss_on_one_batch():
    z = torch.randn(32, 8)
    dirs = view_directions(SigregOptions(seed=4), (), 1, 8, M, torch.device("cpu"))[0]
    for w in (0.0, 0.7):
        direct = sigreg_loss(z, num_directions=M, directions=dirs, moment_weight=w)
        pooled = sigreg_from_sums(sigreg_sums(z, dirs), 32, moment_weight=w)
        torch.testing.assert_close(direct, pooled, rtol=1e-5, atol=1e-6)


# (i) pooled SIGReg == one big batch ------------------------------------------


@pytest.mark.parametrize(
    "opts",
    [
        SigregOptions(seed=3, pool=True),
        SigregOptions(seed=3, orthogonal=True, pool=True),
        SigregOptions(seed=3, moment_weight=0.5, pool=True),
    ],
)
def test_pooled_window_k4_m8_matches_single_batch_32(opts):
    torch.manual_seed(0)
    model = _Stub().double()
    batch = tuple(t.double() for t in _batch(32))
    dev = torch.device("cpu")

    zg, zl = _levjepa_embed(batch, model.encoder, model.projector, dev)
    ref, ref_c = levjepa_loss(zg, zl, lam=LAM, num_directions=M, opts=opts, key=(0, 7))
    ref.backward()
    ref_grads = _grads(model)
    model.zero_grad()

    loss, comps = mesh_round._levjepa_pooled_window(
        _split(batch, 4),
        model.encoder,
        model.projector,
        dev,
        lam=LAM,
        num_directions=M,
        opts=opts,
        key=(0, 7),
        accum_steps=4, autocast_ctx=_nullctx,
    )
    torch.testing.assert_close(loss.double(), ref.detach().double(), rtol=1e-4, atol=1e-8)
    torch.testing.assert_close(
        comps["sigreg"].double(), ref_c["sigreg"].detach().double(), rtol=1e-4, atol=1e-8
    )
    for a, b in zip(_grads(model), ref_grads):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-7)


def test_pooled_window_replays_rng_so_dropout_masks_match_both_passes():
    """Reference: the same 4 micro-batches forwarded with grad in order from
    the same RNG state, concatenated, one loss. Pass 2 only reproduces those
    dropout masks if pass 1's RNG consumption is rewound per micro-batch."""
    opts = SigregOptions(seed=3, pool=True)
    model = _Stub(p=0.5).double()
    chunks = _split(tuple(t.double() for t in _batch(32)), 4)
    dev = torch.device("cpu")

    torch.manual_seed(9)
    zs = [_levjepa_embed(c, model.encoder, model.projector, dev) for c in chunks]
    zg = torch.cat([z[0] for z in zs])
    zl = torch.cat([z[1] for z in zs])
    ref, _ = levjepa_loss(zg, zl, lam=LAM, num_directions=M, opts=opts, key=(1,))
    ref.backward()
    ref_grads = _grads(model)
    model.zero_grad()

    torch.manual_seed(9)
    loss, _ = mesh_round._levjepa_pooled_window(
        chunks,
        model.encoder,
        model.projector,
        dev,
        lam=LAM,
        num_directions=M,
        opts=opts,
        key=(1,),
        accum_steps=4, autocast_ctx=_nullctx,
    )
    torch.testing.assert_close(loss.double(), ref.detach().double(), rtol=1e-4, atol=1e-8)
    for a, b in zip(_grads(model), ref_grads):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-7)


def test_pooled_training_step_matches_unpooled_big_batch_step():
    """End to end through _train_epoch_levjepa: accum 4 x 8 pooled == one
    optimizer step on the 32-batch (same seeded directions, SGD)."""
    batch = _batch(32)

    def run(loader, accum, pool):
        torch.manual_seed(0)
        model = _Stub()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        stats = mesh_round._train_epoch_levjepa(
            model,
            loader,
            opt,
            torch.device("cpu"),
            accum_steps=accum,
            sigreg_lambda=LAM,
            sigreg_directions=M,
            sigreg_opts=SigregOptions(seed=2, pool=pool),
            sigreg_key=(5,),
        )
        return stats, [p.detach().clone() for p in model.parameters()]

    big_stats, big = run([batch], 1, False)
    pooled_stats, pooled = run(_split(batch, 4), 4, True)
    assert pooled_stats["samples"] == big_stats["samples"] == 32
    assert pooled_stats["nan_steps"] == 0
    assert pooled_stats["loss"] == pytest.approx(big_stats["loss"], rel=1e-4)
    for a, b in zip(pooled, big):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)


# audit follow-ups -------------------------------------------------------------


def test_config_keys_reach_sigreg_options():
    cfg = OmegaConf.create(
        {
            "model": {"sigreg_seed": 7, "sigreg_orthogonal": True, "sigreg_moment_weight": 0.5},
            "train": {"sigreg_pool": True},
        }
    )
    assert mesh_round._sigreg_options(cfg) == SigregOptions(7, True, 0.5, True)


def test_orthogonal_directions_carry_the_haar_sign_fix():
    """q * sign(diag(r)) makes diag(q^T g) positive for the Gaussian g it came
    from; plain QR leaves those signs to the LAPACK convention."""
    d = sample_directions(8, 8, torch.device("cpu"),
                          generator=torch.Generator().manual_seed(3), orthogonal=True)
    g = torch.randn(8, 8, generator=torch.Generator().manual_seed(3))
    assert (torch.diagonal(d.T @ g) > 0).all()


def test_unseeded_pool_shares_one_global_rng_direction_set_across_micro_batches():
    """Without a seed the pooled window draws one direction set per view from
    the global RNG (after the first micro-batch's forward), not one per
    micro-batch."""
    model = _Stub().double()
    batch = tuple(t.double() for t in _batch(32))
    dev = torch.device("cpu")
    zg, zl = _levjepa_embed(batch, model.encoder, model.projector, dev)
    torch.manual_seed(4)
    dirs = torch.stack([sample_directions(8, M, dev) for _ in range(4)])  # V+1 = 4
    views = torch.cat([zg.unsqueeze(1), zl], dim=1).float().detach()
    sums = mesh_round._view_sums(views, dirs)
    ref_sig = sigreg_from_sums(sums, 32)

    torch.manual_seed(4)  # stub draws no RNG in its forward
    _, comps = mesh_round._levjepa_pooled_window(
        _split(batch, 4), model.encoder, model.projector, dev,
        lam=LAM, num_directions=M, opts=SigregOptions(pool=True), key=(0,),
        accum_steps=4, autocast_ctx=_nullctx,
    )
    torch.testing.assert_close(comps["sigreg"], ref_sig, rtol=1e-5, atol=1e-6)


def test_non_finite_pooled_window_is_dropped_whole():
    batch = list(_batch(32))
    batch[0] = batch[0].clone()
    batch[0][20] = float("inf")  # poisons the 3rd of 4 micro-batches
    torch.manual_seed(0)
    model = _Stub()
    before = [p.detach().clone() for p in model.parameters()]
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    stats = mesh_round._train_epoch_levjepa(
        model, _split(tuple(batch), 4), opt, torch.device("cpu"), accum_steps=4,
        sigreg_lambda=LAM, sigreg_directions=M,
        sigreg_opts=SigregOptions(seed=1, pool=True),
    )
    assert stats["nan_steps"] == 1
    assert stats["samples"] == 0
    for a, b in zip(model.parameters(), before):
        assert torch.equal(a.detach(), b)


def test_short_last_window_weighs_invariance_like_the_unpooled_path():
    """5 micro-batches at accum 4: the trailing 1-batch window's invariance
    gradient is scaled 1/accum_steps, as unpooled accumulation scales it."""
    chunks = _split(_batch(40), 5)
    model = _Stub()
    loss, _ = mesh_round._levjepa_pooled_window(
        chunks[4:], model.encoder, model.projector, torch.device("cpu"),
        lam=0.0, num_directions=M, opts=SigregOptions(seed=1, pool=True), key=(1,),
        accum_steps=4, autocast_ctx=_nullctx,
    )
    pooled = _grads(model)
    model.zero_grad()
    zg, zl = _levjepa_embed(chunks[4], model.encoder, model.projector, torch.device("cpu"))
    (levjepa_loss(zg, zl, lam=0.0, num_directions=M)[0] / 4).backward()
    for a, b in zip(pooled, _grads(model)):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)


def test_pooled_window_left_over_at_loader_end_is_counted():
    """An iterable loader shorter than accum_steps never hits a boundary; its
    micro-batches must still count, as they do unpooled."""
    model = _Stub()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    def gen():
        yield from _split(_batch(24), 3)

    stats = mesh_round._train_epoch_levjepa(
        model, gen(), opt, torch.device("cpu"), accum_steps=4,
        sigreg_lambda=LAM, sigreg_directions=M,
        sigreg_opts=SigregOptions(seed=1, pool=True),
    )
    assert stats["samples"] == 24
    assert stats["steps"] == 3


def test_moment_term_stays_finite_on_a_collapsed_batch():
    dirs = view_directions(SigregOptions(seed=0), (), 1, 8, M, torch.device("cpu"))[0]
    z = torch.ones(64, 8, requires_grad=True)  # zero variance: hits the clamp
    loss = sigreg_loss(z, num_directions=M, directions=dirs, moment_weight=1.0)
    loss.backward()
    assert torch.isfinite(loss).item() is True
    assert torch.isfinite(z.grad).sum().item() == z.numel()


# train.levjepa_compile ----------------------------------------------------------


def _tiny_levjepa():
    from tests.training.test_mesh_round import _tiny_levjepa_model_cfg

    torch.manual_seed(0)
    return mesh_round._LeVJEPAPretrainModule(_tiny_levjepa_model_cfg()).eval()


def test_levjepa_compile_is_off_unless_its_own_key_is_set():
    """Absent key and train.compile alone both leave forward_cls the class
    method: deploying the patch changes nothing for a running run."""
    for train in ({}, {"compile": True}, {"levjepa_compile": False}):
        model = _tiny_levjepa()
        mesh_round._maybe_compile_levjepa_encoder(model, OmegaConf.create({"train": train}))
        assert "forward_cls" not in vars(model.encoder), train


def test_levjepa_compile_matches_eager_and_keeps_checkpoint_keys():
    model = _tiny_levjepa()
    keys = list(model.state_dict())
    x = torch.randn(2, 3, 16, 32, 32)
    with torch.no_grad():
        eager = model.encoder.forward_cls(x)
    # Behind torch.compile(model) too: train.compile=true wraps the outer module.
    wrapped = torch.compile(model)
    cfg = OmegaConf.create({"train": {"levjepa_compile": True}})
    mesh_round._maybe_compile_levjepa_encoder(wrapped, cfg)
    assert "forward_cls" in vars(model.encoder)
    with torch.no_grad():
        compiled = model.encoder.forward_cls(x)
    torch.testing.assert_close(compiled, eager, rtol=1e-4, atol=1e-5)
    assert list(model.state_dict()) == keys
    model.load_state_dict(_tiny_levjepa().state_dict())  # a master.pt reload
    assert "forward_cls" in vars(model.encoder)


def test_levjepa_compile_leaves_non_levjepa_models_alone():
    model = _Stub()
    mesh_round._maybe_compile_levjepa_encoder(
        model, OmegaConf.create({"train": {"levjepa_compile": True}})
    )
    assert "forward_cls" not in vars(model.encoder)
