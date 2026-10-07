from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch



def invariance_loss(z_global: torch.Tensor, z_locals: torch.Tensor) -> torch.Tensor:
    # z_global: (B, K); z_locals: (B, V, K). Compares each local view only
    # against the global anchor -- the global-vs-itself term is a guaranteed
    # zero and would otherwise dilute the mean by a V/(V+1) factor. Mean
    # over the K dims too (LeJEPA's `.square().mean()`): a sum would scale the
    # term by K against SIGReg and silently change what lam means.
    diff = z_locals - z_global.unsqueeze(1)
    return diff.pow(2).mean()


@dataclass(frozen=True)
class SigregOptions:
    """Opt-in SIGReg upgrades (model.sigreg_seed / sigreg_orthogonal /
    sigreg_moment_weight, train.sigreg_pool). The defaults reproduce the
    original estimator bit-for-bit, so a deploy changes no running run.

    seed: draw directions from a generator keyed on (seed, step, view)
        instead of the global RNG -- reproducible, leaves dropout/view RNG
        untouched, and lets pooled SIGReg reuse one direction set across the
        micro-batches of an optimizer step.
    orthogonal: directions are orthonormal within each block of K (QR of a
        Gaussian): same marginal distribution, lower-variance estimate.
    moment_weight: adds B * (log(1 + mean_p^2 / var_p) + log(var_p)^2 / 4)
        per direction. Epps-Pulley alone goes flat once projections are much
        wider than 1/t_max: cos/sin average to ~0, the loss plateaus and its
        expected gradient vanishes under the noise (measured: scale 20-200
        all give ~325) -- it loses its grip exactly when embeddings blow up.
        NOTE: this changes the objective.
    pool: SIGReg over all accum_steps micro-batches of an optimizer step
        (exact, two forward passes; see mesh_round._levjepa_pooled_window).
        The statistic's x B factor then uses the pooled B, so its weight vs
        invariance grows with the pool size at fixed lambda.
    """

    seed: int | None = None
    orthogonal: bool = False
    moment_weight: float = 0.0
    pool: bool = False


def _direction_generator(seed: int, key: tuple[int, ...]) -> torch.Generator:
    h = seed
    for k in key:
        h = (h * 1_000_003 + k + 1) % (2**63 - 1)
    return torch.Generator().manual_seed(h)


def sample_directions(
    dim: int,
    num_directions: int,
    device: torch.device,
    *,
    generator: torch.Generator | None = None,
    orthogonal: bool = False,
) -> torch.Tensor:
    """(dim, num_directions) fp32 unit directions, drawn on CPU (from the
    global RNG when generator is None) so a seed gives the same set on any
    device."""
    if not orthogonal:
        d = torch.randn(dim, num_directions, generator=generator)
        return (d / d.norm(dim=0, keepdim=True)).to(device)
    blocks, left = [], num_directions
    while left > 0:
        n = min(dim, left)
        g = torch.randn(dim, n, generator=generator)
        q, r = torch.linalg.qr(g)
        # Sign fix makes q Haar-distributed (QR's sign convention otherwise
        # biases it); columns are already unit norm.
        blocks.append(q * torch.sign(torch.diagonal(r)).unsqueeze(0))
        left -= n
    return torch.cat(blocks, dim=1).to(device)


@contextlib.contextmanager
def _no_tf32():
    # train.tf32 turns TF32 on globally; the SIGReg phase is proj * t with
    # t up to 3, so keep its matmul in true fp32. A no-op when TF32 is off.
    try:
        prev = torch.backends.cuda.matmul.allow_tf32
    except RuntimeError:  # newer fp32_precision API in use: leave it alone
        yield
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def _knots(num_knots: int, t_max: float, device, dtype) -> torch.Tensor:
    return torch.linspace(0.0, t_max, num_knots, device=device, dtype=dtype)


def _statistic(
    real: torch.Tensor,
    imag: torch.Tensor,
    t: torch.Tensor,
    B: int | torch.Tensor,
    t_max: float,
) -> torch.Tensor:
    """Epps-Pulley x N per direction from the batch-mean ECF (..., M, knots)."""
    gaussian_char = torch.exp(-0.5 * t.pow(2))  # (num_knots,)
    sq_error = (real - gaussian_char).pow(2) + imag.pow(2)

    # Official lejepa EppsPulley: trapezoid weights doubled for the implicit
    # t<0 half, times the phi(t) weight function, and the statistic times N.
    # The N factor is what makes a collapsed batch's penalty grow with batch
    # size; without it lam=0.02 leaves SIGReg ~N times too weak.
    dt = t_max / (t.shape[0] - 1)
    weights = torch.full_like(t, 2 * dt)
    weights[[0, -1]] = dt
    return sq_error @ (weights * gaussian_char) * B  # (..., num_directions)


def _moment_statistic(
    p_mean: torch.Tensor, p2_mean: torch.Tensor, B: int | torch.Tensor
) -> torch.Tensor:
    # Log-bounded on purpose: a plain (E[p^2] - 1)^2 grows as scale^4 and turns
    # an exploding batch into an exploding gradient. Each term is ~chi^2_1 / B
    # near N(0, 1) (null floor ~1.5 after x B at B >= 64; 1.74 at B=16) and pulls back logarithmically
    # in scale and in mean shift -- exactly where Epps-Pulley goes flat.
    var = (p2_mean - p_mean.pow(2)).clamp_min(1e-6)
    return (torch.log1p(p_mean.pow(2) / var) + 0.25 * torch.log(var).pow(2)) * B


def sigreg_loss(
    z: torch.Tensor,
    num_directions: int = 1024,
    num_knots: int = 17,
    t_max: float = 3.0,
    *,
    directions: torch.Tensor | None = None,
    moment_weight: float = 0.0,
) -> torch.Tensor:
    # z: (B, K), single view's batch of embeddings.
    B, K = z.shape
    if directions is None:
        directions = torch.randn(K, num_directions, device=z.device, dtype=z.dtype)
        directions = directions / directions.norm(dim=0, keepdim=True)

    projections = z @ directions  # (B, num_directions)

    t = _knots(num_knots, t_max, z.device, z.dtype)  # (num_knots,)
    # (B, num_directions, num_knots)
    phase = projections.unsqueeze(-1) * t.view(1, 1, -1)

    real = torch.cos(phase).mean(dim=0)  # (num_directions, num_knots)
    imag = torch.sin(phase).mean(dim=0)  # (num_directions, num_knots)

    per_direction = _statistic(real, imag, t, B, t_max)  # (num_directions,)
    if moment_weight > 0:
        per_direction = per_direction + moment_weight * _moment_statistic(
            projections.mean(dim=0), projections.pow(2).mean(dim=0), B
        )
    return per_direction.mean()


def sigreg_sums(
    z: torch.Tensor,
    directions: torch.Tensor,
    num_knots: int = 17,
    t_max: float = 3.0,
) -> dict[str, torch.Tensor]:
    """Per-sample sums SIGReg depends on: ECF real/imag (M, knots) and the
    first two projection moments (M,). Sums over micro-batches add up to the
    pooled batch's, which is what makes pooled SIGReg exact."""
    projections = z @ directions
    t = _knots(num_knots, t_max, z.device, z.dtype)
    phase = projections.unsqueeze(-1) * t.view(1, 1, -1)
    return {
        "real": torch.cos(phase).sum(dim=0),
        "imag": torch.sin(phase).sum(dim=0),
        "p": projections.sum(dim=0),
        "p2": projections.pow(2).sum(dim=0),
    }


def sigreg_from_sums(
    sums: dict[str, torch.Tensor],
    n: int,
    num_knots: int = 17,
    t_max: float = 3.0,
    *,
    moment_weight: float = 0.0,
) -> torch.Tensor:
    """sigreg_loss of the pooled batch from summed `sigreg_sums` (any leading
    view axes are averaged, like levjepa_loss's per-view mean)."""
    real, imag = sums["real"] / n, sums["imag"] / n
    t = _knots(num_knots, t_max, real.device, real.dtype)
    per_direction = _statistic(real, imag, t, n, t_max)
    if moment_weight > 0:
        per_direction = per_direction + moment_weight * _moment_statistic(
            sums["p"] / n, sums["p2"] / n, n
        )
    return per_direction.mean()


def sigreg_empirical_floor(
    batch_size: int,
    dim: int,
    num_directions: int = 1024,
    num_trials: int = 8,
) -> float:
    """Mean `sigreg_loss` on true `N(0, I)` samples at a given batch size.

    The empirical characteristic function has an O(1/batch_size) bias even at
    the true optimum; `sigreg_loss`'s factor of N turns that into an O(1)
    floor that barely depends on batch size -- this measures it so it can be
    compared against the loss actually observed during training (a training
    loss near this floor means the regularizer has converged, not stalled)."""
    values = [
        sigreg_loss(torch.randn(batch_size, dim), num_directions=num_directions).item()
        for _ in range(num_trials)
    ]
    return sum(values) / len(values)


def view_directions(
    opts: SigregOptions,
    key: tuple[int, ...],
    num_views: int,
    dim: int,
    num_directions: int,
    device: torch.device,
) -> list[torch.Tensor] | None:
    """One direction set per view, or None for the original per-call draw."""
    if opts.seed is None and not opts.orthogonal:
        return None
    gens = (
        [_direction_generator(opts.seed, (*key, v)) for v in range(num_views)]
        if opts.seed is not None
        else [None] * num_views
    )
    return [
        sample_directions(
            dim, num_directions, device, generator=g, orthogonal=opts.orthogonal
        )
        for g in gens
    ]


def levjepa_loss(
    z_global: torch.Tensor,
    z_locals: torch.Tensor,
    lam: float = 0.02,
    num_directions: int = 1024,
    *,
    opts: SigregOptions | None = None,
    key: tuple[int, ...] = (),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # Always fp32: under bf16 autocast the SIGReg matmul and cos/sin would run
    # in bf16, whose ~3 significant digits drown the ECF's O(1/B) signal. With
    # amp off the inputs are fp32 already and this is a no-op.
    opts = opts or SigregOptions()
    with torch.autocast(device_type=z_global.device.type, enabled=False), _no_tf32():
        z_global, z_locals = z_global.float(), z_locals.float()
        inv = invariance_loss(z_global, z_locals)

        views = torch.cat([z_global.unsqueeze(1), z_locals], dim=1)  # (B, V+1, K)
        dirs = view_directions(
            opts, key, views.shape[1], views.shape[2], num_directions, views.device
        )
        per_view_sigreg = torch.stack(
            [
                sigreg_loss(
                    views[:, v],
                    num_directions=num_directions,
                    directions=None if dirs is None else dirs[v],
                    moment_weight=opts.moment_weight,
                )
                for v in range(views.shape[1])
            ]
        )
        sigreg = per_view_sigreg.mean()

        total = (1 - lam) * inv + lam * sigreg
    return total, {"invariance": inv, "sigreg": sigreg}
