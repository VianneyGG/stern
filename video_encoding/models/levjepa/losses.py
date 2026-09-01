from __future__ import annotations


import torch


def invariance_loss(z_global: torch.Tensor, z_locals: torch.Tensor) -> torch.Tensor:
    # z_global: (B, K); z_locals: (B, V, K)
    views = torch.cat([z_global.unsqueeze(1), z_locals], dim=1)  # (B, V+1, K)
    diff = views - z_global.unsqueeze(1)
    return diff.pow(2).sum(dim=-1).mean()


def sigreg_loss(
    z: torch.Tensor,
    num_directions: int = 1024,
    num_knots: int = 17,
    t_max: float = 3.0,
) -> torch.Tensor:
    # z: (B, K), single view's batch of embeddings.
    B, K = z.shape
    directions = torch.randn(K, num_directions, device=z.device, dtype=z.dtype)
    directions = directions / directions.norm(dim=0, keepdim=True)

    projections = z @ directions  # (B, num_directions)

    t = torch.linspace(
        0.0, t_max, num_knots, device=z.device, dtype=z.dtype
    )  # (num_knots,)
    # (B, num_directions, num_knots)
    phase = projections.unsqueeze(-1) * t.view(1, 1, -1)

    real = torch.cos(phase).mean(dim=0)  # (num_directions, num_knots)
    imag = torch.sin(phase).mean(dim=0)  # (num_directions, num_knots)

    gaussian_char = torch.exp(-0.5 * t.pow(2))  # (num_knots,)
    sq_error = (real - gaussian_char.unsqueeze(0)).pow(2) + imag.pow(2)

    per_direction = torch.trapezoid(sq_error, t, dim=-1)  # (num_directions,)
    return per_direction.mean()


def levjepa_loss(
    z_global: torch.Tensor,
    z_locals: torch.Tensor,
    lam: float = 0.02,
    num_directions: int = 1024,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    inv = invariance_loss(z_global, z_locals)

    views = torch.cat([z_global.unsqueeze(1), z_locals], dim=1)  # (B, V+1, K)
    per_view_sigreg = torch.stack(
        [
            sigreg_loss(views[:, v], num_directions=num_directions)
            for v in range(views.shape[1])
        ]
    )
    sigreg = per_view_sigreg.mean()

    total = inv + lam * sigreg
    return total, {"invariance": inv, "sigreg": sigreg}
