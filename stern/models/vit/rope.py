"""Factorised 3D rotary position embedding (RoPE) for (t, h, w) token grids."""

from __future__ import annotations

import torch
from torch import nn


class Rotary3D(nn.Module):
    """Factorized 3D rotary position embedding.

    Splits ``head_dim`` into three equal groups, one per (t, h, w) axis, and
    computes standard RoPE cos/sin for each group from its own coordinate
    column before concatenating back to ``head_dim``.
    """

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0 or head_dim < 6:
            raise ValueError(
                f"head_dim {head_dim} must be even and >= 6 (3 axes x 2D rotation each)"
            )
        self.head_dim = head_dim
        self.group_dims = _balanced_even_split(head_dim, 3)
        inv_freqs = [
            1.0 / (base ** (torch.arange(0, gd, 2, dtype=torch.float32) / gd))
            for gd in self.group_dims
        ]
        # Padded to a common length and stacked so a single buffer survives
        # .to(device)/state_dict round-trips; sliced back per-axis in forward.
        max_len = max(f.numel() for f in inv_freqs)
        padded = torch.zeros(3, max_len)
        for i, f in enumerate(inv_freqs):
            padded[i, : f.numel()] = f
        self.register_buffer("inv_freq", padded, persistent=False)

    def forward(self, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # coords: (B, N, 3) long, columns (t, h, w)
        coords = coords.to(self.inv_freq.dtype)
        cos_parts = []
        sin_parts = []
        for axis, group_dim in enumerate(self.group_dims):
            pos = coords[..., axis]  # (B, N)
            inv_freq = self.inv_freq[axis, : group_dim // 2]
            freqs = pos.unsqueeze(-1) * inv_freq  # (B, N, group_dim/2)
            freqs = torch.cat([freqs, freqs], dim=-1)  # (B, N, group_dim)
            cos_parts.append(freqs.cos())
            sin_parts.append(freqs.sin())
        cos = torch.cat(cos_parts, dim=-1)  # (B, N, head_dim)
        sin = torch.cat(sin_parts, dim=-1)
        return cos, sin


def _balanced_even_split(total: int, num_groups: int) -> list[int]:
    """Split `total` into `num_groups` sizes as equal as possible, each even
    (rotate-half pairs elements within a group, so odd groups would break).

    e.g. (12, 3) -> [4, 4, 4]; (64, 3) -> [22, 22, 20].
    """
    base = (total // num_groups // 2) * 2
    sizes = [base] * num_groups
    remainder = total - base * num_groups
    i = 0
    while remainder > 0:
        sizes[i] += 2
        remainder -= 2
        i += 1
    return sizes


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    # x's last dim is 3 concatenated (t, h, w) groups, possibly uneven sizes
    # (see _balanced_even_split); rotate-half must pair elements within each
    # group, not across the whole head_dim.
    *lead, dim = x.shape
    group_dims = _balanced_even_split(dim, 3)
    parts = []
    offset = 0
    for group_dim in group_dims:
        chunk = x[..., offset : offset + group_dim]
        half = group_dim // 2
        x1, x2 = chunk[..., :half], chunk[..., half:]
        parts.append(torch.cat([-x2, x1], dim=-1))
        offset += group_dim
    return torch.cat(parts, dim=-1)


def apply_rotary(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # q, k: (B, heads, N, head_dim); cos, sin: (B, 1, N, head_dim), already
    # unsqueezed by the caller to broadcast over the heads axis.
    q_out = q * cos + _rotate_half(q) * sin
    k_out = k * cos + _rotate_half(k) * sin
    return q_out, k_out
