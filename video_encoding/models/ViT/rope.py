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
        if head_dim % 6 != 0:
            raise ValueError(
                f"head_dim {head_dim} must be divisible by 6 "
                "(3 axes x 2D rotation each)"
            )
        self.head_dim = head_dim
        group_dim = head_dim // 3
        inv_freq = 1.0 / (
            base ** (torch.arange(0, group_dim, 2, dtype=torch.float32) / group_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # coords: (B, N, 3) long, columns (t, h, w)
        coords = coords.to(self.inv_freq.dtype)
        cos_parts = []
        sin_parts = []
        for axis in range(3):
            pos = coords[..., axis]  # (B, N)
            freqs = pos.unsqueeze(-1) * self.inv_freq  # (B, N, group_dim/2)
            freqs = torch.cat([freqs, freqs], dim=-1)  # (B, N, group_dim)
            cos_parts.append(freqs.cos())
            sin_parts.append(freqs.sin())
        cos = torch.cat(cos_parts, dim=-1)  # (B, N, head_dim)
        sin = torch.cat(sin_parts, dim=-1)
        return cos, sin


def _rotate_half(x: torch.Tensor, num_groups: int = 3) -> torch.Tensor:
    # x's last dim is 3 concatenated (t, h, w) groups; rotate-half must pair
    # elements within each group, not across the whole head_dim.
    *lead, dim = x.shape
    group_dim = dim // num_groups
    x = x.reshape(*lead, num_groups, group_dim)
    half = group_dim // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return rotated.reshape(*lead, dim)


def apply_rotary(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # q, k: (B, heads, N, head_dim); cos, sin: (B, 1, N, head_dim), already
    # unsqueezed by the caller to broadcast over the heads axis.
    q_out = q * cos + _rotate_half(q) * sin
    k_out = k * cos + _rotate_half(k) * sin
    return q_out, k_out
