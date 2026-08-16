from __future__ import annotations

from torch import nn


class TubeletEmbed(nn.Module):
    """`(B, C, T, H, W)` -> `(B, N, embed_dim)` via a Conv3d over tubelets."""

    def __init__(
        self, *, in_channels: int, embed_dim: int, tubelet_size: int, patch_size: int
    ):
        super().__init__()
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x):
        x = self.proj(x)  # (B, D, T', H', W')
        return x.flatten(2).transpose(1, 2)  # (B, N, D), N = T'*H'*W'
