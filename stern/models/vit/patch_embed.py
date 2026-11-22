from __future__ import annotations

from torch import nn


class TubeletEmbed(nn.Module):
    """`(B, C, T, H, W)` -> `(B, N, embed_dim)` via a Conv3d over tubelets."""

    def __init__(
        self,
        *,
        in_channels: int,
        embed_dim: int,
        tubelet_size: int,
        patch_size: int,
        norm: bool = False,
    ):
        super().__init__()
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )
        # Optional LayerNorm on the patch tokens (before pos_embed), so the
        # stream starts at unit scale rather than tiny next to block 1's write.
        self.norm = nn.LayerNorm(embed_dim) if norm else nn.Identity()

    def forward(self, x):
        x = self.proj(x)  # (B, D, T', H', W')
        return self.norm(x.flatten(2).transpose(1, 2))  # (B, N, D), N = T'*H'*W'
