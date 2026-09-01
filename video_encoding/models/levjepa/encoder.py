from __future__ import annotations

import torch
from torch import nn

from video_encoding.models.ViT.block import BlockConfig, ViTBlock
from video_encoding.models.ViT.norm import build_norm
from video_encoding.models.ViT.patch_embed import TubeletEmbed
from video_encoding.models.ViT.rope import Rotary3D


class LeVJEPAEncoder(nn.Module):
    """Per-frame-tokenizing ViT encoder with block-causal attention, factorized
    3D RoPE, and uniform random token dropping during training -- the encoder
    half of LeVJEPA (arXiv:2608.27395).

    Deliberately does not define `patch_embed`/`pos_embed` top-level attributes:
    that absence routes `VideoMAEVARegressor._backbone_tokens`'s hasattr check
    to its `self.backbone(pixel_values=...)` fallback, making this a drop-in
    Stage-2 backbone with zero changes to the regressor.
    """

    def __init__(self, model_cfg):
        super().__init__()
        embed_dim = model_cfg.embed_dim
        num_heads = model_cfg.num_heads
        depth = model_cfg.depth
        mlp_ratio = model_cfg.get("mlp_ratio", 4.0)
        dropout = model_cfg.get("dropout", 0.0)
        drop_path_rate = model_cfg.get("drop_path_rate", 0.0)
        norm_type = model_cfg.get("norm_type", "layernorm")
        qk_norm = model_cfg.get("qk_norm", False)
        in_channels = model_cfg.get("in_channels", 3)

        self.patch_size = model_cfg.patch_size
        self.drop_ratio = model_cfg.get("drop_ratio", 0.95)

        self.tubelet_embed = TubeletEmbed(
            in_channels=in_channels,
            embed_dim=embed_dim,
            tubelet_size=1,
            patch_size=self.patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.rope = Rotary3D(head_dim=embed_dim // num_heads)
        self.blocks = nn.ModuleList(
            [
                ViTBlock(
                    BlockConfig(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        drop_path_rate=drop_path_rate,
                        norm_type=norm_type,
                        qk_norm=qk_norm,
                    )
                )
                for _ in range(depth)
            ]
        )
        self.final_norm = build_norm(norm_type, embed_dim)

        self._coord_cache: dict[tuple[int, int, int, str], torch.Tensor] = {}

    def _patch_coords(
        self, T: int, H: int, W: int, device: torch.device
    ) -> torch.Tensor:
        key = (T, H, W, str(device))
        if key not in self._coord_cache:
            grid_h = H // self.patch_size
            grid_w = W // self.patch_size
            t_idx = (
                torch.arange(T, device=device).view(T, 1, 1).expand(T, grid_h, grid_w)
            )
            h_idx = (
                torch.arange(grid_h, device=device)
                .view(1, grid_h, 1)
                .expand(T, grid_h, grid_w)
            )
            w_idx = (
                torch.arange(grid_w, device=device)
                .view(1, 1, grid_w)
                .expand(T, grid_h, grid_w)
            )
            coords = torch.stack([t_idx, h_idx, w_idx], dim=-1).reshape(-1, 3)
            self._coord_cache[key] = coords
        return self._coord_cache[key]

    def _drop_tokens(
        self, tokens: torch.Tensor, coords: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # tokens: (B, N, D); coords: (N, 3), shared across the batch.
        B, N, _ = tokens.shape
        if not self.training or self.drop_ratio <= 0:
            return tokens, coords.unsqueeze(0).expand(B, N, 3)
        num_keep = max(1, int(round(N * (1 - self.drop_ratio))))
        perm = torch.randperm(N, device=tokens.device)[:num_keep]
        perm, _ = perm.sort()
        kept_tokens = tokens[:, perm, :]
        kept_coords = coords[perm].unsqueeze(0).expand(B, num_keep, 3)
        return kept_tokens, kept_coords

    def _block_causal_mask(self, patch_t: torch.Tensor) -> torch.Tensor:
        # patch_t: (B, N) -- the t-coordinate of each surviving patch token.
        B, N = patch_t.shape
        total = N + 1
        mask = torch.zeros(B, total, total, dtype=torch.bool, device=patch_t.device)
        mask[:, 0, :] = True  # [cls] attends to everything
        t_i = patch_t.unsqueeze(2)
        t_j = patch_t.unsqueeze(1)
        mask[:, 1:, 1:] = t_j <= t_i
        mask[:, 1:, 0] = False  # no patch token attends to [cls]
        return mask.unsqueeze(1)  # (B, 1, total, total), broadcasts over heads

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        B, _, T, H, W = pixel_values.shape
        tokens = self.tubelet_embed(pixel_values)  # (B, N, D)
        coords = self._patch_coords(T, H, W, pixel_values.device)
        tokens, kept_coords = self._drop_tokens(tokens, coords)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, tokens], dim=1)

        cls_coords = torch.zeros(
            B, 1, 3, dtype=kept_coords.dtype, device=kept_coords.device
        )
        full_coords = torch.cat([cls_coords, kept_coords], dim=1)
        cos, sin = self.rope(full_coords)

        attn_mask = self._block_causal_mask(kept_coords[..., 0])

        for block in self.blocks:
            x = block(x, attn_mask=attn_mask, rotary=(cos, sin))
        return self.final_norm(x)

    def forward_cls(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.forward(pixel_values)[:, 0]
