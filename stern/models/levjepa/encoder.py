"""LeVJEPA encoder: a per-frame-tokenising ViT with block-causal attention and 3D RoPE."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from stern.models.vit.block import BlockConfig, ViTBlock
from stern.models.vit.norm import build_norm
from stern.models.vit.patch_embed import TubeletEmbed
from stern.models.vit.rope import Rotary3D


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

        self.num_frames = int(model_cfg.num_frames)
        self.image_size = int(model_cfg.image_size)
        self.embed_dim = int(embed_dim)
        self.patch_size = model_cfg.patch_size
        self.drop_ratio = model_cfg.get("drop_ratio", 0.95)
        # Recompute each block in backward instead of storing its activations:
        # lifts the per-GPU micro-batch (and so SIGReg's batch) at ~1 extra
        # forward. Exact because token dropping happens before the blocks and
        # the blocks draw no randomness (dropout/drop_path 0 in levjepa_lean).
        self.grad_checkpoint = bool(model_cfg.get("grad_checkpoint", False))
        # Reference grid granularity for RoPE coordinates: local-view crops
        # are rescaled into this frame (see _patch_coords_for_crop) so a
        # crop's positional encoding reflects where it came from in the full
        # frame, not its own 0..grid-1 indices.
        self.global_grid = model_cfg.image_size // self.patch_size

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
        # Opt-in (train.rng_split): token-drop scores come from this
        # generator instead of the global RNG; see reseed_drop. None = global.
        self.drop_generator: torch.Generator | None = None

    def reseed_drop(self, seed: int) -> None:
        """Draw token-drop scores from a private generator seeded with
        `seed`, on the encoder's device."""
        device = self.cls_token.device
        self.drop_generator = torch.Generator(device=device).manual_seed(seed)

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

    def _patch_coords_for_crop(
        self, T: int, H: int, W: int, crop_box: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """Per-sample patch coordinates for a local-view crop, expressed in
        the *global* view's patch-grid units.

        `crop_box`: (B, 4) fractional (top, left, height, width) of the
        original frame the crop was taken from. Each local patch's (h, w)
        index is converted to a fractional position within the crop, then
        mapped into `self.global_grid` units via the crop's offset/extent --
        so a bottom-right crop reports coordinates near `global_grid - 1`,
        not near 0.
        """
        base = self._patch_coords(T, H, W, device).float()  # (N, 3): t, h, w
        grid_h = H // self.patch_size
        grid_w = W // self.patch_size
        t = base[:, 0].unsqueeze(0)  # (1, N)
        h_frac = ((base[:, 1] + 0.5) / grid_h).unsqueeze(0)  # (1, N)
        w_frac = ((base[:, 2] + 0.5) / grid_w).unsqueeze(0)  # (1, N)

        top, left, height, width = crop_box.unbind(dim=-1)  # each (B,)
        global_h = (top.unsqueeze(1) + h_frac * height.unsqueeze(1)) * self.global_grid
        global_w = (left.unsqueeze(1) + w_frac * width.unsqueeze(1)) * self.global_grid
        t = t.expand(crop_box.shape[0], -1)
        return torch.stack([t, global_h, global_w], dim=-1)  # (B, N, 3)

    def _drop_tokens(
        self, tokens: torch.Tensor, coords: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # tokens: (B, N, D); coords: (B, N, 3), per-sample.
        B, N, D = tokens.shape
        if not self.training or self.drop_ratio <= 0:
            return tokens, coords
        num_keep = max(1, int(round(N * (1 - self.drop_ratio))))
        # Independent random ranking per batch element -> each sample keeps
        # its own subset, rather than one mask shared across the batch.
        # No generator kwarg when unset: an explicit generator=None breaks
        # torch.compile (train.levjepa_compile) once a second batch shape
        # makes the graph dynamic.
        if self.drop_generator is None:
            scores = torch.rand(B, N, device=tokens.device)
        else:
            scores = torch.rand(
                B, N, device=tokens.device, generator=self.drop_generator
            )
        perm = scores.argsort(dim=1)[:, :num_keep]
        perm, _ = perm.sort(dim=1)
        kept_tokens = torch.gather(tokens, 1, perm.unsqueeze(-1).expand(-1, -1, D))
        kept_coords = torch.gather(coords, 1, perm.unsqueeze(-1).expand(-1, -1, 3))
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

    def forward(
        self, pixel_values: torch.Tensor, crop_box: torch.Tensor | None = None
    ) -> torch.Tensor:
        """`crop_box`: (B, 4) fractional (top, left, height, width) of the
        original frame this view was cropped from -- pass it when
        `pixel_values` is a local view, so its RoPE coordinates land in the
        global view's grid frame instead of the local view's own 0..grid-1
        indices. Omit for the global view (identity crop)."""
        B, _, T, H, W = pixel_values.shape
        tokens = self.tubelet_embed(pixel_values)  # (B, N, D)
        if crop_box is None:
            coords = self._patch_coords(T, H, W, pixel_values.device)
            coords = coords.unsqueeze(0).expand(B, -1, 3)
        else:
            coords = self._patch_coords_for_crop(
                T, H, W, crop_box.to(pixel_values.device), pixel_values.device
            )
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
            if self.grad_checkpoint and torch.is_grad_enabled():
                x = checkpoint(block, x, attn_mask=attn_mask, rotary=(cos, sin), use_reentrant=False)
            else:
                x = block(x, attn_mask=attn_mask, rotary=(cos, sin))
        return self.final_norm(x)

    def forward_cls(
        self, pixel_values: torch.Tensor, crop_box: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.forward(pixel_values, crop_box=crop_box)[:, 0]
