"""VideoMAE for masked autoencoder (MAE) pretraining on unlabelled video.

Architecture:
  1. Encode visible tokens (~10%) through the encoder backbone
  2. Shallow decoder (4 blocks) reconstructs masked tubelet pixels
  3. Loss = MSE on masked positions only

The encoder runs on a subset of tokens (via forward_tokens), the decoder
reconstructs full resolution, and only masked-token predictions are penalized.
"""

from __future__ import annotations

import torch
from omegaconf import DictConfig
from torch import nn

from video_encoding.models.videomaev2.encoder import (
    Block,
    VideoMAEEncoder,
    _IMAGENET_MEAN,
    _IMAGENET_STD,
)


class VideoMAEForPreTraining(nn.Module):
    """Encoder + shallow decoder for MAE pretraining.

    The encoder processes visible tokens (~10%) to extract spatiotemporal
    features. The decoder reconstructs the full spatiotemporal resolution,
    with mask tokens inserted at masked positions. Loss is computed only
    on masked position predictions.
    """

    def __init__(self, model_cfg: DictConfig):
        """
        Args:
            model_cfg: Hydra config with fields:
              - encoder fields (inherited from VideoMAEEncoder)
              - decoder_depth: number of decoder blocks
              - decoder_embed_dim: decoder hidden dimension
              - decoder_num_heads: number of attention heads in decoder
              - mask_ratio: fraction of tokens to mask (e.g., 0.9)
        """
        super().__init__()
        self.model_cfg = model_cfg
        self.encoder = VideoMAEEncoder(model_cfg)
        self.embed_dim = int(model_cfg.embed_dim)
        self.decoder_embed_dim = int(model_cfg.decoder_embed_dim)
        self.mask_ratio = float(model_cfg.mask_ratio)
        self.decoder_mask_ratio = float(model_cfg.get("decoder_mask_ratio", 0.0))

        # Learnable mask token (inserted at masked positions before decoder)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        decoder_dropout = float(model_cfg.get("dropout", 0.0))
        self.decoder_blocks = nn.ModuleList(
            Block(
                embed_dim=self.decoder_embed_dim,
                num_heads=int(model_cfg.decoder_num_heads),
                mlp_ratio=float(model_cfg.get("mlp_ratio", 4.0)),
                dropout=decoder_dropout,
            )
            for _ in range(int(model_cfg.decoder_depth))
        )
        self.decoder_norm = nn.LayerNorm(self.decoder_embed_dim)

        # Decoder needs its own positional embedding: mask tokens are otherwise
        # indistinguishable from one another, so the decoder can't know *which*
        # position it is reconstructing without this.
        num_tokens = int(self.encoder.pos_embed.shape[1])
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_tokens, self.decoder_embed_dim)
        )
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

        # Projection from encoder to decoder embedding space
        self.encoder_to_decoder = nn.Linear(self.embed_dim, self.decoder_embed_dim)

        # Prediction head: reconstruct patch pixels
        # Output shape: (B, num_masked, 3 * patch_size^2) for MAE-style reconstruction
        self.patch_size = int(model_cfg.patch_size)
        self.tubelet_size = int(model_cfg.tubelet_size)
        self.predict_head = nn.Linear(
            self.decoder_embed_dim, 3 * self.patch_size * self.patch_size
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        keep_indices: torch.Tensor | None = None,
        mask_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode, mask, decode, predict.

        Args:
            pixel_values: (B, C, T, H, W) video frames
            keep_indices: (num_kept,) token indices to keep (sampled at train time)
            mask_indices: (num_masked,) token indices to mask

        Returns:
            pred: (B, num_masked, 3*patch_size^2) predicted pixel values
            target: (B, num_masked, 3*patch_size^2) target pixel values
        """
        B, C, T, H, W = pixel_values.shape

        # Embed patches -> (B, N, D)
        x = self.encoder.patch_embed(pixel_values)
        N = x.shape[1]

        if keep_indices is None or mask_indices is None:
            raise ValueError("keep_indices and mask_indices must be provided")

        if self.decoder_mask_ratio > 0.0:
            # V2 dual-masking: subsample which masked tokens contribute to loss/target.
            # The decoder still processes the full token sequence (masked+visible);
            # only the reconstruction target is subsampled, not the decoder's compute footprint.
            n_mask = mask_indices.shape[0]
            n_recon = max(1, round(n_mask * (1.0 - self.decoder_mask_ratio)))
            perm = torch.randperm(n_mask, device=pixel_values.device)[:n_recon]
            mask_indices = mask_indices[perm]

        # Positional embedding is added once, up front, so both the encoder
        # (on visible tokens) and the decoder (on visible + mask tokens) know
        # which spatiotemporal position each token corresponds to.
        x = x + self.encoder.pos_embed

        # Encode only the visible tokens (the whole point of MAE: the encoder
        # never sees masked tokens, so its compute scales with ~10% of N).
        visible = self.encoder.forward_tokens(x, keep_indices)  # (B, num_kept, D)
        visible = self.encoder_to_decoder(visible)  # (B, num_kept, decoder_embed_dim)

        num_masked = mask_indices.shape[0]
        mask_tokens = self.mask_token.expand(B, num_masked, -1)

        # Scatter visible + mask tokens back to their original positions.
        decoder_tokens = visible.new_zeros(B, N, self.decoder_embed_dim)
        # Under autocast, `visible` is the AMP dtype (bf16/fp16) while the
        # mask_token parameter stays fp32; match both to the destination dtype
        # so the index-assign doesn't raise a dtype-mismatch.
        decoder_tokens[:, keep_indices, :] = visible.to(decoder_tokens.dtype)
        decoder_tokens[:, mask_indices, :] = mask_tokens.to(decoder_tokens.dtype)
        decoder_tokens = decoder_tokens + self.decoder_pos_embed

        for block in self.decoder_blocks:
            decoder_tokens = block(decoder_tokens)
        decoder_tokens = self.decoder_norm(decoder_tokens)

        pred_full = self.predict_head(decoder_tokens)  # (B, N, 3*patch_size^2)
        pred = pred_full[:, mask_indices, :]

        target_full = self._patchify_targets(pixel_values)  # (B, N, 3*patch_size^2)
        target = target_full[:, mask_indices, :]

        return pred, target

    def _patchify_targets(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Reconstruction target: per-tubelet, time-averaged, per-patch z-scored
        pixel patches (official VideoMAE recipe -- z-scoring makes the loss
        invariant to a patch's absolute brightness/contrast, matching the
        signal the shallow decoder can actually recover).

        predict_head outputs one patch of `3*patch_size^2` values per token
        (not `tubelet_size` frames worth), so the target must match: average
        pixel values over each tubelet's temporal frames, then patchify
        spatially, then z-score each patch independently. Token order matches
        TubeletEmbed's (T', H', W') layout.
        """
        B, C, T, H, W = pixel_values.shape
        mean = torch.tensor(
            _IMAGENET_MEAN, device=pixel_values.device, dtype=pixel_values.dtype
        ).view(1, C, 1, 1, 1)
        std = torch.tensor(
            _IMAGENET_STD, device=pixel_values.device, dtype=pixel_values.dtype
        ).view(1, C, 1, 1, 1)
        pixel_values = pixel_values * std + mean  # denormalize back to [0, 1]

        t_prime = T // self.tubelet_size
        h_prime = H // self.patch_size
        w_prime = W // self.patch_size

        x = pixel_values.view(B, C, t_prime, self.tubelet_size, H, W).mean(dim=3)
        x = x.view(B, C, t_prime, h_prime, self.patch_size, w_prime, self.patch_size)
        x = x.permute(0, 2, 3, 5, 1, 4, 6)  # (B, T', H', W', C, patch, patch)
        x = x.reshape(
            B, t_prime * h_prime * w_prime, C * self.patch_size * self.patch_size
        )

        mu = x.mean(dim=-1, keepdim=True)
        sigma = x.std(dim=-1, keepdim=True)
        return (x - mu) / (sigma + 1e-6)

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """MSE loss on masked token predictions.

        Args:
            pred: (B, num_masked, 3*patch_size^2) model predictions
            target: (B, num_masked, 3*patch_size^2) ground truth pixel values

        Returns:
            loss: scalar, MSE over masked positions
        """
        loss = torch.mean((pred - target) ** 2)
        return loss
