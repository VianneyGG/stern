"""VideoMAE for masked autoencoder (MAE) pretraining on unlabelled video.

Architecture:
  1. Encode visible tokens (~10%) through the encoder backbone
  2. Shallow decoder (4 blocks) reconstructs masked tubelet pixels
  3. Loss = MSE on masked positions only

The encoder runs on a subset of tokens (via forward_tokens), the decoder
reconstructs full resolution, and only masked-token predictions are penalized.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from omegaconf import DictConfig
from torch import nn

from stern.models.vit.block import BlockConfig, ViTBlock
from stern.models.videomae.encoder import (
    VideoMAEEncoder,
    _IMAGENET_MEAN,
    _IMAGENET_STD,
)


class MAEAux(NamedTuple):
    """Auxiliary loss terms for the E2b ablation arms.

    Every field is empty under the committed recipe, so `compute_loss` with a
    default-config model returns the same scalar it always did.
    """

    visible_pred: torch.Tensor | None = None
    visible_target: torch.Tensor | None = None
    visible_weights: torch.Tensor | None = None
    deep_preds: list[torch.Tensor] = []  # noqa: RUF012 - NamedTuple field default


def visible_isolation_weights(
    keep_indices: torch.Tensor,
    token_grid: tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    """`w_i = 1 - exp(-d_i)` for each visible token, `d_i` its Euclidean distance
    in the (H', W') *spatial* plane to the nearest visible token outside its own
    tube.

    A visible patch with a visible spatial neighbour can be reconstructed by
    copying sideways, so it exerts no pressure to carry its own content through
    depth. An isolated one has nowhere to copy from: the only route to its
    reconstruction is its own token surviving all twelve blocks. Weighting by
    isolation aims arm B's extra term at exactly the tokens whose loss the
    residual stream has to preserve, which is what the ablation is about.

    Two properties of the mask geometry force this definition (measured on the
    production (8, 14, 14) grid; see test_mae_e2b_arms.py):

    - **Distance to the nearest *masked* token cannot work.** At mask_ratio 0.9
      every visible token sits at d=1.0 from a hole -- only 10% of tokens
      survive, so none is ever deep inside a visible region. It is still
      degenerate at 0.75, and only spreads at 0.5, far below anything we run.
    - **The distance must ignore T'.** Tube masking keeps or drops a whole
      (h, w) column across every temporal position, so a token's own tube is
      always visible at d=1 along T' and a 3D metric collapses to a constant
      regardless of ratio.

    Isolation behaves the other way round: it spreads *best* at the production
    ratio (six distinct values, max 3.16, at 0.9) and degenerates as masking
    falls -- so this arm is only meaningful in the regime we actually train in.

    The length scale is one grid cell. It is fixed rather than configurable: the
    arm already has a free parameter (`visible_loss_weight`) to be chosen by
    smoke run, and a second interacting one would make that choice unreadable.
    """
    t, h, w = token_grid
    coords = torch.stack(
        torch.meshgrid(
            torch.arange(t, device=device, dtype=torch.float32),
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 3)  # (N, 3), matching TubeletEmbed's token order

    keep_hw = coords[keep_indices][..., 1:]  # (B, num_kept, 2) -- drop T'
    d = torch.cdist(keep_hw, keep_hw)  # (B, num_kept, num_kept)
    # d == 0 covers both the token itself and its temporal copies: same (h, w),
    # so neither is a spatial neighbour to copy from.
    d = d.masked_fill(d == 0, float("inf"))
    nearest = d.amin(dim=-1)  # (B, num_kept)
    # A single visible column leaves every distance infinite -> exp(-inf) = 0
    # -> w = 1 uniformly. Degenerate but finite; no NaN reaches the loss.
    return 1.0 - torch.exp(-nearest)


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
            ViTBlock(
                BlockConfig(
                    embed_dim=self.decoder_embed_dim,
                    num_heads=int(model_cfg.decoder_num_heads),
                    mlp_ratio=float(model_cfg.get("mlp_ratio", 4.0)),
                    dropout=decoder_dropout,
                )
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
        # Output shape: (B, num_masked, out_frames * 3 * patch_size^2), where
        # out_frames tracks target_mode (1 under time_avg, tubelet_size under per_frame).
        self.patch_size = int(model_cfg.patch_size)
        self.tubelet_size = int(model_cfg.tubelet_size)
        self.target_mode = str(model_cfg.get("target_mode", "time_avg"))
        if self.target_mode not in ("time_avg", "per_frame"):
            raise ValueError(
                f"unknown target_mode {self.target_mode!r}: expected time_avg|per_frame"
            )
        # per_frame keeps one patch per tubelet *frame*, so intra-tubelet motion
        # survives into the target instead of being averaged away.
        out_frames = self.tubelet_size if self.target_mode == "per_frame" else 1
        self.predict_head = nn.Linear(
            self.decoder_embed_dim, out_frames * 3 * self.patch_size * self.patch_size
        )

        # --- E2b ablation knobs. All three default to the committed recipe, so
        # arm A is bit-comparable to the runs already on the cluster.
        self.visible_loss_weight = float(model_cfg.get("visible_loss_weight", 0.0))
        self.visible_loss_isolation_weighting = bool(
            model_cfg.get("visible_loss_isolation_weighting", False)
        )
        depth = int(model_cfg.depth)
        # 1-based to match the prompt's [4, 8, 12]; 0 would silently mean "the
        # last block" under Python indexing, which is the opposite of intent.
        self.deep_supervision_layers = [
            int(i) for i in (model_cfg.get("deep_supervision_layers") or [])
        ]
        for layer in self.deep_supervision_layers:
            if not 1 <= layer <= depth:
                raise ValueError(
                    f"deep_supervision_layers: {layer} out of range for a "
                    f"{depth}-block encoder (expected 1..{depth}, 1-based)"
                )

    def forward(
        self,
        pixel_values: torch.Tensor,
        keep_indices: torch.Tensor | None = None,
        mask_indices: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, "MAEAux"]
    ):
        """Encode, mask, decode, predict.

        Args:
            pixel_values: (B, C, T, H, W) video frames
            keep_indices: (B, num_kept) token indices to keep (sampled at train time)
            mask_indices: (B, num_masked) token indices to mask
            return_aux: also return the E2b auxiliary terms. Off by default, so
                the committed `pred, target = model(...)` call sites are
                untouched and the eval path keeps measuring plain masked MSE.

        Returns:
            pred: (B, num_masked, 3*patch_size^2) predicted pixel values
            target: (B, num_masked, 3*patch_size^2) target pixel values
            aux: `MAEAux`, only when `return_aux=True`
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
            # Independent subsample per row, same n_recon count for every
            # row (decoder_mask_ratio is a fixed fraction of n_mask).
            n_mask = mask_indices.shape[1]
            n_recon = max(1, round(n_mask * (1.0 - self.decoder_mask_ratio)))
            noise = torch.rand(B, n_mask, device=pixel_values.device)
            perm = torch.argsort(noise, dim=1)[:, :n_recon]
            mask_indices = torch.gather(mask_indices, 1, perm)

        # Positional embedding is added once, up front, so both the encoder
        # (on visible tokens) and the decoder (on visible + mask tokens) know
        # which spatiotemporal position each token corresponds to.
        x = x + self.encoder.pos_embed

        # Encode only the visible tokens (the whole point of MAE: the encoder
        # never sees masked tokens, so its compute scales with ~10% of N).
        if self.deep_supervision_layers and return_aux:
            visible, intermediates = self.encoder.forward_tokens(
                x, keep_indices, return_layers=self.deep_supervision_layers
            )
        else:
            visible = self.encoder.forward_tokens(x, keep_indices)
            intermediates = []

        pred_full = self._decode(visible, keep_indices, mask_indices, B, N)

        pred_idx = mask_indices.unsqueeze(-1).expand(-1, -1, pred_full.shape[-1])
        pred = torch.gather(pred_full, 1, pred_idx)

        target_full = self._patchify_targets(pixel_values)  # (B, N, out_dim)
        target_idx = mask_indices.unsqueeze(-1).expand(-1, -1, target_full.shape[-1])
        target = torch.gather(target_full, 1, target_idx)

        if not return_aux:
            return pred, target

        visible_pred = visible_target = visible_weights = None
        if self.visible_loss_weight > 0.0:
            # No extra decoder compute: pred_full already covers all N tokens,
            # so the visible term is a second gather, at keep_indices.
            keep_out_idx = keep_indices.unsqueeze(-1).expand(
                -1, -1, pred_full.shape[-1]
            )
            visible_pred = torch.gather(pred_full, 1, keep_out_idx)
            visible_target = torch.gather(target_full, 1, keep_out_idx)
            if self.visible_loss_isolation_weighting:
                visible_weights = visible_isolation_weights(
                    keep_indices,
                    self.encoder.token_grid,
                    pixel_values.device,
                )

        # Each supervised layer is decoded by the *same* decoder and head. A
        # per-layer decoder would confound "layer 4 carries reconstructable
        # information" with "layer 4's private decoder learned to read it".
        deep_preds = [
            torch.gather(
                self._decode(tokens, keep_indices, mask_indices, B, N), 1, pred_idx
            )
            for tokens in intermediates
        ]

        return (
            pred,
            target,
            MAEAux(
                visible_pred=visible_pred,
                visible_target=visible_target,
                visible_weights=visible_weights,
                deep_preds=deep_preds,
            ),
        )

    def _decode(
        self,
        visible: torch.Tensor,
        keep_indices: torch.Tensor,
        mask_indices: torch.Tensor,
        B: int,
        N: int,
    ) -> torch.Tensor:
        """Encoder tokens -> per-token pixel predictions over all N positions.

        Factored out of `forward` so deep supervision can push an intermediate
        block's tokens through the identical decoder path.
        """
        visible = self.encoder_to_decoder(visible)  # (B, num_kept, decoder_embed_dim)

        num_masked = mask_indices.shape[1]
        mask_tokens = self.mask_token.expand(B, num_masked, -1)

        # Scatter visible + mask tokens back to their per-sample positions.
        decoder_tokens = visible.new_zeros(B, N, self.decoder_embed_dim)
        keep_idx = keep_indices.unsqueeze(-1).expand(-1, -1, self.decoder_embed_dim)
        mask_idx = mask_indices.unsqueeze(-1).expand(-1, -1, self.decoder_embed_dim)
        # Under autocast, `visible` is the AMP dtype (bf16/fp16) while the
        # mask_token parameter stays fp32; match both to the destination dtype
        # so the scatter doesn't raise a dtype-mismatch.
        decoder_tokens.scatter_(1, keep_idx, visible.to(decoder_tokens.dtype))
        decoder_tokens.scatter_(1, mask_idx, mask_tokens.to(decoder_tokens.dtype))
        decoder_tokens = decoder_tokens + self.decoder_pos_embed

        for block in self.decoder_blocks:
            decoder_tokens = block(decoder_tokens)
        decoder_tokens = self.decoder_norm(decoder_tokens)

        return self.predict_head(decoder_tokens)  # (B, N, out_dim)

    def _patchify_targets(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Reconstruction target: per-tubelet, per-patch z-scored pixel patches
        (official VideoMAE recipe -- z-scoring makes the loss invariant to a
        patch's absolute brightness/contrast, matching the signal the shallow
        decoder can actually recover).

        predict_head's width tracks target_mode: under time_avg it outputs one
        patch of `3*patch_size^2` values per token (tubelet frames averaged
        together); under per_frame it outputs `tubelet_size` such patches, so
        the target keeps one patch per tubelet frame instead. Token order
        matches TubeletEmbed's (T', H', W') layout.
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

        ts, ps_ = self.tubelet_size, self.patch_size
        x = pixel_values.view(B, C, t_prime, ts, H, W)
        if self.target_mode == "time_avg":
            x = x.mean(dim=3, keepdim=True)
        n_frames = x.shape[3]  # 1 under time_avg, tubelet_size under per_frame
        x = x.view(B, C, t_prime, n_frames, h_prime, ps_, w_prime, ps_)
        # -> (B, T', H', W', frame, C, ps, ps): frame-major within a token, so
        # collapsing to the time-averaged layout is .view(B, N, f, -1).mean(2)
        x = x.permute(0, 2, 4, 6, 3, 1, 5, 7)
        x = x.reshape(B, t_prime * h_prime * w_prime, n_frames * C * ps_ * ps_)

        mu = x.mean(dim=-1, keepdim=True)
        sigma = x.std(dim=-1, keepdim=True)
        return (x - mu) / (sigma + 1e-6)

    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        aux: "MAEAux | None" = None,
    ) -> torch.Tensor:
        """MSE loss on masked token predictions, plus any E2b auxiliary terms.

        Args:
            pred: (B, num_masked, 3*patch_size^2) model predictions
            target: (B, num_masked, 3*patch_size^2) ground truth pixel values
            aux: auxiliary terms from `forward(..., return_aux=True)`. Omitted
                by the eval path, so val loss stays plain masked MSE and remains
                comparable across arms.

        Returns:
            loss: scalar. Stays a scalar under every arm so both training
            entrypoints are unchanged at the call site.
        """
        loss = torch.mean((pred - target) ** 2)
        if aux is None:
            return loss

        if aux.visible_pred is not None:
            se = (aux.visible_pred - aux.visible_target) ** 2
            if aux.visible_weights is None:
                visible = se.mean()
            else:
                # Weighted mean over tokens: normalise by the weight sum so the
                # term's scale does not drift with the mask draw.
                per_token = se.mean(dim=-1)  # (B, num_kept)
                w = aux.visible_weights.to(per_token.dtype)
                visible = (per_token * w).sum() / (w.sum() + 1e-6)
            loss = loss + self.visible_loss_weight * visible

        if aux.deep_preds:
            # Averaged over layers, then added at unit weight, so the auxiliary
            # term's magnitude is comparable to the main one regardless of how
            # many layers are supervised.
            deep = torch.stack(
                [torch.mean((p - target) ** 2) for p in aux.deep_preds]
            ).mean()
            loss = loss + deep

        return loss
