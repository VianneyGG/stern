"""Tube masking for VideoMAE pretraining.

VideoMAE masks "tubelets" — the same spatial positions (H', W') across all
temporal positions (T'). This differs from random per-frame masking because
the encoder's temporal attention sees entirely masked regions (coherent motion),
which encourages learning motion patterns, not just appearance.

Tokens are laid out T' × H' × W' in sequence:
  - N = T' * H' * W' total tokens
  - A "tube" at position (h, w) masks tokens at indices {(t, h, w) | t ∈ [0, T')}
"""

from __future__ import annotations

import torch


class RandomTubeMasking:
    """Sample which tokens to keep/mask for one training sample.

    Tokens are laid out (B, N, D) where N = T' * H' * W'.
    A tube masks the same (H', W') positions for all T' frames.
    """

    def __init__(self, mask_ratio: float = 0.9):
        """
        Args:
            mask_ratio: fraction of tubelets to mask (0.9 = keep 10%)
        """
        self.mask_ratio = mask_ratio

    def __call__(
        self,
        pixel_values: "torch.Tensor | None",
        grid: tuple[int, int, int],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample tube mask for one training sample.

        Args:
            pixel_values: unused (random masking doesn't look at content); kept
                for signature parity with MotionGuidedTubeMasking so callers can
                select either masker interchangeably.
            grid: (T', H', W') token grid dimensions (from encoder.token_grid).
                Tokens are laid out in this order, so the flat index of the token
                at temporal position t and spatial position s (0 <= s < H'*W') is
                    token_id(t, s) = t * (H' * W') + s
            device: torch device (cpu or cuda)

        Returns:
            keep_indices: (num_kept,) indices of visible tokens
            mask_indices: (num_masked,) indices of masked tokens

        Postcondition (tube structure):
            - A spatial position s is either kept in EVERY frame t or masked in
              EVERY frame t — never mixed.
            - num kept spatial positions = round((1 - mask_ratio) * H' * W')
            - len(keep_indices) = T' * (kept spatial positions)
            - keep_indices ∪ mask_indices = [0, T'*H'*W')  (disjoint, complete)
        """
        del pixel_values  # unused: random masking doesn't look at content
        T, H, W = grid
        n_spatial = H * W
        keep_spatial = round((1 - self.mask_ratio) * n_spatial)

        perm = torch.randperm(n_spatial, device=device)
        keep_spatial_ids = perm[:keep_spatial]
        mask_spatial_ids = perm[keep_spatial:]

        frame_offsets = torch.arange(T, device=device) * n_spatial  # (T,)
        keep_indices = (frame_offsets[:, None] + keep_spatial_ids[None, :]).flatten()
        mask_indices = (frame_offsets[:, None] + mask_spatial_ids[None, :]).flatten()

        keep_indices, _ = torch.sort(keep_indices)
        mask_indices, _ = torch.sort(mask_indices)
        return keep_indices, mask_indices


class RandomFrameMasking:
    """Mask whole timesteps, then top up with scattered tokens.

    The complement of tube masking. Tube masking leaves every masked spatial
    column visible at every *other* timestep, so a model can solve the task by
    pooling its own column over time and never learn temporal structure. Here
    whole timesteps are hidden, so reconstructing them requires predicting
    across time.

    `min_visible_timesteps` bounds how far that goes. Without it the production
    geometry (T'=8, mask_ratio 0.9) hides int(0.9*8) = 7 of 8 timesteps, leaving
    the encoder a single frame -- that is not "no guaranteed temporal twin", it
    is single-frame MAE plus seven-step extrapolation, and it confounds the
    factorial: a worse score could equally mean the encoder was starved of
    input rather than that temporal modelling is hard. One token per surviving
    timestep is also held back from the scattered top-up, so the floor is a
    guarantee and not merely likely.

    The remaining budget is taken as scattered tokens from the surviving
    timesteps, so the total masked count matches RandomTubeMasking and the two
    are comparable at a fixed token budget. The reserved tokens cap that budget:
    at mask_ratio 1.0 the count falls short by `min_visible_timesteps`, because
    masking everything and keeping a visible timestep cannot both hold.
    """

    def __init__(self, mask_ratio: float = 0.9, min_visible_timesteps: int = 2):
        self.mask_ratio = mask_ratio
        self.min_visible_timesteps = min_visible_timesteps

    def __call__(
        self,
        pixel_values: "torch.Tensor | None",
        grid: tuple[int, int, int],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del pixel_values  # unused: random masking doesn't look at content
        T, H, W = grid
        n_spatial = H * W
        n_tokens = T * n_spatial
        n_mask = round(self.mask_ratio * n_tokens)

        n_full = min(int(self.mask_ratio * T), max(0, T - self.min_visible_timesteps))

        masked = torch.zeros(n_tokens, dtype=torch.bool, device=device)
        order = torch.randperm(T, device=device)
        full_t, survivor_t = order[:n_full], order[n_full:]
        spatial = torch.arange(n_spatial, device=device)
        flat = ((full_t * n_spatial)[:, None] + spatial[None, :]).reshape(-1)
        masked[flat] = True

        # One token per surviving timestep is exempt from the top-up below.
        reserved = torch.zeros(n_tokens, dtype=torch.bool, device=device)
        keep_one = torch.randint(n_spatial, (survivor_t.numel(),), device=device)
        reserved[survivor_t * n_spatial + keep_one] = True

        n_extra = n_mask - int(masked.sum())
        if n_extra > 0:
            candidates = (~masked & ~reserved).nonzero(as_tuple=True)[0]
            pick = torch.randperm(candidates.numel(), device=device)[:n_extra]
            masked[candidates[pick]] = True

        # nonzero returns ascending indices, matching RandomTubeMasking's sort.
        return (~masked).nonzero(as_tuple=True)[0], masked.nonzero(as_tuple=True)[0]


class MixedTubeFrameMasking:
    """Per-call coin flip between tube and frame-level masking.

    Used rather than pure frame masking for the factorial's mask arm: pure frame
    masking removes the spatial-inpainting signal entirely, changing more than
    one variable. Mixing preserves tube masking half the time and removes only
    the *guarantee* of a visible temporal twin.
    """

    def __init__(
        self,
        mask_ratio: float = 0.9,
        p_frame: float = 0.5,
        min_visible_timesteps: int = 2,
    ):
        self.mask_ratio = mask_ratio
        self.p_frame = p_frame
        self._tube = RandomTubeMasking(mask_ratio)
        self._frame = RandomFrameMasking(mask_ratio, min_visible_timesteps)

    def __call__(
        self,
        pixel_values: "torch.Tensor | None",
        grid: tuple[int, int, int],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if float(torch.rand(1, device=device)) < self.p_frame:
            return self._frame(pixel_values, grid, device)
        return self._tube(pixel_values, grid, device)


class MotionGuidedTubeMasking:
    """Tube masking biased toward high-motion spatial tubelets: patches with
    large frame-to-frame pixel change are preferentially masked, forcing the
    encoder to reconstruct moving regions from static context instead of
    getting to "cheat" by keeping every visible-motion patch.

    Same tube structure as RandomTubeMasking: identical spatial positions
    masked across all T' tubelets. Motion energy is averaged across BOTH time
    and the batch to produce one shared mask per call, matching
    RandomTubeMasking's batch-shared masking contract (STERN samples one mask
    per training step, not one per sample -- unlike the per-sample reference
    implementation this was adapted from).
    """

    def __init__(self, mask_ratio: float = 0.9):
        self.mask_ratio = mask_ratio

    def __call__(
        self,
        pixel_values: torch.Tensor,
        grid: tuple[int, int, int],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        T, H, W = grid
        n_spatial = H * W
        _, _, _, h_full, w_full = pixel_values.shape
        ph, pw = h_full // H, w_full // W

        diffs = (
            pixel_values[:, :, 1:] - pixel_values[:, :, :-1]
        ).abs()  # (B, C, T_full-1, h_full, w_full)
        diffs = diffs.mean(
            dim=1
        )  # average over channels -> (B, T_full-1, h_full, w_full)
        diffs = diffs.unfold(2, ph, ph).unfold(3, pw, pw)  # (B, T_full-1, H, W, ph, pw)
        motion = diffs.mean(
            dim=(1, 4, 5)
        )  # (B, H, W): avg over time + within-patch pixels
        motion = motion.mean(dim=0).reshape(n_spatial)  # (n_spatial,): avg over batch

        # Tie-break noise so a perfectly static batch still yields a valid mask.
        motion = motion + torch.rand(n_spatial, device=device) * 1e-6

        keep_spatial = round((1 - self.mask_ratio) * n_spatial)
        n_mask_s = n_spatial - keep_spatial
        ids_sorted = torch.argsort(motion, descending=True)  # highest motion first
        ids_mask_s = ids_sorted[:n_mask_s]
        ids_keep_s = ids_sorted[n_mask_s:]

        t_off = torch.arange(T, device=device) * n_spatial
        keep_indices = (ids_keep_s.unsqueeze(0) + t_off.unsqueeze(1)).reshape(-1)
        mask_indices = (ids_mask_s.unsqueeze(0) + t_off.unsqueeze(1)).reshape(-1)
        keep_indices, _ = torch.sort(keep_indices)
        mask_indices, _ = torch.sort(mask_indices)
        return keep_indices, mask_indices


def build_masker(model_cfg):
    """Select the masking strategy from a model config.

    Lives here rather than in pretrain.py because both training entrypoints
    need it and pretrain.py <-> mesh_round.py is already a real import cycle
    (broken only by a function-local import); importing the selector from
    either one into the other would tighten it.

    `motion_mask` predates `mask_type` and stays authoritative, so no existing
    config changes meaning when `mask_type` is absent.
    """
    ratio = float(model_cfg.mask_ratio)
    if bool(model_cfg.get("motion_mask", False)):
        return MotionGuidedTubeMasking(mask_ratio=ratio)
    mask_type = str(model_cfg.get("mask_type", "tube"))
    if mask_type == "tube":
        return RandomTubeMasking(mask_ratio=ratio)
    if mask_type == "frame":
        return RandomFrameMasking(mask_ratio=ratio)
    if mask_type == "mixed":
        return MixedTubeFrameMasking(mask_ratio=ratio)
    raise ValueError(f"unknown mask_type {mask_type!r}: expected tube|frame|mixed")
