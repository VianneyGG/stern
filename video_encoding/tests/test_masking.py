"""Unit tests for tube masking."""

import torch

from video_encoding.models.videomaev2.masking import (
    MotionGuidedTubeMasking,
    RandomTubeMasking,
)

# videomae_base token grid: T'=8, H'=W'=14  ->  8 * 14 * 14 = 1568 tokens
BASE_GRID = (8, 14, 14)


def _expected_kept(grid: tuple[int, int, int], ratio: float) -> int:
    """Tube masking keeps whole spatial columns across all T' frames."""
    t, h, w = grid
    keep_spatial = round((1 - ratio) * (h * w))
    return t * keep_spatial


class TestRandomTubeMasking:
    """Verify tube masking keeps exactly the right number of tokens."""

    def test_mask_ratio_preserved(self):
        """Check that exactly T' * round((1 - mask_ratio) * H'*W') tokens are kept."""
        masker = RandomTubeMasking(mask_ratio=0.9)
        device = torch.device("cpu")

        keep_indices, mask_indices = masker(None, BASE_GRID, device)

        num_tokens = BASE_GRID[0] * BASE_GRID[1] * BASE_GRID[2]
        expected_kept = _expected_kept(BASE_GRID, 0.9)
        assert len(keep_indices) == expected_kept, (
            f"expected {expected_kept} kept tokens, got {len(keep_indices)}"
        )
        assert len(mask_indices) == num_tokens - expected_kept

    def test_indices_disjoint_and_complete(self):
        """Check that keep + mask indices partition [0, N)."""
        masker = RandomTubeMasking(mask_ratio=0.9)
        device = torch.device("cpu")

        keep_indices, mask_indices = masker(None, BASE_GRID, device)

        num_tokens = BASE_GRID[0] * BASE_GRID[1] * BASE_GRID[2]

        # Indices must be disjoint
        combined = torch.cat([keep_indices, mask_indices])
        unique = torch.unique(combined)
        assert len(unique) == len(combined), "keep and mask indices overlap"

        # Indices must cover [0, N)
        assert len(unique) == num_tokens, (
            f"indices do not cover full range: got {len(unique)}, expected {num_tokens}"
        )

    def test_is_tube_structured(self):
        """The defining property: each spatial position is kept in ALL frames or none.

        This is what separates true tube masking from random token masking — a
        random mask would fail here because a spatial position could be visible in
        some frames and masked in others.
        """
        masker = RandomTubeMasking(mask_ratio=0.9)
        device = torch.device("cpu")

        keep_indices, mask_indices = masker(None, BASE_GRID, device)

        t, h, w = BASE_GRID
        n_spatial = h * w

        # Spatial position of each kept token = index % n_spatial.
        kept_spatial = torch.unique(keep_indices % n_spatial)
        masked_spatial = torch.unique(mask_indices % n_spatial)

        # A spatial position is exclusively kept or exclusively masked.
        overlap = set(kept_spatial.tolist()) & set(masked_spatial.tolist())
        assert not overlap, f"spatial positions in both kept and masked: {overlap}"

        # Every kept spatial position must appear in all T' frames.
        for s in kept_spatial.tolist():
            frames = torch.tensor([t_i * n_spatial + s for t_i in range(t)])
            assert torch.isin(frames, keep_indices).all(), (
                f"spatial position {s} is not kept in all {t} frames"
            )

    def test_different_mask_ratios(self):
        """Test various mask ratios."""
        device = torch.device("cpu")
        grid = (4, 25, 10)  # 1000 tokens, keeps tube math exact

        for ratio in [0.5, 0.75, 0.9, 0.95]:
            masker = RandomTubeMasking(mask_ratio=ratio)
            keep_indices, mask_indices = masker(None, grid, device)
            expected_kept = _expected_kept(grid, ratio)
            assert len(keep_indices) == expected_kept, (
                f"mask_ratio={ratio}: expected {expected_kept} kept, got {len(keep_indices)}"
            )


class TestMotionGuidedTubeMasking:
    """Verify motion-guided masking picks out the actual high-motion region."""

    def test_masks_highest_motion_spatial_positions(self):
        device = torch.device("cpu")
        grid = (4, 2, 2)  # T'=4, H'=2, W'=2 -> n_spatial=4
        B, C, T_full, H_full, W_full = 2, 3, 8, 32, 32  # patch=16 -> H'=W'=2

        pixel_values = torch.zeros(B, C, T_full, H_full, W_full)
        # Top-left quadrant (spatial position 0 in row-major 2x2) ramps every
        # frame -> large frame-to-frame diff. Every other quadrant is static.
        for t in range(T_full):
            pixel_values[:, :, t, 0:16, 0:16] = float(t)

        masker = MotionGuidedTubeMasking(
            mask_ratio=0.75
        )  # mask 3 of 4 spatial positions
        keep_indices, mask_indices = masker(pixel_values, grid, device)

        n_spatial = 4
        masked_spatial = set((mask_indices % n_spatial).tolist())
        kept_spatial = set((keep_indices % n_spatial).tolist())
        assert 0 in masked_spatial
        assert 0 not in kept_spatial

    def test_indices_disjoint_and_complete(self):
        device = torch.device("cpu")
        grid = (4, 2, 2)
        pixel_values = torch.randn(2, 3, 8, 32, 32)
        masker = MotionGuidedTubeMasking(mask_ratio=0.75)
        keep_indices, mask_indices = masker(pixel_values, grid, device)

        num_tokens = 4 * 2 * 2
        combined = torch.cat([keep_indices, mask_indices])
        unique = torch.unique(combined)
        assert len(unique) == len(combined)
        assert len(unique) == num_tokens
