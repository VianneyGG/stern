"""Unit tests for tube masking."""

import torch

from video_encoding.models.videomaev2.masking import (
    MixedTubeFrameMasking,
    MotionGuidedTubeMasking,
    RandomFrameMasking,
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


def _is_tube_structured(keep_indices: torch.Tensor, grid: tuple[int, int, int]) -> bool:
    """True when every spatial column is kept in every frame or in none."""
    t, h, w = grid
    n_spatial = h * w
    counts = torch.zeros(n_spatial, dtype=torch.long)
    for i in keep_indices.tolist():
        counts[i % n_spatial] += 1
    return bool(((counts == 0) | (counts == t)).all())


def _expected_masked_tube(grid: tuple[int, int, int], ratio: float) -> int:
    t, h, w = grid
    return t * (h * w - round((1 - ratio) * (h * w)))


class TestRandomFrameMasking:
    def test_partitions_all_tokens(self):
        keep, mask = RandomFrameMasking(mask_ratio=0.9)(
            None, BASE_GRID, torch.device("cpu")
        )
        n = BASE_GRID[0] * BASE_GRID[1] * BASE_GRID[2]
        assert torch.equal(torch.sort(torch.cat([keep, mask]))[0], torch.arange(n))

    def test_exact_mask_count(self):
        n = BASE_GRID[0] * BASE_GRID[1] * BASE_GRID[2]
        keep, mask = RandomFrameMasking(mask_ratio=0.9)(
            None, BASE_GRID, torch.device("cpu")
        )
        assert len(mask) == round(0.9 * n)
        assert len(keep) == n - round(0.9 * n)

    def test_whole_timesteps_are_hidden(self):
        """The point of this masker: whole timesteps have no visible token.

        The count is T' - min_visible_timesteps, not floor(ratio * T'): the
        floor deliberately caps how much temporal context can be removed. See
        TestFrameMaskingLeavesEnoughTemporalContext.
        """
        t, h, w = BASE_GRID
        keep, _ = RandomFrameMasking(mask_ratio=0.9, min_visible_timesteps=2)(
            None, BASE_GRID, torch.device("cpu")
        )
        visible_per_t = torch.zeros(t, dtype=torch.long)
        for i in keep.tolist():
            visible_per_t[i // (h * w)] += 1
        assert int((visible_per_t == 0).sum()) == t - 2

    def test_deterministic_under_seed(self):
        masker = RandomFrameMasking(mask_ratio=0.9)
        torch.manual_seed(1234)
        a, _ = masker(None, BASE_GRID, torch.device("cpu"))
        torch.manual_seed(1234)
        b, _ = masker(None, BASE_GRID, torch.device("cpu"))
        assert torch.equal(a, b)


class TestMixedTubeFrameMasking:
    def test_both_regimes_occur(self):
        masker = MixedTubeFrameMasking(mask_ratio=0.9, p_frame=0.5)
        torch.manual_seed(0)
        seen_tube = seen_frame = False
        for _ in range(100):
            keep, _ = masker(None, BASE_GRID, torch.device("cpu"))
            if _is_tube_structured(keep, BASE_GRID):
                seen_tube = True
            else:
                seen_frame = True
        assert seen_tube and seen_frame

    def test_tube_draws_keep_tube_structure(self):
        """A mixed draw must be a *valid* draw of whichever regime it picked."""
        masker = MixedTubeFrameMasking(mask_ratio=0.9, p_frame=0.5)
        n = BASE_GRID[0] * BASE_GRID[1] * BASE_GRID[2]
        torch.manual_seed(7)
        for _ in range(50):
            keep, mask = masker(None, BASE_GRID, torch.device("cpu"))
            assert torch.equal(torch.sort(torch.cat([keep, mask]))[0], torch.arange(n))
            assert len(mask) in (round(0.9 * n), _expected_masked_tube(BASE_GRID, 0.9))


def _visible_timesteps(keep, grid) -> int:
    t, h, w = grid
    seen = {int(i) // (h * w) for i in keep.tolist()}
    return len(seen)


class TestFrameMaskingLeavesEnoughTemporalContext:
    """At the production geometry the mask ratio drives whole-timestep removal
    into the clamp: int(0.9 * 8) = 7 of 8 timesteps, leaving a single visible
    timestep. That is not "no guaranteed temporal twin", it is single-frame MAE
    with seven-step extrapolation, and it confounds the factorial's mask arm --
    a worse score could equally mean the encoder was starved of input.
    """

    def test_production_geometry_leaves_the_configured_timesteps(self):
        keep, _ = RandomFrameMasking(mask_ratio=0.9)(
            None, BASE_GRID, torch.device("cpu")
        )
        assert _visible_timesteps(keep, BASE_GRID) == 2

    def test_the_floor_is_configurable(self):
        keep, _ = RandomFrameMasking(mask_ratio=0.9, min_visible_timesteps=3)(
            None, BASE_GRID, torch.device("cpu")
        )
        assert _visible_timesteps(keep, BASE_GRID) == 3

    def test_the_floor_holds_at_mask_ratio_one(self):
        """The degenerate ratio must still leave a usable encoder input."""
        keep, _ = RandomFrameMasking(mask_ratio=1.0)(
            None, BASE_GRID, torch.device("cpu")
        )
        assert _visible_timesteps(keep, BASE_GRID) == 2
        assert len(keep) > 0

    def test_a_floor_wider_than_the_grid_cannot_hide_everything(self):
        keep, _ = RandomFrameMasking(mask_ratio=0.9, min_visible_timesteps=99)(
            None, BASE_GRID, torch.device("cpu")
        )
        assert _visible_timesteps(keep, BASE_GRID) == BASE_GRID[0]

    def test_mixed_forwards_the_floor_to_its_frame_draws(self):
        masker = MixedTubeFrameMasking(
            mask_ratio=0.9, p_frame=1.0, min_visible_timesteps=3
        )
        keep, _ = masker(None, BASE_GRID, torch.device("cpu"))
        assert _visible_timesteps(keep, BASE_GRID) == 3
