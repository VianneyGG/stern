import torch

from video_encoding.ssl import tube_mask


def test_tube_mask_has_correct_total_length():
    mask = tube_mask(num_patches_per_frame=196, num_tubelets=8, mask_ratio=0.9)
    assert mask.shape == (196 * 8,)
    assert mask.dtype == torch.bool


def test_tube_mask_masks_correct_ratio_per_tubelet():
    mask = tube_mask(num_patches_per_frame=196, num_tubelets=8, mask_ratio=0.9)
    per_tubelet = mask.view(8, 196)
    expected_masked = round(196 * 0.9)
    for row in per_tubelet:
        assert row.sum().item() == expected_masked


def test_tube_mask_repeats_identical_spatial_pattern_across_tubelets():
    mask = tube_mask(num_patches_per_frame=196, num_tubelets=8, mask_ratio=0.9)
    per_tubelet = mask.view(8, 196)
    for row in per_tubelet[1:]:
        assert torch.equal(row, per_tubelet[0])


def test_tube_mask_deterministic_with_fixed_generator():
    gen1 = torch.Generator().manual_seed(42)
    gen2 = torch.Generator().manual_seed(42)

    mask1 = tube_mask(num_patches_per_frame=196, num_tubelets=8, mask_ratio=0.9, generator=gen1)
    mask2 = tube_mask(num_patches_per_frame=196, num_tubelets=8, mask_ratio=0.9, generator=gen2)

    assert torch.equal(mask1, mask2)


def test_tube_mask_varies_without_fixed_seed():
    mask1 = tube_mask(num_patches_per_frame=196, num_tubelets=8, mask_ratio=0.5)
    mask2 = tube_mask(num_patches_per_frame=196, num_tubelets=8, mask_ratio=0.5)

    assert not torch.equal(mask1, mask2)
