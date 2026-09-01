from __future__ import annotations

import torch

from video_encoding.models.levjepa.views import build_views


def test_build_views_shapes():
    clip = torch.randint(0, 256, (16, 224, 224, 3), dtype=torch.uint8)
    global_view, local_views = build_views(
        clip, num_local=4, global_size=224, local_size=96
    )
    assert global_view.shape == (3, 16, 224, 224)
    assert local_views.shape == (4, 3, 16, 96, 96)


def test_build_views_shares_temporal_window():
    """All views must cover the same T frames -- only spatial/photometric
    content may differ, per the design's 'same temporal window' requirement."""
    clip = torch.randint(0, 256, (16, 224, 224, 3), dtype=torch.uint8)
    global_view, local_views = build_views(clip, num_local=2)
    assert global_view.shape[1] == clip.shape[0]
    assert local_views.shape[2] == clip.shape[0]


def test_build_views_local_views_differ_from_each_other():
    torch.manual_seed(0)
    clip = torch.randint(0, 256, (16, 224, 224, 3), dtype=torch.uint8)
    _, local_views = build_views(clip, num_local=4)
    assert not torch.allclose(local_views[0], local_views[1])


def test_build_views_respects_num_local():
    clip = torch.randint(0, 256, (16, 224, 224, 3), dtype=torch.uint8)
    _, local_views = build_views(clip, num_local=6)
    assert local_views.shape[0] == 6


def test_build_views_global_view_is_center_crop_not_random_augmented():
    """The global view must be deterministic given a fixed clip (resize/
    center-crop only) -- no random photometric jitter, unlike local views.
    Calling build_views twice on the same clip must yield the same global
    view even though local views differ run to run."""
    torch.manual_seed(0)
    clip = torch.randint(0, 256, (16, 224, 224, 3), dtype=torch.uint8)
    global_a, _ = build_views(clip, num_local=2)
    torch.manual_seed(1)
    global_b, _ = build_views(clip, num_local=2)
    assert torch.allclose(global_a, global_b, atol=1e-5)
