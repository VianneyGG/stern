import random

import numpy as np
from PIL import Image

from video_encoding.data.augment import (
    GroupAugmentParams,
    apply_group_augment,
    sample_group_augment,
)


def _gradient_frame(offset: int, size: int = 32) -> np.ndarray:
    """A frame whose every pixel is distinguishable by position.

    Solid-color fixtures are useless here: a flipped or differently-cropped
    solid frame is pixel-identical to the original, so they cannot witness
    the property under test. Row/column-varying content can.
    """
    rows = np.arange(size, dtype=np.int32)[:, None]
    cols = np.arange(size, dtype=np.int32)[None, :]
    plane = (rows * 3 + cols * 5 + offset) % 256
    return np.repeat(plane.astype(np.uint8)[:, :, None], 3, axis=2)


def _reference(frame: np.ndarray, params: GroupAugmentParams, image_size: int):
    img = (
        Image.fromarray(frame)
        .crop(params.crop_box)
        .resize((image_size, image_size), Image.BILINEAR)
    )
    if params.flip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return np.array(img)


def test_all_frames_in_a_window_share_one_crop_and_flip():
    """The spec's core property: ONE crop box and ONE flip for the window.

    Per-frame-independent augmentation would inject fake motion. Asserted by
    reconstructing each frame from the *same* params and requiring an exact
    match -- a per-frame redraw of crop or flip fails this.
    """
    params = GroupAugmentParams(crop_box=(3, 7, 27, 31), flip=True)
    frames = [_gradient_frame(i * 40) for i in range(4)]
    out = apply_group_augment(frames, params, image_size=32)

    assert len(out) == 4
    for src, got in zip(frames, out, strict=True):
        assert got.shape == (32, 32, 3)
        np.testing.assert_array_equal(got, _reference(src, params, 32))


def test_augmentation_actually_changes_pixels():
    """A non-identity crop+flip must not be a silent no-op."""
    params = GroupAugmentParams(crop_box=(3, 7, 27, 31), flip=True)
    frame = _gradient_frame(0)
    (out,) = apply_group_augment([frame], params, image_size=32)
    assert not np.array_equal(out, frame)


def test_flip_is_applied_when_set():
    """Isolates the flip: a full-frame crop leaves flip as the only effect."""
    frame = _gradient_frame(0)
    (flipped,) = apply_group_augment(
        [frame], GroupAugmentParams(crop_box=(0, 0, 32, 32), flip=True), image_size=32
    )
    (plain,) = apply_group_augment(
        [frame], GroupAugmentParams(crop_box=(0, 0, 32, 32), flip=False), image_size=32
    )
    np.testing.assert_array_equal(flipped, plain[:, ::-1, :])


def test_different_rng_state_yields_different_params():
    rng_a = random.Random(0)
    rng_b = random.Random(1)
    params_a = sample_group_augment(image_size=32, rng=rng_a)
    params_b = sample_group_augment(image_size=32, rng=rng_b)
    assert params_a != params_b


def test_sampled_scale_comes_from_the_configured_set():
    """Crop sizes must be the paper's scales, not arbitrary rectangles."""
    rng = random.Random(3)
    allowed = {max(1, round(224 * s)) for s in (1.0, 0.875, 0.75, 0.66)}
    seen = set()
    for _ in range(200):
        left, top, right, bottom = sample_group_augment(224, rng).crop_box
        assert right - left == bottom - top, "crop must be square"
        seen.add(right - left)
    assert seen <= allowed
    assert len(seen) > 1, "every draw picked the same scale -- rng not consumed"


def test_crop_box_within_image_bounds():
    rng = random.Random(42)
    for _ in range(50):
        params = sample_group_augment(image_size=224, rng=rng)
        left, top, right, bottom = params.crop_box
        assert 0 <= left < right <= 224
        assert 0 <= top < bottom <= 224
