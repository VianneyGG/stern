"""The masker is selected from config, defaulting to today's behaviour."""

import pytest
from omegaconf import DictConfig

from video_encoding.models.videomaev2.masking import (
    MixedTubeFrameMasking,
    MotionGuidedTubeMasking,
    RandomFrameMasking,
    RandomTubeMasking,
)
from video_encoding.training.pretrain import _build_masker


def _cfg(**model_kwargs) -> DictConfig:
    return DictConfig({"model": {"mask_ratio": 0.9, **model_kwargs}})


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({}, RandomTubeMasking),
        ({"mask_type": "tube"}, RandomTubeMasking),
        ({"mask_type": "frame"}, RandomFrameMasking),
        ({"mask_type": "mixed"}, MixedTubeFrameMasking),
        ({"motion_mask": True}, MotionGuidedTubeMasking),
    ],
)
def test_selects_expected_masker(kwargs, expected):
    assert isinstance(_build_masker(_cfg(**kwargs)), expected)


def test_mask_ratio_is_forwarded():
    assert _build_masker(_cfg(mask_type="mixed")).mask_ratio == 0.9


def test_unknown_mask_type_raises():
    with pytest.raises(ValueError, match="mask_type"):
        _build_masker(_cfg(mask_type="banana"))


def test_motion_mask_wins_over_mask_type():
    """motion_mask predates mask_type; keeping it authoritative means no
    existing config silently changes meaning."""
    assert isinstance(
        _build_masker(_cfg(motion_mask=True, mask_type="frame")),
        MotionGuidedTubeMasking,
    )
