"""target_mode controls whether the reconstruction target keeps intra-tubelet motion."""

import pytest
import torch
from omegaconf import DictConfig

from video_encoding.models.videomaev2.mae import VideoMAEForPreTraining
from video_encoding.models.videomaev2.masking import RandomTubeMasking


def _tiny_cfg(**overrides) -> DictConfig:
    """Small enough to build in well under a second; geometry is what matters."""
    base = {
        "num_frames": 4,
        "image_size": 32,
        "in_channels": 3,
        "tubelet_size": 2,
        "patch_size": 16,
        "embed_dim": 32,
        "depth": 1,
        "num_heads": 4,
        "mlp_ratio": 4.0,
        "dropout": 0.0,
        "decoder_depth": 1,
        "decoder_embed_dim": 16,
        "decoder_num_heads": 4,
        "mask_ratio": 0.5,
    }
    base.update(overrides)
    return DictConfig(base)


class TestPredictHeadWidth:
    def test_time_avg_head_is_one_frame_wide(self):
        model = VideoMAEForPreTraining(_tiny_cfg())
        assert model.predict_head.out_features == 3 * 16 * 16

    def test_per_frame_head_is_tubelet_wide(self):
        model = VideoMAEForPreTraining(_tiny_cfg(target_mode="per_frame"))
        assert model.predict_head.out_features == 2 * 3 * 16 * 16

    def test_unknown_target_mode_raises(self):
        with pytest.raises(ValueError, match="target_mode"):
            VideoMAEForPreTraining(_tiny_cfg(target_mode="banana"))


class TestTargetShape:
    def test_per_frame_target_shape(self):
        model = VideoMAEForPreTraining(_tiny_cfg(target_mode="per_frame"))
        target = model._patchify_targets(torch.randn(1, 3, 4, 32, 32))
        assert target.shape == (1, 2 * 2 * 2, 2 * 3 * 16 * 16)


class TestEquivalence:
    def test_modes_agree_when_tubelet_is_one(self):
        """With tubelet_size=1 there is nothing to average, so the two modes must
        be numerically identical. Pins the reshape and permute against drift."""
        pixel_values = torch.randn(1, 3, 4, 32, 32)
        avg = VideoMAEForPreTraining(_tiny_cfg(tubelet_size=1))
        per = VideoMAEForPreTraining(_tiny_cfg(tubelet_size=1, target_mode="per_frame"))
        torch.testing.assert_close(
            avg._patchify_targets(pixel_values), per._patchify_targets(pixel_values)
        )


class TestMotionIsPreserved:
    def test_per_frame_target_varies_along_the_tubelet(self):
        """The behavioural test. If `.mean(dim=3)` is left in, the per-frame
        slices become identical and this fails — a shape assertion would not."""
        pixel_values = torch.zeros(1, 3, 4, 32, 32)
        pixel_values[:, :, 0::2] = -1.0  # even frames
        pixel_values[:, :, 1::2] = 1.0  # odd frames

        model = VideoMAEForPreTraining(_tiny_cfg(target_mode="per_frame"))
        target = model._patchify_targets(pixel_values)
        B, N, D = target.shape
        per_frame = target.view(B, N, 2, D // 2)
        assert not torch.allclose(per_frame[:, :, 0], per_frame[:, :, 1])

    def test_time_avg_target_discards_that_motion(self):
        """Same input, time_avg mode: the two frames cancel, so nothing about the
        motion survives into the target. This is the deficiency being fixed."""
        motion = torch.zeros(1, 3, 4, 32, 32)
        motion[:, :, 0::2] = -1.0
        motion[:, :, 1::2] = 1.0
        static = torch.zeros(1, 3, 4, 32, 32)

        model = VideoMAEForPreTraining(_tiny_cfg())
        torch.testing.assert_close(
            model._patchify_targets(motion), model._patchify_targets(static)
        )


class TestForwardAndLoss:
    def test_per_frame_forward_shapes_agree(self):
        model = VideoMAEForPreTraining(_tiny_cfg(target_mode="per_frame")).eval()
        keep, mask = RandomTubeMasking(0.5)(None, (2, 2, 2), torch.device("cpu"))
        pred, target = model(torch.randn(1, 3, 4, 32, 32), keep, mask)
        assert pred.shape == target.shape == (1, len(mask), 2 * 3 * 16 * 16)
        assert torch.isfinite(model.compute_loss(pred, target))


class TestCheckpointIncompatibility:
    def test_time_avg_weights_refuse_to_load_into_per_frame(self):
        """predict_head widths differ, so the arms' checkpoints are mutually
        unloadable. It must fail loudly rather than silently reinitialise."""
        avg = VideoMAEForPreTraining(_tiny_cfg())
        per = VideoMAEForPreTraining(_tiny_cfg(target_mode="per_frame"))
        with pytest.raises(RuntimeError, match="size mismatch"):
            per.load_state_dict(avg.state_dict(), strict=False)
