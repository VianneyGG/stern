"""Unit tests for VideoMAEForPreTraining."""

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from video_encoding.models.videomaev2.mae import VideoMAEForPreTraining
from video_encoding.models.videomaev2.masking import RandomTubeMasking


class TestVideoMAEForPreTraining:
    """Verify MAE encoder+decoder forward pass and reconstruction loss."""

    @pytest.fixture
    def model_cfg(self):
        """Minimal model config for testing."""
        return DictConfig(
            {
                "num_frames": 16,
                "image_size": 224,
                "in_channels": 3,
                "tubelet_size": 2,
                "patch_size": 16,
                "embed_dim": 768,
                "depth": 2,  # small for testing
                "num_heads": 12,
                "mlp_ratio": 4.0,
                "dropout": 0.0,
                "decoder_depth": 2,
                "decoder_embed_dim": 256,
                "decoder_num_heads": 8,
                "mask_ratio": 0.9,
            }
        )

    def test_forward_shape(self, model_cfg):
        """Check output shapes after forward pass."""
        model = VideoMAEForPreTraining(model_cfg)
        model.eval()

        B, C, T, H, W = 2, 3, 16, 224, 224
        pixel_values = torch.randn(B, C, T, H, W)

        grid = (8, 14, 14)  # T', H', W'
        masker = RandomTubeMasking(mask_ratio=0.9)
        keep_indices, mask_indices = masker(pixel_values, grid, torch.device("cpu"))

        with torch.no_grad():
            pred, target = model(pixel_values, keep_indices, mask_indices)

        # Prediction shape: (B, num_masked, 3 * patch_size^2)
        patch_size = 16
        expected_pred_shape = (B, len(mask_indices), 3 * patch_size * patch_size)
        assert pred.shape == expected_pred_shape, (
            f"pred shape mismatch: got {pred.shape}, expected {expected_pred_shape}"
        )

        # Target shape should match pred
        assert target.shape == pred.shape, (
            f"target shape mismatch: got {target.shape}, expected {pred.shape}"
        )

    def test_reconstruction_loss_is_scalar(self, model_cfg):
        """Check that compute_loss returns a scalar."""
        model = VideoMAEForPreTraining(model_cfg)
        model.eval()

        B, num_masked = 2, 157  # ~10% of 1568 tokens
        patch_size = 16
        hidden = 3 * patch_size * patch_size

        pred = torch.randn(B, num_masked, hidden)
        target = torch.randn(B, num_masked, hidden)

        loss = model.compute_loss(pred, target)
        assert loss.ndim == 0, f"loss should be scalar, got shape {loss.shape}"
        assert loss.item() > 0, "loss should be positive"

    def test_loss_decreases_on_overfit_batch(self, model_cfg):
        """Verify loss decreases over optimization steps on a single batch."""
        model = VideoMAEForPreTraining(model_cfg)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        B, C, T, H, W = 1, 3, 16, 224, 224
        pixel_values = torch.randn(B, C, T, H, W)

        grid = (8, 14, 14)  # T', H', W'
        masker = RandomTubeMasking(mask_ratio=0.9)
        keep_indices, mask_indices = masker(pixel_values, grid, torch.device("cpu"))

        losses = []
        for step in range(10):
            optimizer.zero_grad()
            pred, target = model(pixel_values, keep_indices, mask_indices)
            loss = model.compute_loss(pred, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease (not necessarily strictly, but trend down)
        assert losses[-1] < losses[0], (
            f"loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
        )

    def test_encoder_checkpoint_load_shape(self, model_cfg):
        """Verify encoder state_dict can be loaded without shape errors."""
        model = VideoMAEForPreTraining(model_cfg)
        state = model.encoder.state_dict()

        # Reconstruct encoder and load state
        from video_encoding.models.videomaev2.encoder import VideoMAEEncoder

        encoder2 = VideoMAEEncoder(model_cfg)
        encoder2.load_state_dict(state)

        # Quick forward pass to verify loaded weights work
        pixel_values = torch.randn(1, 3, 16, 224, 224)
        with torch.no_grad():
            output = encoder2(pixel_values)
        assert output.shape == (1, 768)

    def test_targets_are_per_patch_normalized(self, model_cfg):
        """Reconstruction targets must be z-scored per patch (official VideoMAE
        recipe) -- mean ~0, std ~1 along the last (pixel) dimension of each
        patch token, not raw normalized pixel values."""
        model = VideoMAEForPreTraining(model_cfg)

        pixel_values = torch.randn(2, 3, 16, 224, 224)
        target_full = model._patchify_targets(pixel_values)

        mu = target_full.mean(dim=-1)
        sigma = target_full.std(dim=-1)
        assert torch.allclose(mu, torch.zeros_like(mu), atol=1e-4)
        assert torch.allclose(sigma, torch.ones_like(sigma), atol=1e-2)

    def test_targets_denormalize_uses_correct_imagenet_constants(self, model_cfg):
        """test_targets_are_per_patch_normalized can't tell a correct
        denormalize from a broken one -- z-scoring erases any per-channel
        constant regardless of correctness. This test pins the actual
        denormalize step against a hand-computed reference using known
        per-channel constant pixel values, independent of the source's own
        internal constants."""
        from video_encoding.models.videomaev2.encoder import (
            _IMAGENET_MEAN,
            _IMAGENET_STD,
        )

        model = VideoMAEForPreTraining(model_cfg)
        B, C, T, H, W = 1, 3, 16, 224, 224
        # Spatially/temporally constant per channel -> every patch token
        # collapses to 3 distinct repeated values (256 R's, 256 G's, 256 B's
        # after patchify, since patch_size=16 -> 16*16=256 per channel block).
        normalized_vals = torch.tensor([0.1, -0.2, 0.5])
        pixel_values = (
            normalized_vals.view(1, 3, 1, 1, 1).expand(B, C, T, H, W).contiguous()
        )

        target_full = model._patchify_targets(pixel_values)

        denorm = normalized_vals * torch.tensor(_IMAGENET_STD) + torch.tensor(
            _IMAGENET_MEAN
        )
        expected_patch = torch.cat([denorm[c].expand(256) for c in range(3)])
        mu = expected_patch.mean()
        sigma = expected_patch.std()
        expected = (expected_patch - mu) / (sigma + 1e-6)

        # Every patch token must match this exact reference pattern (uniform input).
        assert torch.allclose(target_full[0, 0], expected, atol=1e-4)
        assert torch.allclose(target_full[0, -1], expected, atol=1e-4)

    def test_decoder_mask_ratio_subsamples_reconstruction_targets(self, model_cfg):
        """V2 dual masking: decoder_mask_ratio > 0 reconstructs only a random
        subset of the masked tokens, shrinking pred/target along dim 1."""
        cfg = OmegaConf.merge(model_cfg, {"decoder_mask_ratio": 0.5})
        model = VideoMAEForPreTraining(cfg)
        model.eval()

        B, C, T, H, W = 2, 3, 16, 224, 224
        pixel_values = torch.randn(B, C, T, H, W)
        grid = (8, 14, 14)
        masker = RandomTubeMasking(mask_ratio=0.9)
        keep_indices, mask_indices = masker(pixel_values, grid, torch.device("cpu"))
        n_mask = len(mask_indices)
        expected_n_recon = max(1, round(n_mask * 0.5))

        with torch.no_grad():
            pred, target = model(pixel_values, keep_indices, mask_indices)

        assert pred.shape[1] == expected_n_recon
        assert target.shape == pred.shape

    def test_decoder_mask_ratio_zero_reconstructs_every_masked_token(self, model_cfg):
        """decoder_mask_ratio=0.0 (the default) must reconstruct every masked
        token -- this is the existing V1 behavior and must not regress."""
        model = VideoMAEForPreTraining(
            model_cfg
        )  # model_cfg has no decoder_mask_ratio -> defaults to 0.0
        model.eval()

        B, C, T, H, W = 2, 3, 16, 224, 224
        pixel_values = torch.randn(B, C, T, H, W)
        grid = (8, 14, 14)
        masker = RandomTubeMasking(mask_ratio=0.9)
        keep_indices, mask_indices = masker(pixel_values, grid, torch.device("cpu"))

        with torch.no_grad():
            pred, target = model(pixel_values, keep_indices, mask_indices)

        assert pred.shape[1] == len(mask_indices)
