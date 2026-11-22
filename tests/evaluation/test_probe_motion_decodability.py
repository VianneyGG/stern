"""Is motion information present in the representation at all?"""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from stern.models.videomae.mae import VideoMAEForPreTraining
from stern.evaluation.probes import motion_decodability as pmd
from stern.evaluation.probes.motion_decodability import (
    frame_diff_targets,
    ridge_r2,
)

GRID = (2, 2, 2)


class TestFrameDiffTargets:
    def test_a_static_clip_has_zero_motion_everywhere(self):
        static = torch.ones(1, 3, 4, 32, 32)
        assert float(frame_diff_targets(static, GRID, 16).abs().max()) < 1e-6

    def test_a_moving_region_scores_above_a_still_one(self):
        """Left half flickers, right half is constant."""
        clip = torch.ones(1, 3, 4, 32, 32)
        clip[:, :, 1::2, :, :16] = -1.0
        targets = frame_diff_targets(clip, GRID, 16).view(GRID)
        assert float(targets[..., 0].mean()) > float(targets[..., 1].mean())

    def test_output_is_one_value_per_token(self):
        assert frame_diff_targets(torch.randn(1, 3, 4, 32, 32), GRID, 16).shape == (8,)


class TestRidgeR2:
    def test_a_perfectly_linear_target_is_recovered(self):
        rng = np.random.default_rng(0)
        features = rng.normal(size=(200, 8))
        weights = rng.normal(size=8)
        assert ridge_r2(features, features @ weights, alpha=1e-6) > 0.99

    def test_an_unrelated_target_scores_near_zero_or_below(self):
        rng = np.random.default_rng(1)
        assert ridge_r2(rng.normal(size=(200, 8)), rng.normal(size=200)) < 0.2

    def test_is_deterministic_for_a_seed(self):
        rng = np.random.default_rng(2)
        features, targets = rng.normal(size=(120, 6)), rng.normal(size=120)
        assert ridge_r2(features, targets, seed=4) == ridge_r2(
            features, targets, seed=4
        )

    def test_noise_scores_low_even_when_regressors_rival_samples(self):
        """The regression guard for the degenerate-R^2 bug.

        The earlier positional probe fit 769 regressors against 160 samples in
        sample and reported R^2 = 1.000 for every checkpoint. Only this regime
        exposes that: with n >> d an in-sample fit also scores ~0, so a
        well-conditioned test cannot tell the two apart. Held out, unrelated
        features must not explain unrelated targets no matter how many there are.
        """
        rng = np.random.default_rng(11)
        features = rng.normal(size=(40, 38))
        targets = rng.normal(size=40)
        assert ridge_r2(features, targets, alpha=1e-6) < 0.2


class TestRidgeR2IgnoresTheInterceptPenalty:
    """The bias column must not be shrunk by alpha.

    Frame-difference magnitudes are absolute values, so the probe's targets
    always carry a large positive mean. If the intercept is penalised, the ridge
    cannot reach that mean, predictions collapse toward zero, and R^2 goes
    arbitrarily negative for reasons that have nothing to do with the features.

    None of the tests above could see it: two pass alpha=1e-6 (effectively no
    penalty) and the third accepts any value below 0.2, which -30 satisfies.
    """

    def _linear(self, seed=0, offset=0.0, n=200, d=8):
        rng = np.random.default_rng(seed)
        features = rng.normal(size=(n, d))
        return features, features @ rng.normal(size=d) + offset

    def test_r2_is_invariant_to_a_constant_shift_in_the_target(self):
        features, targets = self._linear()
        centred = ridge_r2(features, targets, alpha=1.0)
        shifted = ridge_r2(features, targets + 100.0, alpha=1.0)
        assert abs(centred - shifted) < 1e-3

    def test_a_strong_penalty_degrades_toward_zero_not_minus_infinity(self):
        """The limit of an over-penalised ridge is 'predict the mean', which is
        R^2 = 0. Reaching the mean requires a free intercept."""
        features, targets = self._linear(offset=50.0)
        assert ridge_r2(features, targets, alpha=1e6) > -0.05

    def test_a_constant_target_is_matched_exactly(self):
        """Nothing to explain and nothing to vary: a free intercept fits it."""
        rng = np.random.default_rng(3)
        features = rng.normal(size=(120, 6))
        targets = np.full(120, 7.5)
        # ss_tot is 0 here, so the guard decides the value; it must not be
        # dominated by an intercept the penalty pushed away from 7.5.
        assert ridge_r2(features, targets, alpha=1.0) > -1e-3


class TestRandomInitIsTheControl:
    """Every checkpoint in this probe is read against random_init, so what
    random_init scores decides what the comparison means.

    A random ViT is close to a random projection of the tubelet pixels, and
    frame-difference magnitude is an absolute value -- a nonlinearity a linear
    readout cannot undo. So the control sits near zero rather than high, and a
    trained checkpoint has real headroom above it. Pinned here so the report is
    read correctly in either direction: near-zero at init is the expected
    baseline, not a broken probe.
    """

    def _random_init_r2(self, alpha: float) -> float:
        from omegaconf import DictConfig

        from stern.models.videomae.mae import VideoMAEForPreTraining
        from stern.evaluation.probes.motion_decodability import _tokens

        cfg = DictConfig(
            {
                "num_frames": 8,
                "image_size": 64,
                "in_channels": 3,
                "tubelet_size": 2,
                "patch_size": 16,
                "embed_dim": 64,
                "depth": 2,
                "num_heads": 4,
                "mlp_ratio": 4.0,
                "dropout": 0.0,
                "decoder_depth": 1,
                "decoder_embed_dim": 16,
                "decoder_num_heads": 4,
                "mask_ratio": 0.9,
            }
        )
        torch.manual_seed(0)
        model = VideoMAEForPreTraining(cfg).eval()
        g = torch.Generator().manual_seed(1)
        feats, targs = [], []
        for _ in range(24):
            still = torch.randn(1, 3, 1, 64, 64, generator=g).repeat(1, 1, 8, 1, 1)
            clip = still + 0.4 * torch.randn(1, 3, 8, 64, 64, generator=g)
            feats.append(_tokens(model, clip).numpy())
            targs.append(frame_diff_targets(clip, model.encoder.token_grid, 16).numpy())
        return ridge_r2(np.concatenate(feats), np.concatenate(targs), alpha=alpha)

    def test_the_control_sits_near_zero(self):
        assert -0.3 < self._random_init_r2(alpha=1.0) < 0.3

    def test_the_control_does_not_move_with_the_penalty_strength(self):
        """If it did, the headline number would be an artefact of alpha rather
        than a property of the representation."""
        weak, strong = self._random_init_r2(1.0), self._random_init_r2(100.0)
        assert abs(weak - strong) < 0.3


class TestFrameDiffTargetsEdges:
    def test_a_batch_is_rejected_rather_than_averaged_across_clips(self):
        """The function returns one vector per token grid, which cannot
        represent more than one clip. It averaged over the batch dimension
        instead -- silently correct at B=1, silently wrong above it."""
        with pytest.raises(ValueError, match="batch"):
            frame_diff_targets(torch.randn(2, 3, 4, 32, 32), GRID, 16)

    def test_the_last_tubelet_reuses_the_final_difference(self):
        """T frames yield T-1 differences, so the last one is duplicated to make
        the tubelet reshape exact. Zero-padding instead would bias the final
        tubelet toward 'no motion'; duplication is the deliberate choice. Pinned
        so it is a decision rather than an accident.
        """
        clip = torch.zeros(1, 3, 4, 32, 32)
        clip[:, :, 3] = 1.0  # only the last frame differs
        targets = frame_diff_targets(clip, GRID, 16).view(GRID)
        # tubelet 1 covers frames 2,3: diffs (2->3)=1 and the duplicated (2->3)=1
        assert float(targets[1].mean()) == pytest.approx(1.0)
        assert float(targets[0].mean()) == pytest.approx(0.0)


class TestLoadAcceptsLegacyCheckpoints:
    """_load must read pool-cars-best-120r, which predates the ViT refactor.

    load_checkpoint_file renames its attention and MLP keys, but the refactor
    also *added* encoder.registers.tokens, which no old checkpoint carries. At
    the default num_registers: 0 that parameter is an empty (1, 0, D) tensor
    that RegisterTokens.prepend/strip pass through untouched, so its absence
    changes no computation and must not abort the probe. Any other mismatch
    still must.
    """

    def _legacy_state(self, model):
        inverse = {
            ".attn.qkv.": ".attn.in_proj_",
            ".attn.proj.": ".attn.out_proj.",
            ".mlp.net.": ".mlp.",
        }
        legacy = {}
        for key, value in model.state_dict().items():
            if key == "encoder.registers.tokens":
                continue
            for new, old in inverse.items():
                key = key.replace(new, old)
            legacy[key] = value
        return legacy

    def test_a_legacy_checkpoint_loads(self, tmp_path):
        cfg = OmegaConf.load("configs/model/videomae_base.yaml")
        model = VideoMAEForPreTraining(cfg)
        path = tmp_path / "legacy.pt"
        torch.save({"model_state_dict": self._legacy_state(model)}, path)

        loaded = pmd._load(path, cfg, "cpu")

        assert loaded.encoder.registers.tokens.numel() == 0

    def test_a_genuinely_wrong_checkpoint_still_raises(self, tmp_path):
        cfg = OmegaConf.load("configs/model/videomae_base.yaml")
        model = VideoMAEForPreTraining(cfg)
        state = self._legacy_state(model)
        state.pop("encoder.blocks.0.attn.in_proj_weight")
        path = tmp_path / "broken.pt"
        torch.save({"model_state_dict": state}, path)

        with pytest.raises(ValueError, match="key mismatch"):
            pmd._load(path, cfg, "cpu")
