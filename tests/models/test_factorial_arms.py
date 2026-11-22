"""The four factorial arms must be four different experiments.

The whole motion-blindness factorial rests on (mask_type x target_mode)
producing four distinguishable gradient signals. Nothing verified that. If
`mixed` silently degenerated to `tube` -- a broken coin flip, a masker built
from the wrong config key -- the result would be a two-arm factorial run for
3000 steps across four pools, reported as four arms, with no signal that
anything was wrong.

These are slower than the unit tests because they actually step the optimiser;
that is the point. A config-level assertion cannot see a degenerate arm.
"""

from itertools import combinations

import pytest
import torch
from omegaconf import DictConfig

from stern.models.videomae.mae import VideoMAEForPreTraining
from stern.training.pretrain import _build_masker

ARMS = [
    ("tube", "time_avg"),
    ("tube", "per_frame"),
    ("mixed", "time_avg"),
    ("mixed", "per_frame"),
]
STEPS = 8


def _cfg(mask_type: str, target_mode: str) -> DictConfig:
    return DictConfig(
        {
            "model": {
                "num_frames": 8,
                "image_size": 32,
                "in_channels": 3,
                "tubelet_size": 2,
                "patch_size": 16,
                "embed_dim": 32,
                "depth": 2,
                "num_heads": 4,
                "mlp_ratio": 4.0,
                "dropout": 0.0,
                "decoder_depth": 1,
                "decoder_embed_dim": 16,
                "decoder_num_heads": 4,
                "mask_ratio": 0.75,
                "motion_mask": False,
                "mask_type": mask_type,
                "target_mode": target_mode,
            }
        }
    )


def _moving_clip(seed: int = 0) -> torch.Tensor:
    """A static scene plus real inter-frame change.

    Motion has to exist in the input or every arm optimises the same thing and
    the test would pass vacuously.
    """
    g = torch.Generator().manual_seed(seed)
    still = torch.randn(1, 3, 1, 32, 32, generator=g).repeat(1, 1, 8, 1, 1)
    return still + 0.5 * torch.randn(1, 3, 8, 32, 32, generator=g)


def _train(mask_type: str, target_mode: str, seed: int = 0):
    """Step one arm and return (encoder weights, loss trajectory, n_frame_draws)."""
    cfg = _cfg(mask_type, target_mode)
    torch.manual_seed(seed)
    model = VideoMAEForPreTraining(cfg.model)
    masker = _build_masker(cfg)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    clip = _moving_clip()
    grid = model.encoder.token_grid

    losses, frame_draws = [], 0
    torch.manual_seed(seed + 1000)  # mask draws share a stream across arms
    for _ in range(STEPS):
        keep, mask = masker(clip, grid, torch.device("cpu"), batch_size=clip.shape[0])
        if len({int(i) // (grid[1] * grid[2]) for i in keep[0].tolist()}) < grid[0]:
            frame_draws += 1  # a tube draw keeps every timestep visible
        loss = model.compute_loss(*model(clip, keep, mask))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))

    weights = torch.cat(
        [p.detach().flatten() for p in model.encoder.parameters()]
    ).clone()
    return weights, losses, frame_draws


@pytest.fixture(scope="module")
def trained():
    return {arm: _train(*arm) for arm in ARMS}


class TestTheArmsAreDistinguishable:
    def test_every_pair_of_arms_ends_at_different_encoder_weights(self, trained):
        """The load-bearing assertion: two arms that converge to bit-identical
        encoder weights after 8 steps on identical data are one experiment."""
        for a, b in combinations(ARMS, 2):
            delta = float((trained[a][0] - trained[b][0]).abs().max())
            assert delta > 1e-6, (
                f"{a} and {b} are the same experiment (max|dw|={delta})"
            )

    def test_mixed_actually_draws_frame_masks(self, trained):
        """The arm most at risk of silent degeneration: if the coin flip never
        lands on frame, `mixed` IS `tube` and the mask axis does not exist."""
        for target_mode in ("time_avg", "per_frame"):
            assert trained[("mixed", target_mode)][2] > 0

    def test_tube_never_draws_a_frame_mask(self, trained):
        """The control for the assertion above -- it must be able to be zero."""
        for target_mode in ("time_avg", "per_frame"):
            assert trained[("tube", target_mode)][2] == 0

    def test_every_arm_actually_learns(self, trained):
        """A frozen arm would also be 'distinguishable'. Each must move."""
        for arm in ARMS:
            losses = trained[arm][1]
            assert losses[-1] < losses[0]

    def test_the_target_mode_axis_changes_the_objective_width(self, trained):
        """time_avg and per_frame are not comparable losses -- their targets have
        different widths. Stated here so the factorial table is never read as a
        single ranking across the target_mode axis."""
        avg = VideoMAEForPreTraining(_cfg("tube", "time_avg").model)
        per = VideoMAEForPreTraining(_cfg("tube", "per_frame").model)
        clip = _moving_clip()
        assert (
            per._patchify_targets(clip).shape[-1]
            == 2 * avg._patchify_targets(clip).shape[-1]
        )


class TestTheArmsAreReproducible:
    def test_the_same_arm_and_seed_gives_the_same_weights(self):
        """Distinctness only means anything if a rerun is identical -- otherwise
        the test above would pass on RNG drift alone."""
        a, _, _ = _train("mixed", "per_frame", seed=3)
        b, _, _ = _train("mixed", "per_frame", seed=3)
        assert torch.equal(a, b)
