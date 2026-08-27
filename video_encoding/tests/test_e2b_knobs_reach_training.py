"""The E2b knobs must reach the *training* loss and stay out of the *val* loss.

Two separate hazards, one file.

1. `mask_type` was added and `mesh_round` silently ignored it (MEMORY.md ->
   factorial-mask-type-trap), so a four-arm factorial ran as two arms for 3000
   steps with nothing reporting a problem. The structural fix since then is that
   `train_one_round` is a thin wrapper over `_train_epoch_ssl` -- there is now
   exactly one training-step implementation, reached by both entrypoints. These
   tests pin that: a knob set on the model must move the loss the shared step
   actually optimises.

2. The inverse hazard for the val loss. `evaluate_val_loss` is a *separate*
   `compute_loss` call site, and its output feeds `save_best`'s all-time-best
   guard and every cross-run comparison. If arm B's val loss picked up the
   visible term it would sit on a different scale from arm A's, so the
   all-time-best guard and the 3-pool comparison would both quietly break.
   Val must stay pure masked MSE under every arm.

See docs/research/2026-08-27-depth-collapse-3pool-prompt.md.
"""

import numpy as np
import pytest
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from video_encoding.data.mesh_shard_dataset import MeshShardIterableDataset
from video_encoding.scripts.build_wds_shards import (
    BalancedShardWriter,
    encode_frames_jpeg,
)
from video_encoding.models.videomaev2.mae import VideoMAEForPreTraining
from video_encoding.models.videomaev2.masking import build_masker
from video_encoding.training.diloco.mesh_round import evaluate_val_loss
from video_encoding.training.pretrain import _train_epoch_ssl

ARM_B = {"visible_loss_weight": 0.5}
ARM_B_WEIGHTED = {
    "visible_loss_weight": 0.5,
    "visible_loss_isolation_weighting": True,
}
ARM_C = {"deep_supervision_layers": [2]}


def _cfg(**overrides) -> DictConfig:
    base = {
        "num_frames": 4,
        "image_size": 64,
        "in_channels": 3,
        "tubelet_size": 2,
        "patch_size": 16,
        "embed_dim": 32,
        "depth": 4,
        "num_heads": 4,
        "mlp_ratio": 2.0,
        "dropout": 0.0,
        "drop_path_rate": 0.0,
        "decoder_depth": 1,
        "decoder_embed_dim": 32,
        "decoder_num_heads": 4,
        "mask_ratio": 0.5,
    }
    base.update(overrides)
    return DictConfig(base)


def _dataset(root):
    """One tiny real WDS shard -- the same fixture shape test_mesh_round.py uses."""
    writer = BalancedShardWriter(root, n_shards=1)
    frames = [np.full((32, 32, 3), i * 40 % 256, dtype=np.uint8) for i in range(8)]
    writer.write("clip0", encode_frames_jpeg(frames))
    writer.close()
    return MeshShardIterableDataset(
        ["shard-000000.tar"],
        mesh_root=root,
        shard_map={},
        subnet="/254",
        num_frames=4,
        image_size=64,
        overlap=0.0,
    )


def _train_loss(tmp_path, **knobs) -> float:
    """Run the shared training step for a couple of micro-steps and return the
    reconstruction loss it reports."""
    torch.manual_seed(0)
    cfg = _cfg(**knobs)
    model = VideoMAEForPreTraining(cfg)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)  # lr=0: weights fixed,
    # so any difference in the reported loss comes from the objective, not from
    # the two runs having drifted to different parameters.
    dataset = _dataset(tmp_path)
    loader = DataLoader(dataset, batch_size=2, num_workers=0)
    torch.manual_seed(0)  # pin the mask draw
    stats = _train_epoch_ssl(
        model,
        loader,
        optimizer,
        torch.device("cpu"),
        build_masker(cfg),
        max_steps=2,
    )
    return float(stats["loss"])


def _val_loss(tmp_path, **knobs) -> float:
    torch.manual_seed(0)
    cfg = _cfg(**knobs)
    model = VideoMAEForPreTraining(cfg)
    dataset = _dataset(tmp_path)
    torch.manual_seed(0)
    return evaluate_val_loss(
        model, dataset, torch.device("cpu"), build_masker(cfg), amp="off"
    )


# --- the knobs must reach the shared training step ------------------------


CONTROL_TRAIN_LOSS = 0.34398273
CONTROL_VAL_LOSS = 0.34398276  # the eval path, at a different accumulation order

# Pinned per-arm values rather than "differs from the control": a refactor that
# shifted every arm together would satisfy an inequality and still mean the
# ablation had stopped measuring the committed model.
#
# ARM_B_WEIGHTED sits 1.3e-4 below ARM_B here. That gap is small *because the
# fixture is small*: image_size=64 gives a 4x4 spatial grid, where isolation
# distances span barely two values. The claim that the weighting discriminates
# is carried by test_mae_e2b_arms.py's measurement on the production
# (8, 14, 14) grid, not by this margin. What this file pins is only that the
# knob reaches the shared training step at all.
EXPECTED_TRAIN_LOSS = {
    "arm_b": 0.51418376,
    "arm_b_weighted": 0.51405442,
    "arm_c": 0.68812066,
}


@pytest.mark.parametrize(
    "knobs", [ARM_B, ARM_B_WEIGHTED, ARM_C], ids=["arm_b", "arm_b_weighted", "arm_c"]
)
def test_knob_changes_the_loss_the_shared_training_step_optimises(
    tmp_path, knobs, request
):
    """Fails if the training call site drops `aux` -- the exact shape of the
    mask_type trap, one layer down."""
    arm = request.node.callspec.id
    control = _train_loss(tmp_path / "a")
    armed = _train_loss(tmp_path / "b", **knobs)

    assert control == pytest.approx(CONTROL_TRAIN_LOSS, abs=1e-7)
    assert armed == pytest.approx(EXPECTED_TRAIN_LOSS[arm], abs=1e-7), (
        f"{knobs} did not produce its expected training loss; the shared step "
        "may be ignoring the auxiliary terms"
    )


def test_control_training_loss_is_reproducible(tmp_path):
    """Guards the tests above: if the harness were nondeterministic, a pinned
    value would be unmaintainable and 'the loss changed' would prove nothing."""
    first = _train_loss(tmp_path / "a")
    second = _train_loss(tmp_path / "b")

    assert first == second
    assert first == pytest.approx(CONTROL_TRAIN_LOSS, abs=1e-7)


# --- ...and must stay out of the val loss ---------------------------------


@pytest.mark.parametrize(
    "knobs", [ARM_B, ARM_B_WEIGHTED, ARM_C], ids=["arm_b", "arm_b_weighted", "arm_c"]
)
def test_val_loss_is_pure_masked_mse_under_every_arm(tmp_path, knobs):
    """val_loss feeds save_best's all-time-best guard and every cross-arm
    comparison. It must measure the same quantity in all three pools."""
    control = _val_loss(tmp_path / "a")
    armed = _val_loss(tmp_path / "b", **knobs)

    assert control == pytest.approx(CONTROL_VAL_LOSS, abs=1e-7)
    assert armed == pytest.approx(CONTROL_VAL_LOSS, abs=1e-7), (
        f"{knobs} changed the val loss; arm A and this arm are no longer "
        "comparable, and save_best's guard changes meaning mid-experiment"
    )
