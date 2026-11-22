"""Tests for `video_encoding.training.diloco.mesh_round` (Task 2 of the mesh
round-loop-integration plan) plus the `pretrain.py` dispatch branch and
`_train_epoch_ssl`'s new `progress_desc` param.

Builds real, tiny WDS shard fixtures via `build_wds_shards.BalancedShardWriter`
+ `encode_frames_jpeg` (the same helpers `test_build_wds_shards.py` uses) so
`MeshShardIterableDataset.__iter__`'s prefetch/decode path is exercised for
real, not mocked -- `test_mesh_shard_dataset.py` deliberately skips this for
lack of a shard fixture; this file supplies one.
"""

from __future__ import annotations

import json
import math
import socket
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

import video_encoding.training.diloco.mesh_round as mesh_round
from video_encoding.data.mesh_shard_dataset import MeshShardIterableDataset
from video_encoding.models.videomaev2.mae import VideoMAEForPreTraining
from video_encoding.models.videomaev2.masking import (
    MixedTubeFrameMasking,
    MotionGuidedTubeMasking,
    RandomFrameMasking,
    RandomTubeMasking,
)
from video_encoding.scripts.build_wds_shards import (
    BalancedShardWriter,
    encode_frames_jpeg,
)
from video_encoding.training.diloco.mesh_round import (
    evaluate_val_loss,
    mesh_round_lr,
    round_already_merged,
    push_checkpoint_and_wait,
    run_mesh_pretrain_stage,
    train_one_round,
)
from video_encoding.training.pretrain import (
    _scaled_lr,
    _select_stage,
    _train_epoch_ssl,
    run_pretrain_stage,
)

pytestmark = pytest.mark.integration


def _tiny_cfg() -> DictConfig:
    # Same tiny model as test_pretrain_loop.py: image 32 / patch 16 -> 2x2
    # spatial; frames 4 / tubelet 2 -> 2 temporal.
    return DictConfig(
        {
            "num_frames": 4,
            "image_size": 32,
            "in_channels": 3,
            "tubelet_size": 2,
            "patch_size": 16,
            "embed_dim": 32,
            "depth": 1,
            "num_heads": 4,
            "mlp_ratio": 2.0,
            "dropout": 0.0,
            "decoder_depth": 1,
            "decoder_embed_dim": 32,
            "decoder_num_heads": 4,
            "mask_ratio": 0.5,
        }
    )


def _write_shard(root, clip_key: str, n_frames: int, image_size: int = 32) -> str:
    """Write one real, tiny WDS shard with a single clip of `n_frames`
    solid-color JPEG frames. Returns the shard's filename (relative to
    `root`), matching what MeshShardIterableDataset expects in shard_names.
    """
    writer = BalancedShardWriter(root, n_shards=1)
    frames = [
        np.full((image_size, image_size, 3), i % 256, dtype=np.uint8)
        for i in range(n_frames)
    ]
    writer.write(clip_key, encode_frames_jpeg(frames))
    writer.close()
    return "shard-000000.tar"


def _mesh_dataset(root, clip_key: str, n_frames: int, overlap: float = 0.0):
    shard_name = _write_shard(root, clip_key, n_frames)
    return MeshShardIterableDataset(
        [shard_name],
        mesh_root=root,
        shard_map={},
        subnet="/254",
        num_frames=4,
        image_size=32,
        overlap=overlap,
    )


def test_train_one_round_respects_steps_per_round(tmp_path):
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = VideoMAEForPreTraining(_tiny_cfg()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    masker = RandomTubeMasking(mask_ratio=0.5)

    # overlap=0.75 -> step=1 -> n_frames-3 windows; 53 frames -> 50 windows,
    # comfortably more than steps_per_round=5 (batch_size=1 inside
    # train_one_round, so windows == micro-steps here).
    dataset = _mesh_dataset(tmp_path / "mesh", "clip0", n_frames=53, overlap=0.75)

    stats = train_one_round(
        model,
        optimizer,
        device,
        masker,
        dataset,
        steps_per_round=5,
        amp="off",
        accum_steps=1,
        grad_clip=0.0,
        round_num=0,
    )

    assert stats["steps"] == 5


def test_set_round_changes_window_order_with_real_shard(tmp_path):
    """Real shard/decode path (unlike test_mesh_shard_dataset.py's
    resolution-order-only test): each window's frame values are distinct
    (np.full(..., i % 256) per frame index in _write_shard), so a window's
    pixel sum is a reliable identity signature for order comparison."""
    dataset = _mesh_dataset(tmp_path / "mesh", "clip0", n_frames=53, overlap=0.75)

    dataset.set_round(0)
    round0 = [t.sum().item() for t in iter(dataset)]

    dataset.set_round(1)
    round1 = [t.sum().item() for t in iter(dataset)]

    assert sorted(round0) == pytest.approx(sorted(round1))  # same windows...
    assert round0 != pytest.approx(round1)  # ...different order


def test_set_round_is_deterministic_with_real_shard(tmp_path):
    ds_a = _mesh_dataset(tmp_path / "mesh_a", "clip0", n_frames=53, overlap=0.75)
    ds_b = _mesh_dataset(tmp_path / "mesh_b", "clip0", n_frames=53, overlap=0.75)
    ds_a.set_round(3)
    ds_b.set_round(3)

    seq_a = [t.sum().item() for t in iter(ds_a)]
    seq_b = [t.sum().item() for t in iter(ds_b)]

    assert seq_a == pytest.approx(seq_b)


def test_progress_desc_does_not_change_returned_stats():
    device = torch.device("cpu")

    def _run(progress_desc):
        torch.manual_seed(0)
        model = VideoMAEForPreTraining(_tiny_cfg()).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        masker = RandomTubeMasking(mask_ratio=0.5)
        clips = [torch.randn(3, 4, 32, 32) for _ in range(8)]
        loader = DataLoader(clips, batch_size=2, num_workers=0)
        return _train_epoch_ssl(
            model,
            loader,
            optimizer,
            device,
            masker,
            accum_steps=1,
            max_steps=0,
            timing_warmup_steps=0,
            amp="off",
            progress_desc=progress_desc,
        )

    stats_none = _run(None)
    stats_x = _run("round 3")

    assert stats_none["loss"] == pytest.approx(stats_x["loss"])
    assert stats_none["samples"] == stats_x["samples"]
    assert stats_none["steps"] == stats_x["steps"]


def test_evaluate_val_loss_leaves_model_in_train_mode(tmp_path):
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = VideoMAEForPreTraining(_tiny_cfg()).to(device)
    model.train()
    masker = RandomTubeMasking(mask_ratio=0.5)
    val_dataset = _mesh_dataset(tmp_path / "val_mesh", "val_clip0", n_frames=8)

    loss = evaluate_val_loss(model, val_dataset, device, masker, "off", max_windows=10)

    assert model.training is True
    assert math.isfinite(loss)


def test_evaluate_val_loss_uses_no_grad(tmp_path):
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = VideoMAEForPreTraining(_tiny_cfg()).to(device)
    masker = RandomTubeMasking(mask_ratio=0.5)
    val_dataset = _mesh_dataset(tmp_path / "val_mesh", "val_clip0", n_frames=8)

    for p in model.parameters():
        assert p.grad is None

    evaluate_val_loss(model, val_dataset, device, masker, "off", max_windows=10)

    # No backward() ran under torch.no_grad(): every parameter's .grad is
    # still exactly what it was before the call (None -- never populated).
    for p in model.parameters():
        assert p.grad is None


def test_push_checkpoint_and_wait_returns_immediately_when_merge_false(
    tmp_path, monkeypatch
):
    sleep_calls = []
    monkeypatch.setattr(mesh_round.time, "sleep", lambda s: sleep_calls.append(s))

    push_checkpoint_and_wait(
        {"model_state_dict": {}, "n_samples": 1},
        mesh_shared=tmp_path,
        run_id="r1",
        node="nodeA",
        round_num=0,
        merge=False,
    )

    assert sleep_calls == []
    pending = tmp_path / "checkpoints" / "pending" / "r1_0_nodeA.pt"
    assert pending.exists()


def test_push_checkpoint_and_wait_times_out_and_exits(tmp_path, monkeypatch):
    monkeypatch.setattr(mesh_round.time, "sleep", lambda s: None)

    with pytest.raises(SystemExit, match="1"):
        push_checkpoint_and_wait(
            {"model_state_dict": {}, "n_samples": 1},
            mesh_shared=tmp_path,
            run_id="r1",
            node="nodeA",
            round_num=0,
            merge=True,
            sync_timeout_s=0.01,
        )


def test_push_checkpoint_and_wait_succeeds_when_flag_appears(tmp_path, monkeypatch):
    flag_dir = tmp_path / "checkpoints" / "flags"
    flag_dir.mkdir(parents=True)
    (flag_dir / "sync_complete_r1_0.flag").touch()

    def _fail_sleep(_):
        raise AssertionError(
            "sleep should never be called: the flag already exists before "
            "the first check"
        )

    monkeypatch.setattr(mesh_round.time, "sleep", _fail_sleep)

    push_checkpoint_and_wait(
        {"model_state_dict": {}, "n_samples": 1},
        mesh_shared=tmp_path,
        run_id="r1",
        node="nodeA",
        round_num=0,
        merge=True,
        sync_timeout_s=5.0,
    )


def test_push_checkpoint_and_wait_casts_shared_payload_to_fp16(tmp_path):
    payload = {
        "model_state_dict": {"w": torch.ones(2, 2), "b": torch.ones(2)},
        "n_samples": 1,
    }

    push_checkpoint_and_wait(
        payload,
        mesh_shared=tmp_path,
        run_id="r1",
        node="nodeA",
        round_num=0,
        merge=False,
    )

    shared = torch.load(
        tmp_path / "checkpoints" / "pending" / "r1_0_nodeA.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert shared["model_state_dict"]["w"].dtype == torch.float16
    assert shared["model_state_dict"]["b"].dtype == torch.float16
    # Caller's own dict must be untouched: local checkpoint saves (which
    # happen before this call in run_mesh_pretrain_stage) must stay fp32.
    assert payload["model_state_dict"]["w"].dtype == torch.float32


def test_push_checkpoint_and_wait_pushes_delta_not_weights_when_master_sd_given(
    tmp_path,
):
    """Phase 3.2: once a worker has a master snapshot to diff against, it
    pushes master-worker (quantized with error feedback), not the full
    weights -- the naked full-weight fp16 cast biases the small difference
    the merge actually cares about."""
    master_sd = {"w": torch.full((2, 2), 1.0), "b": torch.full((2,), 1.0)}
    payload = {
        "model_state_dict": {
            "w": torch.full((2, 2), 0.75),
            "b": torch.full((2,), 0.75),
        },
        "n_samples": 1,
    }

    push_checkpoint_and_wait(
        payload,
        mesh_shared=tmp_path,
        run_id="r1",
        node="nodeA",
        round_num=0,
        merge=False,
        master_sd=master_sd,
        master_version=3,
        residual_path=tmp_path / "push_residual.pt",
    )

    shared = torch.load(
        tmp_path / "checkpoints" / "pending" / "r1_0_nodeA.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert "model_state_dict" not in shared
    assert shared["master_version"] == 3
    delta = shared["delta_state_dict"]
    assert delta["w"].dtype == torch.float16
    assert torch.allclose(delta["w"].float(), torch.full((2, 2), 0.25), atol=1e-3)
    # Caller's own dict must be untouched -- same guarantee as the
    # full-weight path above.
    assert payload["model_state_dict"]["w"].dtype == torch.float32
    assert "delta_state_dict" not in payload


def test_push_checkpoint_and_wait_keeps_full_weights_when_master_sd_is_none(tmp_path):
    """Round 0 / unmerged solo runs have no master to diff against and must
    keep pushing full weights exactly as before Phase 3.2."""
    payload = {"model_state_dict": {"w": torch.ones(2, 2)}, "n_samples": 1}

    push_checkpoint_and_wait(
        payload,
        mesh_shared=tmp_path,
        run_id="r1",
        node="nodeA",
        round_num=0,
        merge=False,
    )

    shared = torch.load(
        tmp_path / "checkpoints" / "pending" / "r1_0_nodeA.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert "delta_state_dict" not in shared
    assert "master_version" not in shared
    assert shared["model_state_dict"]["w"].dtype == torch.float16


def test_fp16_state_dict_leaves_non_float_buffers_untouched():
    state_dict = {"w": torch.ones(2), "step_count": torch.tensor(5, dtype=torch.int64)}

    cast = mesh_round._fp16_state_dict(state_dict)

    assert cast["w"].dtype == torch.float16
    assert cast["step_count"].dtype == torch.int64


def test_run_mesh_pretrain_stage_cold_start_has_no_master_checkpoint(
    tmp_path, monkeypatch
):
    torch.manual_seed(0)
    # Assignments are keyed by the short hostname (matches production:
    # mesh_round.py normalizes socket.gethostname() the same way before
    # lookup), independent of whether this machine's own hostname happens
    # to include a domain suffix.
    hostname = socket.gethostname().split(".", 1)[0]

    mesh_root = tmp_path / "mesh_root"
    shard_name = _write_shard(mesh_root, "clip0", n_frames=4)  # exactly 1 window

    mesh_shared = tmp_path / "mesh_shared"
    assignments_path = mesh_shared / "metadata" / "shard_assignments_run1.json"
    assignments_path.parent.mkdir(parents=True)
    assignments_path.write_text(json.dumps({hostname: [shard_name]}))

    shard_map_path = mesh_shared / "metadata" / "shard_map.json"
    shard_map_path.write_text(json.dumps({}))

    checkpoint_root = tmp_path / "stern_checkpoints"

    monkeypatch.setenv("MESH_RUN_ID", "run1")
    monkeypatch.setenv("MESH_SHARED", str(mesh_shared))
    monkeypatch.setenv("MESH_ASSIGNMENTS", str(assignments_path))
    monkeypatch.setenv("MESH_SHARD_MAP", str(shard_map_path))
    monkeypatch.setenv("MESH_ROOT", str(mesh_root))
    monkeypatch.setenv("STERN_CHECKPOINT_ROOT", str(checkpoint_root))

    # master.pt intentionally never created -- this is the cold-start case.
    assert not (mesh_shared / "checkpoints" / "run1" / "master.pt").exists()

    cfg = DictConfig(
        {
            "model": _tiny_cfg(),
            "train": {
                "amp": "off",
                "accum_steps": 1,
                "grad_clip": 0.0,
                "weight_decay": 0.0,
                "learning_rate": 1e-3,
            },
            "preprocess": {"window": {"overlap": 0.0}},
            "mesh": {
                "rounds": 1,
                "steps_per_round": 1,
                "merge": False,
                "outer_lr": 0.7,
                "outer_momentum": 0.9,
                "sync_timeout_s": 5.0,
            },
        }
    )

    result = run_mesh_pretrain_stage(cfg, device=torch.device("cpu"))

    assert result["stage"] == "mesh_pretrain"
    assert result["final_round"] == 0
    assert math.isfinite(result["recon_loss"])


def test_run_mesh_pretrain_stage_with_ema_decay_writes_val_loss_ema(
    tmp_path, monkeypatch
):
    """train.ema_decay > 0 must not disturb the round loop, and the local
    checkpoint payload's val_loss_ema must be a finite, real second eval --
    not just a copy of val_loss."""
    torch.manual_seed(0)
    hostname = socket.gethostname().split(".", 1)[0]

    mesh_root = tmp_path / "mesh_root"
    shard_name = _write_shard(mesh_root, "clip0", n_frames=4)
    # evaluate_val_loss reads mesh_root/val/*.tar directly (empty -> nan);
    # seed one so val_loss/val_loss_ema are real numbers to compare.
    _write_shard(mesh_root / "val", "clip0_val", n_frames=4)

    mesh_shared = tmp_path / "mesh_shared"
    assignments_path = mesh_shared / "metadata" / "shard_assignments_run1.json"
    assignments_path.parent.mkdir(parents=True)
    assignments_path.write_text(json.dumps({hostname: [shard_name]}))

    shard_map_path = mesh_shared / "metadata" / "shard_map.json"
    shard_map_path.write_text(json.dumps({}))

    checkpoint_root = tmp_path / "stern_checkpoints"

    monkeypatch.setenv("MESH_RUN_ID", "run1")
    monkeypatch.setenv("MESH_SHARED", str(mesh_shared))
    monkeypatch.setenv("MESH_ASSIGNMENTS", str(assignments_path))
    monkeypatch.setenv("MESH_SHARD_MAP", str(shard_map_path))
    monkeypatch.setenv("MESH_ROOT", str(mesh_root))
    monkeypatch.setenv("STERN_CHECKPOINT_ROOT", str(checkpoint_root))

    cfg = DictConfig(
        {
            "model": _tiny_cfg(),
            "train": {
                "amp": "off",
                "accum_steps": 1,
                "grad_clip": 0.0,
                "weight_decay": 0.0,
                "learning_rate": 1e-3,
                "ema_decay": 0.9,
            },
            "preprocess": {"window": {"overlap": 0.0}},
            "mesh": {
                "rounds": 1,
                "steps_per_round": 1,
                "merge": False,
                "outer_lr": 0.7,
                "outer_momentum": 0.9,
                "sync_timeout_s": 5.0,
            },
        }
    )

    result = run_mesh_pretrain_stage(cfg, device=torch.device("cpu"))
    assert result["final_round"] == 0

    payload = torch.load(
        checkpoint_root / "run1.pt", map_location="cpu", weights_only=False
    )
    assert math.isfinite(payload["val_loss_ema"])
    # A freshly-initialized shadow (decay=0.9) after exactly 1 optimizer step
    # has moved only 10% of the way -- it should differ from the raw val_loss
    # rather than accidentally aliasing it.
    assert payload["val_loss_ema"] != payload["val_loss"]


def test_round_loop_stops_at_boundary_when_stop_file_present(tmp_path, monkeypatch):
    hostname = socket.gethostname().split(".", 1)[0]

    mesh_root = tmp_path / "mesh_root"
    shard_name = _write_shard(mesh_root, "clip0", n_frames=4)  # exactly 1 window

    mesh_shared = tmp_path / "mesh_shared"
    assignments_path = mesh_shared / "metadata" / "shard_assignments_run1.json"
    assignments_path.parent.mkdir(parents=True)
    assignments_path.write_text(json.dumps({hostname: [shard_name]}))

    shard_map_path = mesh_shared / "metadata" / "shard_map.json"
    shard_map_path.write_text(json.dumps({}))

    checkpoint_root = tmp_path / "stern_checkpoints"

    stop_file = tmp_path / "stop"
    stop_file.touch()

    monkeypatch.setenv("MESH_RUN_ID", "run1")
    monkeypatch.setenv("MESH_SHARED", str(mesh_shared))
    monkeypatch.setenv("MESH_ASSIGNMENTS", str(assignments_path))
    monkeypatch.setenv("MESH_SHARD_MAP", str(shard_map_path))
    monkeypatch.setenv("MESH_ROOT", str(mesh_root))
    monkeypatch.setenv("STERN_CHECKPOINT_ROOT", str(checkpoint_root))
    monkeypatch.setenv("MESH_STOP_FILE", str(stop_file))

    cfg = DictConfig(
        {
            "model": _tiny_cfg(),
            "train": {
                "amp": "off",
                "accum_steps": 1,
                "grad_clip": 0.0,
                "weight_decay": 0.0,
                "learning_rate": 1e-3,
            },
            "preprocess": {"window": {"overlap": 0.0}},
            "mesh": {
                "rounds": 5,
                "steps_per_round": 1,
                "merge": False,
                "outer_lr": 0.7,
                "outer_momentum": 0.9,
                "sync_timeout_s": 5.0,
            },
        }
    )

    result = run_mesh_pretrain_stage(cfg, device=torch.device("cpu"))

    assert result["stage"] == "mesh_pretrain"
    assert result["final_round"] == -1


def test_shared_pending_payload_drops_optimizer_state_local_keeps_it(
    tmp_path, monkeypatch
):
    torch.manual_seed(0)
    hostname = socket.gethostname().split(".", 1)[0]

    mesh_root = tmp_path / "mesh_root"
    shard_name = _write_shard(mesh_root, "clip0", n_frames=4)  # exactly 1 window

    mesh_shared = tmp_path / "mesh_shared"
    assignments_path = mesh_shared / "metadata" / "shard_assignments_run1.json"
    assignments_path.parent.mkdir(parents=True)
    assignments_path.write_text(json.dumps({hostname: [shard_name]}))

    shard_map_path = mesh_shared / "metadata" / "shard_map.json"
    shard_map_path.write_text(json.dumps({}))

    checkpoint_root = tmp_path / "stern_checkpoints"

    monkeypatch.setenv("MESH_RUN_ID", "run1")
    monkeypatch.setenv("MESH_SHARED", str(mesh_shared))
    monkeypatch.setenv("MESH_ASSIGNMENTS", str(assignments_path))
    monkeypatch.setenv("MESH_SHARD_MAP", str(shard_map_path))
    monkeypatch.setenv("MESH_ROOT", str(mesh_root))
    monkeypatch.setenv("STERN_CHECKPOINT_ROOT", str(checkpoint_root))

    cfg = DictConfig(
        {
            "model": _tiny_cfg(),
            "train": {
                "amp": "off",
                "accum_steps": 1,
                "grad_clip": 0.0,
                "weight_decay": 0.0,
                "learning_rate": 1e-3,
            },
            "preprocess": {"window": {"overlap": 0.0}},
            "mesh": {
                "rounds": 1,
                "steps_per_round": 1,
                "merge": False,
                "outer_lr": 0.7,
                "outer_momentum": 0.9,
                "sync_timeout_s": 5.0,
            },
        }
    )

    run_mesh_pretrain_stage(cfg, device=torch.device("cpu"))

    local_payload = torch.load(
        checkpoint_root / "run1.pt", map_location="cpu", weights_only=False
    )
    assert "optimizer_state_dict" in local_payload
    assert "optimizer_name" in local_payload

    shared_payload = torch.load(
        mesh_shared / "checkpoints" / "pending" / f"run1_0_{hostname}.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert "optimizer_state_dict" not in shared_payload
    assert "optimizer_name" not in shared_payload
    assert shared_payload.keys() >= {
        "model_state_dict",
        "n_samples",
        "val_loss",
        "round",
        "samples_per_s",
        "outer_lr",
        "outer_momentum",
        "epoch",
        "best_val_metric",
        "model_name",
    }


def test_run_mesh_pretrain_stage_warm_starts_round_zero_from_existing_master(
    tmp_path, monkeypatch
):
    """A worker relaunched after an interruption must not discard a
    master.pt that already reflects prior mesh rounds. Round 0 is not
    special-cased for "no merge partner yet" (that's cfg.mesh.merge=False,
    covered by the cold-start test above) -- it is simply the first round
    of *this process*, which may or may not be the first round of the run.
    """
    torch.manual_seed(0)
    # Assignments are keyed by the short hostname (matches production:
    # mesh_round.py normalizes socket.gethostname() the same way before
    # lookup), independent of whether this machine's own hostname happens
    # to include a domain suffix.
    hostname = socket.gethostname().split(".", 1)[0]

    mesh_root = tmp_path / "mesh_root"
    shard_name = _write_shard(mesh_root, "clip0", n_frames=4)

    mesh_shared = tmp_path / "mesh_shared"
    assignments_path = mesh_shared / "metadata" / "shard_assignments_run1.json"
    assignments_path.parent.mkdir(parents=True)
    assignments_path.write_text(json.dumps({hostname: [shard_name]}))

    shard_map_path = mesh_shared / "metadata" / "shard_map.json"
    shard_map_path.write_text(json.dumps({}))

    checkpoint_root = tmp_path / "stern_checkpoints"

    monkeypatch.setenv("MESH_RUN_ID", "run1")
    monkeypatch.setenv("MESH_SHARED", str(mesh_shared))
    monkeypatch.setenv("MESH_ASSIGNMENTS", str(assignments_path))
    monkeypatch.setenv("MESH_SHARD_MAP", str(shard_map_path))
    monkeypatch.setenv("MESH_ROOT", str(mesh_root))
    monkeypatch.setenv("STERN_CHECKPOINT_ROOT", str(checkpoint_root))

    cfg = DictConfig(
        {
            "model": _tiny_cfg(),
            "train": {
                "amp": "off",
                "accum_steps": 1,
                "grad_clip": 0.0,
                "weight_decay": 0.0,
                "learning_rate": 1e-3,
            },
            "preprocess": {"window": {"overlap": 0.0}},
            "mesh": {
                "rounds": 1,
                "steps_per_round": 1,
                "merge": True,
                "outer_lr": 0.7,
                "outer_momentum": 0.9,
                "sync_timeout_s": 5.0,
            },
        }
    )

    # Simulate a master.pt already advanced past round 0 by prior mesh
    # rounds this process never saw -- a marker value no fresh init or
    # training step would produce by chance.
    marker_model = VideoMAEForPreTraining(_tiny_cfg())
    with torch.no_grad():
        for p in marker_model.parameters():
            p.fill_(7.0)
    master_dir = mesh_shared / "checkpoints" / "run1"
    master_dir.mkdir(parents=True)
    torch.save(
        {"model_state_dict": marker_model.state_dict()},
        master_dir / "master.pt",
    )

    # Stub out real training/merge-wait so the only thing under test is
    # whether round 0 picked up the marker weights before training began.
    captured_first_param = {}

    def _fake_train_one_round(model, *args, **kwargs):
        if not captured_first_param:
            captured_first_param["value"] = next(model.parameters()).flatten()[0].item()
        return {
            "loss": 0.0,
            "samples": 1,
            "steps": 1,
            "samples_per_s_per_gpu": 1.0,
        }

    monkeypatch.setattr(mesh_round, "train_one_round", _fake_train_one_round)
    monkeypatch.setattr(
        mesh_round,
        "push_checkpoint_and_wait",
        # Returns the per-round I/O stats the caller folds into its mesh_io log.
        lambda *args, **kwargs: {
            "pending_push_s": 0.0,
            "pending_bytes": 0,
            "flag_wait_s": 0.0,
        },
    )

    run_mesh_pretrain_stage(cfg, device=torch.device("cpu"))

    assert captured_first_param["value"] == pytest.approx(7.0)


def test_round_loop_pushes_delta_against_the_master_snapshot_it_loaded(
    tmp_path, monkeypatch
):
    """Phase 3.2: once a round starts from an existing master.pt, the round
    loop must hand push_checkpoint_and_wait that exact snapshot (upcast to
    fp32) and its version tag -- not recompute or skip either."""
    torch.manual_seed(0)
    hostname = socket.gethostname().split(".", 1)[0]

    mesh_root = tmp_path / "mesh_root"
    shard_name = _write_shard(mesh_root, "clip0", n_frames=4)

    mesh_shared = tmp_path / "mesh_shared"
    assignments_path = mesh_shared / "metadata" / "shard_assignments_run1.json"
    assignments_path.parent.mkdir(parents=True)
    assignments_path.write_text(json.dumps({hostname: [shard_name]}))
    shard_map_path = mesh_shared / "metadata" / "shard_map.json"
    shard_map_path.write_text(json.dumps({}))

    seed_model = VideoMAEForPreTraining(_tiny_cfg())
    with torch.no_grad():
        for p in seed_model.parameters():
            p.fill_(7.0)
    master_path = mesh_shared / "checkpoints" / "run1" / "master.pt"
    master_path.parent.mkdir(parents=True)
    torch.save(
        {"model_state_dict": seed_model.state_dict(), "master_version": 5},
        master_path,
    )

    checkpoint_root = tmp_path / "stern_checkpoints"
    monkeypatch.setenv("MESH_RUN_ID", "run1")
    monkeypatch.setenv("MESH_SHARED", str(mesh_shared))
    monkeypatch.setenv("MESH_ASSIGNMENTS", str(assignments_path))
    monkeypatch.setenv("MESH_SHARD_MAP", str(shard_map_path))
    monkeypatch.setenv("MESH_ROOT", str(mesh_root))
    monkeypatch.setenv("STERN_CHECKPOINT_ROOT", str(checkpoint_root))

    cfg = DictConfig(
        {
            "model": _tiny_cfg(),
            "train": {
                "amp": "off",
                "accum_steps": 1,
                "grad_clip": 0.0,
                "weight_decay": 0.0,
                "learning_rate": 1e-3,
            },
            "preprocess": {"window": {"overlap": 0.0}},
            "mesh": {
                "rounds": 1,
                "steps_per_round": 1,
                "merge": True,
                "outer_lr": 0.7,
                "outer_momentum": 0.9,
                "sync_timeout_s": 5.0,
            },
        }
    )

    captured = {}

    def _fake_push(payload, **kwargs):
        captured.update(kwargs)
        return {"pending_push_s": 0.0, "pending_bytes": 0, "flag_wait_s": 0.0}

    monkeypatch.setattr(mesh_round, "push_checkpoint_and_wait", _fake_push)

    run_mesh_pretrain_stage(cfg, device=torch.device("cpu"))

    assert captured["master_version"] == 5
    assert captured["master_sd"] is not None
    for tensor in captured["master_sd"].values():
        assert tensor.dtype == torch.float32
        assert torch.equal(tensor, torch.full_like(tensor, 7.0))


def test_round_loop_omits_master_sd_on_a_cold_start(tmp_path, monkeypatch):
    """Round 0 with no master.pt on disk has nothing to diff against --
    push_checkpoint_and_wait must fall back to its pre-3.2 full-weight path."""
    torch.manual_seed(0)
    hostname = socket.gethostname().split(".", 1)[0]

    mesh_root = tmp_path / "mesh_root"
    shard_name = _write_shard(mesh_root, "clip0", n_frames=4)

    mesh_shared = tmp_path / "mesh_shared"
    assignments_path = mesh_shared / "metadata" / "shard_assignments_run1.json"
    assignments_path.parent.mkdir(parents=True)
    assignments_path.write_text(json.dumps({hostname: [shard_name]}))
    shard_map_path = mesh_shared / "metadata" / "shard_map.json"
    shard_map_path.write_text(json.dumps({}))

    checkpoint_root = tmp_path / "stern_checkpoints"
    monkeypatch.setenv("MESH_RUN_ID", "run1")
    monkeypatch.setenv("MESH_SHARED", str(mesh_shared))
    monkeypatch.setenv("MESH_ASSIGNMENTS", str(assignments_path))
    monkeypatch.setenv("MESH_SHARD_MAP", str(shard_map_path))
    monkeypatch.setenv("MESH_ROOT", str(mesh_root))
    monkeypatch.setenv("STERN_CHECKPOINT_ROOT", str(checkpoint_root))

    cfg = DictConfig(
        {
            "model": _tiny_cfg(),
            "train": {
                "amp": "off",
                "accum_steps": 1,
                "grad_clip": 0.0,
                "weight_decay": 0.0,
                "learning_rate": 1e-3,
            },
            "preprocess": {"window": {"overlap": 0.0}},
            "mesh": {
                "rounds": 1,
                "steps_per_round": 1,
                "merge": True,
                "outer_lr": 0.7,
                "outer_momentum": 0.9,
                "sync_timeout_s": 5.0,
            },
        }
    )

    captured = {}

    def _fake_push(payload, **kwargs):
        captured.update(kwargs)
        return {"pending_push_s": 0.0, "pending_bytes": 0, "flag_wait_s": 0.0}

    monkeypatch.setattr(mesh_round, "push_checkpoint_and_wait", _fake_push)

    run_mesh_pretrain_stage(cfg, device=torch.device("cpu"))

    assert captured["master_sd"] is None
    assert captured["master_version"] is None


def test_late_joiner_loads_master_before_first_local_round(tmp_path, monkeypatch):
    torch.manual_seed(0)
    hostname = socket.gethostname().split(".", 1)[0]

    mesh_root = tmp_path / "mesh_root"
    shard_name = _write_shard(mesh_root, "clip0", n_frames=4)

    mesh_shared = tmp_path / "mesh_shared"
    assignments_path = mesh_shared / "metadata" / "shard_assignments_run1.json"
    assignments_path.parent.mkdir(parents=True)
    assignments_path.write_text(json.dumps({hostname: [shard_name]}))
    shard_map_path = mesh_shared / "metadata" / "shard_map.json"
    shard_map_path.write_text(json.dumps({}))

    # Pre-seed master.pt (the merge that already happened for round 0/1 on
    # the rest of the mesh) and the round-2 sync flag, so
    # push_checkpoint_and_wait doesn't block this test.
    from video_encoding.training import checkpoint as ckpt_io

    master_path = mesh_shared / "checkpoints" / "run1" / "master.pt"
    master_path.parent.mkdir(parents=True)
    seed_model = VideoMAEForPreTraining(_tiny_cfg())
    ckpt_io.save(master_path, {"model_state_dict": seed_model.state_dict()})
    flags_dir = mesh_shared / "checkpoints" / "flags"
    flags_dir.mkdir(parents=True)
    (flags_dir / "sync_complete_run1_2.flag").touch()

    calls = []
    real_load = mesh_round.torch.load

    def spy_load(path, *a, **kw):
        calls.append(Path(path))
        return real_load(path, *a, **kw)

    monkeypatch.setattr(mesh_round.torch, "load", spy_load)

    checkpoint_root = tmp_path / "stern_checkpoints"
    monkeypatch.setenv("MESH_RUN_ID", "run1")
    monkeypatch.setenv("MESH_SHARED", str(mesh_shared))
    monkeypatch.setenv("MESH_ASSIGNMENTS", str(assignments_path))
    monkeypatch.setenv("MESH_SHARD_MAP", str(shard_map_path))
    monkeypatch.setenv("MESH_ROOT", str(mesh_root))
    monkeypatch.setenv("STERN_CHECKPOINT_ROOT", str(checkpoint_root))

    cfg = DictConfig(
        {
            "model": _tiny_cfg(),
            "train": {
                "amp": "off",
                "accum_steps": 1,
                "grad_clip": 0.0,
                "weight_decay": 0.0,
                "learning_rate": 1e-3,
            },
            "preprocess": {"window": {"overlap": 0.0}},
            "mesh": {
                "rounds": 3,
                "start_round": 2,
                "steps_per_round": 1,
                "merge": True,
                "outer_lr": 0.7,
                "outer_momentum": 0.9,
                "sync_timeout_s": 5.0,
            },
        }
    )

    result = run_mesh_pretrain_stage(cfg, device=torch.device("cpu"))

    assert result["final_round"] == 2
    assert calls.count(master_path) == 1


def test_run_mesh_pretrain_stage_strips_fqdn_before_shard_lookup(tmp_path, monkeypatch):
    """shard_assignments_*.json (written by seed_mesh.py/assignment.py) is
    keyed by short hostnames ('node008'), matching NODE_SUBNETS and the
    per-node checkpoint naming convention. socket.gethostname() returns
    whatever the node's NSS/DNS config reports, which can be the FQDN
    ('node008.polytechnique.fr') depending on cluster configuration --
    observed in the wild on 2026-07-22 when every mesh worker crashed with
    KeyError after a cluster-wide config change made gethostname() start
    returning FQDNs. The lookup must be robust to either form.
    """
    torch.manual_seed(0)
    monkeypatch.setattr(socket, "gethostname", lambda: "node008.polytechnique.fr")

    mesh_root = tmp_path / "mesh_root"
    shard_name = _write_shard(mesh_root, "clip0", n_frames=4)

    mesh_shared = tmp_path / "mesh_shared"
    assignments_path = mesh_shared / "metadata" / "shard_assignments_run1.json"
    assignments_path.parent.mkdir(parents=True)
    # Keyed by the short hostname, as seed_mesh.py actually writes it.
    assignments_path.write_text(json.dumps({"node008": [shard_name]}))

    shard_map_path = mesh_shared / "metadata" / "shard_map.json"
    shard_map_path.write_text(json.dumps({}))

    checkpoint_root = tmp_path / "stern_checkpoints"

    monkeypatch.setenv("MESH_RUN_ID", "run1")
    monkeypatch.setenv("MESH_SHARED", str(mesh_shared))
    monkeypatch.setenv("MESH_ASSIGNMENTS", str(assignments_path))
    monkeypatch.setenv("MESH_SHARD_MAP", str(shard_map_path))
    monkeypatch.setenv("MESH_ROOT", str(mesh_root))
    monkeypatch.setenv("STERN_CHECKPOINT_ROOT", str(checkpoint_root))

    cfg = DictConfig(
        {
            "model": _tiny_cfg(),
            "train": {
                "amp": "off",
                "accum_steps": 1,
                "grad_clip": 0.0,
                "weight_decay": 0.0,
                "learning_rate": 1e-3,
            },
            "preprocess": {"window": {"overlap": 0.0}},
            "mesh": {
                "rounds": 1,
                "steps_per_round": 1,
                "merge": False,
                "outer_lr": 0.7,
                "outer_momentum": 0.9,
                "sync_timeout_s": 5.0,
            },
        }
    )

    result = run_mesh_pretrain_stage(cfg, device=torch.device("cpu"))

    assert result["stage"] == "mesh_pretrain"
    assert result["final_round"] == 0


def test_mesh_dispatch_branch_routes_on_env_var(monkeypatch):
    cfg = DictConfig({})
    device = torch.device("cpu")

    monkeypatch.setenv("MESH_RUN_ID", "x")
    stage = _select_stage(cfg, rank=0, world_size=1, local_rank=0, device=device)
    assert stage.func is run_mesh_pretrain_stage

    monkeypatch.delenv("MESH_RUN_ID")
    stage = _select_stage(cfg, rank=0, world_size=1, local_rank=0, device=device)
    assert stage.func is run_pretrain_stage


def test_resolve_subnet_uses_real_per_host_mapping_not_a_single_constant():
    """Review finding 1: a single yaml-wide subnet constant would report
    every node as the same subnet regardless of which physical host it runs
    on. node001 (fish tier) and node024 (birds tier) are real hostnames
    from seed_mesh.py's NODE_SUBNETS roster with genuinely different real
    subnets -- this must resolve them differently, not to one hardcoded
    value."""
    cfg = DictConfig({"mesh": {"subnet": "/999"}})

    fish_subnet = mesh_round._resolve_subnet("node001", cfg)
    birds_subnet = mesh_round._resolve_subnet("node024", cfg)

    assert fish_subnet == "/254"
    assert birds_subnet == "/253"
    assert fish_subnet != birds_subnet


def test_resolve_subnet_falls_back_to_cfg_for_unknown_hostname():
    cfg = DictConfig({"mesh": {"subnet": "/999"}})

    assert mesh_round._resolve_subnet("not-a-real-cluster-host", cfg) == "/999"


def test_scaled_lr_uses_local_batch_override_not_train_batch_size():
    """Review finding 2: the mesh path's real per-optimizer-step batch is
    _MESH_BATCH_SIZE (1), not cfg.train.batch_size (16, from ssl.yaml,
    never overridden by videomae_mesh_ssl.yaml). Passing local_batch must
    change the resulting LR; omitting it must reproduce the exact existing
    (pre-fix) behavior for every non-mesh caller."""
    cfg = DictConfig(
        {
            "train": {
                "base_lr": 1.5e-4,
                "batch_size": 16,
                "accum_steps": 1,
                "lr_mult": 1.0,
            }
        }
    )

    lr_unmodified_callers = _scaled_lr(cfg, world_size=1)
    lr_mesh = _scaled_lr(cfg, world_size=1, local_batch=mesh_round._MESH_BATCH_SIZE)

    assert lr_unmodified_callers == pytest.approx(1.5e-4 * 16 / 256.0)
    assert lr_mesh == pytest.approx(1.5e-4 * 1 / 256.0)
    assert lr_mesh != pytest.approx(lr_unmodified_callers)


def test_push_checkpoint_and_wait_reports_push_and_barrier_time_separately(
    tmp_path, monkeypatch
):
    """The two costs have to be distinguishable: only pending_push_s shrinks
    when the payload shrinks, while flag_wait_s is barrier time that a
    smaller payload cannot fix. Conflating them makes any NFS optimization
    unfalsifiable."""
    mesh_shared = tmp_path / "shared"
    flag_dir = mesh_shared / "checkpoints" / "flags"
    flag_dir.mkdir(parents=True)
    (flag_dir / "sync_complete_run1_0.flag").touch()

    monkeypatch.setattr(mesh_round.time, "sleep", lambda _s: None)

    stats = push_checkpoint_and_wait(
        {"model_state_dict": {"w": torch.ones(64, 64)}, "n_samples": 1},
        mesh_shared=mesh_shared,
        run_id="run1",
        node="node001",
        round_num=0,
        merge=True,
    )

    assert set(stats) == {"pending_push_s", "pending_bytes", "flag_wait_s", "pushed"}
    assert stats["pending_bytes"] > 0
    assert stats["pending_push_s"] >= 0.0
    assert stats["flag_wait_s"] >= 0.0
    assert stats["pushed"] is True


def test_push_checkpoint_and_wait_reports_bytes_when_not_merging(tmp_path):
    mesh_shared = tmp_path / "shared"

    stats = push_checkpoint_and_wait(
        {"model_state_dict": {"w": torch.ones(32, 32)}, "n_samples": 1},
        mesh_shared=mesh_shared,
        run_id="run1",
        node="node001",
        round_num=0,
        merge=False,
    )

    assert stats["pending_bytes"] > 0
    assert stats["flag_wait_s"] == 0.0
    assert stats["pushed"] is True


def test_push_checkpoint_and_wait_returns_pushed_false_without_raising_when_save_always_fails(
    tmp_path, monkeypatch
):
    """0.1: a node that cannot write NFS must skip contributing this round,
    not die -- and the barrier wait must never start for a push that never
    landed."""
    monkeypatch.setattr(mesh_round.ckpt_io, "try_save", lambda *a, **kw: False)
    sleep_calls = []
    monkeypatch.setattr(mesh_round.time, "sleep", lambda s: sleep_calls.append(s))

    stats = push_checkpoint_and_wait(
        {"model_state_dict": {}, "n_samples": 1},
        mesh_shared=tmp_path,
        run_id="r1",
        node="nodeA",
        round_num=0,
        merge=True,
        sync_timeout_s=5.0,
    )

    assert stats["pushed"] is False
    assert stats["flag_wait_s"] == 0.0
    assert sleep_calls == []


def test_run_mesh_pretrain_stage_writes_one_round_metrics_line(tmp_path, monkeypatch):
    """0c: per-round loss otherwise exists only in worker stdout, scattered
    across one SLURM job per node per run -- unworkable for an 8-run sweep."""
    torch.manual_seed(0)
    hostname = socket.gethostname().split(".", 1)[0]

    mesh_root = tmp_path / "mesh_root"
    shard_name = _write_shard(mesh_root, "clip0", n_frames=4)  # exactly 1 window

    mesh_shared = tmp_path / "mesh_shared"
    assignments_path = mesh_shared / "metadata" / "shard_assignments_run1.json"
    assignments_path.parent.mkdir(parents=True)
    assignments_path.write_text(json.dumps({hostname: [shard_name]}))

    shard_map_path = mesh_shared / "metadata" / "shard_map.json"
    shard_map_path.write_text(json.dumps({}))

    monkeypatch.setenv("MESH_RUN_ID", "run1")
    monkeypatch.setenv("MESH_SHARED", str(mesh_shared))
    monkeypatch.setenv("MESH_ASSIGNMENTS", str(assignments_path))
    monkeypatch.setenv("MESH_SHARD_MAP", str(shard_map_path))
    monkeypatch.setenv("MESH_ROOT", str(mesh_root))
    monkeypatch.setenv("STERN_CHECKPOINT_ROOT", str(tmp_path / "stern_checkpoints"))

    cfg = DictConfig(
        {
            "model": _tiny_cfg(),
            "train": {
                "amp": "off",
                "accum_steps": 1,
                "grad_clip": 0.0,
                "weight_decay": 0.0,
                "learning_rate": 1e-3,
            },
            "preprocess": {"window": {"overlap": 0.0}},
            "mesh": {
                "rounds": 1,
                "steps_per_round": 1,
                "merge": False,
                "outer_lr": 0.7,
                "outer_momentum": 0.9,
                "sync_timeout_s": 5.0,
            },
        }
    )

    run_mesh_pretrain_stage(cfg, device=torch.device("cpu"))

    metrics_path = mesh_shared / "metrics" / "run1" / f"{hostname}.jsonl"
    lines = metrics_path.read_text().splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert record["run_id"] == "run1"
    assert record["node"] == hostname
    assert record["round"] == 1  # 1-based, matching the printed "round 1/1" line
    assert record["n_shards"] == 1  # matches len(assignments[hostname])
    # The exact compute axis comes from train_stats, not from the printed
    # throughput -- one window, one step, for this single-window fixture.
    assert record["n_samples"] == 1
    assert record["n_steps"] == 1
    assert math.isfinite(record["train_loss"])
    # val_loss is nan here -- this fixture seeds no val shards (val_root
    # never exists), same as every other run_mesh_pretrain_stage test in
    # this file; only its presence/type is this test's concern.
    assert isinstance(record["val_loss"], float)
    assert record["samp_s"] >= 0.0
    assert isinstance(record["ts"], float)


def test_run_mesh_pretrain_stage_continues_past_a_failed_push(tmp_path, monkeypatch):
    """0.1: 74 production crashes were a node dying when its checkpoint push
    failed. The round loop must skip a failed round's barrier and keep
    training, not exit."""
    torch.manual_seed(0)
    hostname = socket.gethostname().split(".", 1)[0]

    mesh_root = tmp_path / "mesh_root"
    shard_name = _write_shard(mesh_root, "clip0", n_frames=4)

    mesh_shared = tmp_path / "mesh_shared"
    assignments_path = mesh_shared / "metadata" / "shard_assignments_run1.json"
    assignments_path.parent.mkdir(parents=True)
    assignments_path.write_text(json.dumps({hostname: [shard_name]}))

    shard_map_path = mesh_shared / "metadata" / "shard_map.json"
    shard_map_path.write_text(json.dumps({}))

    monkeypatch.setenv("MESH_RUN_ID", "run1")
    monkeypatch.setenv("MESH_SHARED", str(mesh_shared))
    monkeypatch.setenv("MESH_ASSIGNMENTS", str(assignments_path))
    monkeypatch.setenv("MESH_SHARD_MAP", str(shard_map_path))
    monkeypatch.setenv("MESH_ROOT", str(mesh_root))
    monkeypatch.setenv("STERN_CHECKPOINT_ROOT", str(tmp_path / "stern_checkpoints"))

    # Round 1's (0-indexed round_num=1) sync flag pre-exists so its barrier
    # clears immediately -- this test is not about barrier waiting, only
    # about round 0's push failure not killing the worker.
    flag_dir = mesh_shared / "checkpoints" / "flags"
    flag_dir.mkdir(parents=True)
    (flag_dir / "sync_complete_run1_1.flag").touch()

    real_try_save = mesh_round.ckpt_io.try_save
    calls = []

    def flaky_try_save(path, payload, **kwargs):
        calls.append(path)
        if len(calls) == 1:
            return False
        return real_try_save(path, payload, **kwargs)

    monkeypatch.setattr(mesh_round.ckpt_io, "try_save", flaky_try_save)
    monkeypatch.setattr(mesh_round.time, "sleep", lambda _s: None)

    cfg = DictConfig(
        {
            "model": _tiny_cfg(),
            "train": {
                "amp": "off",
                "accum_steps": 1,
                "grad_clip": 0.0,
                "weight_decay": 0.0,
                "learning_rate": 1e-3,
            },
            "preprocess": {"window": {"overlap": 0.0}},
            "mesh": {
                "rounds": 2,
                "steps_per_round": 1,
                "merge": True,
                "outer_lr": 0.7,
                "outer_momentum": 0.9,
                "sync_timeout_s": 5.0,
            },
        }
    )

    run_mesh_pretrain_stage(cfg, device=torch.device("cpu"))

    # Both rounds ran to completion despite round 0's push failing outright.
    metrics_path = mesh_shared / "metrics" / "run1" / f"{hostname}.jsonl"
    lines = metrics_path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["round"] == 1
    assert json.loads(lines[1])["round"] == 2
    assert len(calls) == 2


def test_run_mesh_pretrain_stage_skips_push_on_zero_sample_round(tmp_path, monkeypatch):
    """0.3: a worker that fetched no shards must not push a checkpoint of an
    untrained model into the merge -- that previously counted toward quorum
    with train_loss=nan. Skip the push and keep going, do not exit."""
    torch.manual_seed(0)
    hostname = socket.gethostname().split(".", 1)[0]

    mesh_root = tmp_path / "mesh_root"
    shard_name = _write_shard(mesh_root, "clip0", n_frames=4)

    mesh_shared = tmp_path / "mesh_shared"
    assignments_path = mesh_shared / "metadata" / "shard_assignments_run1.json"
    assignments_path.parent.mkdir(parents=True)
    assignments_path.write_text(json.dumps({hostname: [shard_name]}))

    shard_map_path = mesh_shared / "metadata" / "shard_map.json"
    shard_map_path.write_text(json.dumps({}))

    monkeypatch.setenv("MESH_RUN_ID", "run1")
    monkeypatch.setenv("MESH_SHARED", str(mesh_shared))
    monkeypatch.setenv("MESH_ASSIGNMENTS", str(assignments_path))
    monkeypatch.setenv("MESH_SHARD_MAP", str(shard_map_path))
    monkeypatch.setenv("MESH_ROOT", str(mesh_root))
    monkeypatch.setenv("STERN_CHECKPOINT_ROOT", str(tmp_path / "stern_checkpoints"))

    monkeypatch.setattr(
        mesh_round,
        "train_one_round",
        lambda *a, **kw: {
            "loss": float("nan"),
            "samples": 0,
            "steps": 0,
            "samples_per_s_per_gpu": 0.0,
        },
    )

    push_calls = []
    monkeypatch.setattr(
        mesh_round,
        "push_checkpoint_and_wait",
        lambda *a, **kw: (
            push_calls.append(1)
            or {
                "pending_push_s": 0.0,
                "pending_bytes": 0,
                "flag_wait_s": 0.0,
                "pushed": True,
            }
        ),
    )

    cfg = DictConfig(
        {
            "model": _tiny_cfg(),
            "train": {
                "amp": "off",
                "accum_steps": 1,
                "grad_clip": 0.0,
                "weight_decay": 0.0,
                "learning_rate": 1e-3,
            },
            "preprocess": {"window": {"overlap": 0.0}},
            "mesh": {
                "rounds": 1,
                "steps_per_round": 1,
                "merge": True,
                "outer_lr": 0.7,
                "outer_momentum": 0.9,
                "sync_timeout_s": 5.0,
            },
        }
    )

    result = run_mesh_pretrain_stage(cfg, device=torch.device("cpu"))

    assert push_calls == []
    assert result["final_round"] == 0
    local_path = tmp_path / "stern_checkpoints" / "run1.pt"
    assert not local_path.exists()
    pending_dir = mesh_shared / "checkpoints" / "pending"
    assert list(pending_dir.glob("run1_*.pt")) == [] if pending_dir.exists() else True


def test_append_round_metrics_swallows_unwritable_metrics_dir(tmp_path):
    """A failed metrics write must never kill a training round -- metrics are
    diagnostic, not correctness-critical."""
    mesh_shared = tmp_path / "mesh_shared"
    mesh_shared.mkdir()
    # A plain file where the metrics/<run_id> directory needs to go forces
    # the mkdir inside _append_round_metrics to fail.
    (mesh_shared / "metrics").write_text("not a directory")

    mesh_round._append_round_metrics(
        mesh_shared,
        "run1",
        "node001",
        1,
        0.5,
        0.4,
        12.0,
        3,
        0.0015,
        n_samples=64,
        n_steps=2,
    )  # must not raise

    # untouched, still the same plain file (not silently replaced by a dir)
    assert (mesh_shared / "metrics").read_text() == "not a directory"


def test_append_round_metrics_includes_slurm_job_id_from_environ(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "987654")
    mesh_shared = tmp_path / "mesh_shared"

    mesh_round._append_round_metrics(
        mesh_shared,
        "run1",
        "node001",
        1,
        0.5,
        0.4,
        12.0,
        3,
        0.0015,
        n_samples=64,
        n_steps=2,
    )

    lines = (
        (mesh_shared / "metrics" / "run1" / "node001.jsonl").read_text().splitlines()
    )
    record = json.loads(lines[0])
    assert record["slurm_job_id"] == "987654"
    assert record["lr"] == 0.0015


def _run_with_step_capture(tmp_path, monkeypatch, *, mesh_cfg, rates, seed=None):
    """Run `run_mesh_pretrain_stage` with training stubbed out, returning the
    `steps_per_round` it asked for on each round plus the checkpoint root.

    `rates` supplies the throughput each stubbed round reports, which is what
    the next round's budget is derived from.
    """
    hostname = socket.gethostname().split(".", 1)[0]

    mesh_root = tmp_path / "mesh_root"
    shard_name = _write_shard(mesh_root, "clip0", n_frames=4)

    mesh_shared = tmp_path / "mesh_shared"
    assignments_path = mesh_shared / "metadata" / "shard_assignments_run1.json"
    assignments_path.parent.mkdir(parents=True)
    assignments_path.write_text(json.dumps({hostname: [shard_name]}))
    shard_map_path = mesh_shared / "metadata" / "shard_map.json"
    shard_map_path.write_text(json.dumps({}))

    checkpoint_root = tmp_path / "stern_checkpoints"
    monkeypatch.setenv("MESH_RUN_ID", "run1")
    monkeypatch.setenv("MESH_SHARED", str(mesh_shared))
    monkeypatch.setenv("MESH_ASSIGNMENTS", str(assignments_path))
    monkeypatch.setenv("MESH_SHARD_MAP", str(shard_map_path))
    monkeypatch.setenv("MESH_ROOT", str(mesh_root))
    monkeypatch.setenv("STERN_CHECKPOINT_ROOT", str(checkpoint_root))

    cfg = DictConfig(
        {
            "model": _tiny_cfg(),
            "train": {
                "amp": "off",
                "accum_steps": 1,
                "grad_clip": 0.0,
                "weight_decay": 0.0,
                "learning_rate": 1e-3,
                **({} if seed is None else {"seed": seed}),
            },
            "preprocess": {"window": {"overlap": 0.0}},
            "mesh": mesh_cfg,
        }
    )

    captured: list[int] = []

    def _fake_train_one_round(*args, steps_per_round: int, **kwargs):
        captured.append(steps_per_round)
        return {
            "loss": 0.0,
            # Deliberately != steps: a stub where samples == steps can't
            # tell apart a production bug that reads n_steps from the wrong
            # field (see test_local_payload_records_steps_for_the_merge_weighting).
            "samples": steps_per_round * 3,
            "steps": steps_per_round,
            "samples_per_s_per_gpu": rates[len(captured) - 1],
        }

    monkeypatch.setattr(mesh_round, "train_one_round", _fake_train_one_round)
    monkeypatch.setattr(
        mesh_round,
        "push_checkpoint_and_wait",
        lambda *a, **kw: {
            "pending_push_s": 0.0,
            "pending_bytes": 0,
            "flag_wait_s": 0.0,
        },
    )

    run_mesh_pretrain_stage(cfg, device=torch.device("cpu"))
    return captured, checkpoint_root


_BUDGET_MESH_CFG = {
    "rounds": 2,
    "steps_per_round": 100,
    "merge": False,
    "outer_lr": 0.7,
    "outer_momentum": 0.9,
    "sync_timeout_s": 5.0,
}


def test_round_budget_absent_keeps_the_fixed_step_count(tmp_path, monkeypatch):
    """round_budget_s is opt-in: with it unset every round asks for exactly
    cfg.mesh.steps_per_round, byte-for-byte the pre-pacing behaviour. Every
    mesh run currently configured on the cluster predates this field."""
    captured, _ = _run_with_step_capture(
        tmp_path,
        monkeypatch,
        mesh_cfg=dict(_BUDGET_MESH_CFG),
        rates=[10.0, 10.0],
    )
    assert captured == [100, 100]


def test_round_budget_paces_later_rounds_from_the_measured_rate(tmp_path, monkeypatch):
    """Round 0 is the calibration round (no rate observed yet, so it uses the
    configured step count); round 1 trains what fits in the budget at the
    rate round 0 measured -- 10 steps/s x 5s = 50, against a configured 100.
    Equalising round *duration* across heterogeneous nodes is what removes
    the barrier the fast nodes otherwise wait out."""
    captured, _ = _run_with_step_capture(
        tmp_path,
        monkeypatch,
        mesh_cfg={**_BUDGET_MESH_CFG, "round_budget_s": 5.0},
        rates=[10.0, 10.0],
    )
    assert captured == [100, 50]


def test_local_payload_records_steps_for_the_merge_weighting(tmp_path, monkeypatch):
    """Unequal step counts across nodes are only sound if the merge can see
    them; n_samples alone can't distinguish a node that took few steps from
    one that was assigned less data."""
    captured, checkpoint_root = _run_with_step_capture(
        tmp_path,
        monkeypatch,
        mesh_cfg={**_BUDGET_MESH_CFG, "round_budget_s": 5.0},
        rates=[10.0, 10.0],
    )
    payload = torch.load(checkpoint_root / "run1.pt", weights_only=False)
    assert payload["n_steps"] == captured[-1]
    assert (
        payload["n_samples"] == captured[-1] * 3
    )  # distinct from n_steps, not a stand-in for it


def test_run_mesh_pretrain_stage_reproducible_across_ambient_rng_state(
    tmp_path, monkeypatch
):
    """Production never calls torch.manual_seed before this function runs, so
    two cold starts with identical cfg currently produce different recon_loss
    depending on whatever RNG state the process happened to be in (model init
    + RandomTubeMasking sampling both draw from it). run_mesh_pretrain_stage
    must reseed from cfg.train.seed itself so cold-start sweep cells are
    comparable. Proven here by deliberately setting DIFFERENT ambient seeds
    right before two otherwise-identical cold starts and asserting identical
    results -- if the function doesn't reseed internally, the two ambient
    seeds diverge and this fails."""

    def _cold_start_recon_loss(subdir: str, ambient_seed: int) -> float:
        torch.manual_seed(ambient_seed)
        hostname = socket.gethostname().split(".", 1)[0]
        root = tmp_path / subdir
        mesh_root = root / "mesh_root"
        shard_name = _write_shard(mesh_root, "clip0", n_frames=4)
        mesh_shared = root / "mesh_shared"
        assignments_path = mesh_shared / "metadata" / "shard_assignments_run1.json"
        assignments_path.parent.mkdir(parents=True)
        assignments_path.write_text(json.dumps({hostname: [shard_name]}))
        shard_map_path = mesh_shared / "metadata" / "shard_map.json"
        shard_map_path.write_text(json.dumps({}))
        checkpoint_root = root / "stern_checkpoints"

        monkeypatch.setenv("MESH_RUN_ID", "run1")
        monkeypatch.setenv("MESH_SHARED", str(mesh_shared))
        monkeypatch.setenv("MESH_ASSIGNMENTS", str(assignments_path))
        monkeypatch.setenv("MESH_SHARD_MAP", str(shard_map_path))
        monkeypatch.setenv("MESH_ROOT", str(mesh_root))
        monkeypatch.setenv("STERN_CHECKPOINT_ROOT", str(checkpoint_root))

        cfg = DictConfig(
            {
                "model": _tiny_cfg(),
                "train": {
                    "seed": 42,
                    "amp": "off",
                    "accum_steps": 1,
                    "grad_clip": 0.0,
                    "weight_decay": 0.0,
                    "learning_rate": 1e-3,
                },
                "preprocess": {"window": {"overlap": 0.0}},
                "mesh": {
                    "rounds": 1,
                    "steps_per_round": 1,
                    "merge": False,
                    "outer_lr": 0.7,
                    "outer_momentum": 0.9,
                    "sync_timeout_s": 5.0,
                },
            }
        )
        result = run_mesh_pretrain_stage(cfg, device=torch.device("cpu"))
        return result["recon_loss"]

    loss_a = _cold_start_recon_loss("a", ambient_seed=111)
    loss_b = _cold_start_recon_loss("b", ambient_seed=222)
    assert loss_a == loss_b


def _mesh_env(tmp_path, monkeypatch, run_id="maskrun"):
    """Minimal MESH_* setup for one cold-start round, mirroring the
    cold-start test above."""
    hostname = socket.gethostname().split(".", 1)[0]
    mesh_root = tmp_path / "mesh_root"
    shard_name = _write_shard(mesh_root, "clip0", n_frames=4)
    mesh_shared = tmp_path / "mesh_shared"
    assignments_path = mesh_shared / "metadata" / f"shard_assignments_{run_id}.json"
    assignments_path.parent.mkdir(parents=True)
    assignments_path.write_text(json.dumps({hostname: [shard_name]}))
    shard_map_path = mesh_shared / "metadata" / "shard_map.json"
    shard_map_path.write_text(json.dumps({}))
    monkeypatch.setenv("MESH_RUN_ID", run_id)
    monkeypatch.setenv("MESH_SHARED", str(mesh_shared))
    monkeypatch.setenv("MESH_ASSIGNMENTS", str(assignments_path))
    monkeypatch.setenv("MESH_SHARD_MAP", str(shard_map_path))
    monkeypatch.setenv("MESH_ROOT", str(mesh_root))
    monkeypatch.setenv("STERN_CHECKPOINT_ROOT", str(tmp_path / "ckpt"))


def _mask_cfg(**model_kwargs):
    return DictConfig(
        {
            "model": DictConfig({**_tiny_cfg(), **model_kwargs}),
            "train": {
                "amp": "off",
                "accum_steps": 1,
                "grad_clip": 0.0,
                "weight_decay": 0.0,
                "learning_rate": 1e-3,
            },
            "preprocess": {"window": {"overlap": 0.0}},
            "mesh": {
                "rounds": 1,
                "steps_per_round": 1,
                "merge": False,
                "sync_timeout_s": 5.0,
            },
        }
    )


@pytest.mark.parametrize(
    "model_kwargs,expected",
    [
        ({}, RandomTubeMasking),
        ({"mask_type": "tube"}, RandomTubeMasking),
        ({"mask_type": "frame"}, RandomFrameMasking),
        ({"mask_type": "mixed"}, MixedTubeFrameMasking),
        ({"motion_mask": True}, MotionGuidedTubeMasking),
    ],
)
def test_mesh_round_honours_mask_type(tmp_path, monkeypatch, model_kwargs, expected):
    """The mesh worker must select its masker from cfg.model.mask_type, the
    same way pretrain.py does.

    This is the only training path with data staged on the cluster, so the
    mask_type x target_mode factorial runs through here. It previously
    hardcoded RandomTubeMasking and honoured only motion_mask, which would
    have silently collapsed the two `mixed` arms onto their `tube`
    counterparts -- four jobs, two distinct conditions, no error.
    """
    _mesh_env(tmp_path, monkeypatch)
    seen = []
    real = mesh_round.build_masker

    def spy(model_cfg):
        masker = real(model_cfg)
        seen.append(masker)
        return masker

    monkeypatch.setattr(mesh_round, "build_masker", spy)

    run_mesh_pretrain_stage(_mask_cfg(**model_kwargs), device=torch.device("cpu"))

    assert len(seen) == 1
    assert isinstance(seen[0], expected)
    assert seen[0].mask_ratio == 0.5


def test_append_round_metrics_records_exact_sample_and_step_counts(tmp_path):
    """The scaling analysis needs a measured compute axis: `samp_s *
    round_budget_s` is an inference that is wrong for exactly the
    fetch-starved low-node-count cells the sweep characterises."""
    mesh_shared = tmp_path / "mesh_shared"

    mesh_round._append_round_metrics(
        mesh_shared,
        "run1",
        "node001",
        7,
        0.5,
        0.4,
        12.0,
        3,
        0.0015,
        n_samples=4096,
        n_steps=150,
    )

    lines = (
        (mesh_shared / "metrics" / "run1" / "node001.jsonl").read_text().splitlines()
    )
    record = json.loads(lines[0])
    assert record["n_samples"] == 4096
    assert record["n_steps"] == 150
    # Every pre-existing key must survive with its passed value, so the
    # addition cannot silently drop a field the sweep analysis reads.
    assert record["run_id"] == "run1"
    assert record["node"] == "node001"
    assert record["round"] == 7
    assert record["train_loss"] == 0.5
    assert record["val_loss"] == 0.4
    assert record["samp_s"] == 12.0
    assert record["n_shards"] == 3
    assert record["lr"] == 0.0015
    assert math.isfinite(record["ts"])
    assert "slurm_job_id" in record


def _flag(mesh_shared: Path, run_id: str, round_num: int) -> Path:
    d = mesh_shared / "checkpoints" / "flags"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"sync_complete_{run_id}_{round_num}.flag"
    p.write_text("")
    return p


def test_round_already_merged_false_when_no_flag(tmp_path):
    assert round_already_merged(tmp_path, "RUN1", 3) is False


def test_round_already_merged_true_for_this_round(tmp_path):
    _flag(tmp_path, "RUN1", 3)
    assert round_already_merged(tmp_path, "RUN1", 3) is True


def test_round_already_merged_ignores_other_rounds(tmp_path):
    """A prior round's flag is always present mid-run -- only the flag for
    the round being trained means the merge went ahead without us."""
    _flag(tmp_path, "RUN1", 0)
    _flag(tmp_path, "RUN1", 1)
    _flag(tmp_path, "RUN1", 2)
    assert round_already_merged(tmp_path, "RUN1", 3) is False


def test_round_already_merged_ignores_other_runs(tmp_path):
    _flag(tmp_path, "RUN2", 3)
    assert round_already_merged(tmp_path, "RUN1", 3) is False


def test_micro_batch_size_configurable(tmp_path):
    """micro_batch_size must reach the DataLoader as a real batch dimension.

    Asserting only stats["steps"] == steps_per_round is not enough: the
    default batch_size of 1 yields the same step count, so that assertion
    passes unchanged even if the parameter is ignored entirely. Spying on the
    tensor the model actually receives is what pins the behavior down.
    """
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = VideoMAEForPreTraining(_tiny_cfg()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    masker = RandomTubeMasking(mask_ratio=0.5)
    # 53 frames, overlap=0.75 -> step=1 -> 50 windows: comfortably more than
    # the 8 loader batches * micro_batch_size 4 = 32 windows this test needs.
    dataset = _mesh_dataset(tmp_path / "mesh", "clip0", n_frames=53, overlap=0.75)

    seen_batch_dims = []
    real_forward = model.forward

    def _spy(pixel_values, *args, **kwargs):
        seen_batch_dims.append(int(pixel_values.shape[0]))
        return real_forward(pixel_values, *args, **kwargs)

    model.forward = _spy
    try:
        stats = train_one_round(
            model,
            optimizer,
            device,
            masker,
            dataset,
            steps_per_round=8,
            amp="off",
            accum_steps=1,
            grad_clip=0.0,
            round_num=0,
            micro_batch_size=4,
        )
    finally:
        model.forward = real_forward

    # stats["steps"] counts DataLoader iterations (loader batches), not
    # samples or optimizer steps -- same convention test_train_one_round_
    # respects_steps_per_round already relies on at the default batch_size=1.
    assert stats["steps"] == 8
    assert seen_batch_dims == [4] * 8, (
        f"model saw batch dims {seen_batch_dims}, expected eight batches of 4 "
        "-- micro_batch_size did not reach the DataLoader"
    )
    # 8 batches x 4 windows: the whole point of widening the micro-batch is
    # that a round consumes micro_batch_size times more data per step.
    assert stats["samples"] == 32


def test_per_sample_masking_reaches_the_training_loop(tmp_path):
    """Independent masks per batch row must survive into train_one_round.

    The masker's own unit test proves per-sample masking in isolation; this
    proves the wiring, i.e. that the loop passes the REAL batch size through
    rather than a hardcoded 1 (which would silently restore one-shared-mask
    -per-batch at any micro_batch_size).
    """
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = VideoMAEForPreTraining(_tiny_cfg()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dataset = _mesh_dataset(tmp_path / "mesh", "clip0", n_frames=53, overlap=0.75)

    inner = RandomTubeMasking(mask_ratio=0.5)
    seen = []

    class _SpyMasker:
        def __call__(self, pixel_values, grid, dev, batch_size):
            keep, mask = inner(pixel_values, grid, dev, batch_size)
            seen.append(keep)
            return keep, mask

    train_one_round(
        model,
        optimizer,
        device,
        _SpyMasker(),
        dataset,
        steps_per_round=4,
        amp="off",
        accum_steps=1,
        grad_clip=0.0,
        round_num=0,
        micro_batch_size=4,
    )

    assert seen, "masker was never called"
    assert all(k.shape[0] == 4 for k in seen), (
        f"masker saw batch sizes {[k.shape[0] for k in seen]}, expected 4 -- "
        "the loop is not forwarding the real batch size"
    )
    assert any(not torch.equal(k[0], k[1]) for k in seen), (
        "every batch row got the identical mask inside the training loop"
    )


class TestMeshRoundLR:
    def test_warmup_ramps_linearly_to_peak(self):
        peak = 1.0
        total = 100
        # warmup_frac=0.05 -> 5 warmup rounds; round 4 is the last warmup round
        lr0 = mesh_round_lr(peak, round_num=0, total_rounds=total, warmup_frac=0.05)
        lr4 = mesh_round_lr(peak, round_num=4, total_rounds=total, warmup_frac=0.05)
        assert lr0 == pytest.approx(peak * 1 / 5)
        assert lr4 == pytest.approx(peak * 5 / 5)

    def test_cosine_decays_to_min_lr_at_final_round(self):
        lr_final = mesh_round_lr(
            1.0, round_num=99, total_rounds=100, warmup_frac=0.05, min_lr=0.01
        )
        assert lr_final == pytest.approx(0.01, abs=1e-3)

    def test_pure_function_of_round_num(self):
        """A chunked relaunch must reproduce the same LR from round_num alone."""
        a = mesh_round_lr(3e-4, round_num=17, total_rounds=120)
        b = mesh_round_lr(3e-4, round_num=17, total_rounds=120)
        assert a == b

    def test_continuous_across_the_warmup_cosine_boundary(self):
        """The last warmup round and the first cosine round must agree.

        They are computed by two different branches; an off-by-one in either
        shows up as an LR discontinuity exactly here and nowhere else.
        """
        peak, total = 3e-4, 120
        warmup_rounds = round(0.05 * total)  # 6
        last_warmup = mesh_round_lr(peak, warmup_rounds - 1, total)
        first_cosine = mesh_round_lr(peak, warmup_rounds, total)
        assert last_warmup == pytest.approx(peak)
        assert first_cosine == pytest.approx(peak)

    def test_decays_monotonically_after_warmup(self):
        peak, total = 3e-4, 120
        after = [mesh_round_lr(peak, r, total) for r in range(6, total)]
        assert all(b <= a for a, b in zip(after, after[1:]))
        assert after[0] > after[-1]

    def test_overrun_past_the_final_round_clamps_to_min_lr(self):
        """A worker relaunched past `rounds` must not wrap back up the cosine."""
        for r in (120, 150, 10_000):
            assert mesh_round_lr(3e-4, r, 120, min_lr=1e-5) == pytest.approx(1e-5)

    def test_single_round_run_still_warms_up_to_peak(self):
        """round(0.05 * 1) == 0; the max(1, ...) floor must keep it usable."""
        assert mesh_round_lr(3e-4, round_num=0, total_rounds=1) == pytest.approx(3e-4)


def _minimal_mesh_cfg(encoder: str) -> DictConfig:
    return _mask_cfg(encoder=encoder)


def test_run_mesh_pretrain_stage_dispatches_on_model_encoder_field(monkeypatch):
    """cfg.model.encoder == 'levjepa' must route model construction to
    LeVJEPAEncoder/Projector, not VideoMAEForPreTraining -- verified by
    spying on both constructors."""
    import video_encoding.training.diloco.mesh_round as mod

    calls = {"levjepa": 0, "videomae": 0}

    class _StubLeVJEPAEncoder(torch.nn.Module):
        def __init__(self, cfg):
            super().__init__()
            calls["levjepa"] += 1
            self.dummy = torch.nn.Linear(1, 1)

    class _StubVideoMAE(torch.nn.Module):
        def __init__(self, cfg):
            super().__init__()
            calls["videomae"] += 1
            self.dummy = torch.nn.Linear(1, 1)

    monkeypatch.setattr(mod, "LeVJEPAEncoder", _StubLeVJEPAEncoder)
    monkeypatch.setattr(mod, "VideoMAEForPreTraining", _StubVideoMAE)

    cfg_levjepa = _minimal_mesh_cfg(encoder="levjepa")
    mod._build_pretrain_model(cfg_levjepa, device=torch.device("cpu"))
    assert calls == {"levjepa": 1, "videomae": 0}

    calls["levjepa"] = calls["videomae"] = 0
    cfg_videomae = _minimal_mesh_cfg(encoder="videomae")
    mod._build_pretrain_model(cfg_videomae, device=torch.device("cpu"))
    assert calls == {"levjepa": 0, "videomae": 1}


@pytest.mark.parametrize("seed, expected", [(None, 42), (0, 0)])
def test_payload_records_the_seed_the_model_was_initialised_from(
    tmp_path, monkeypatch, seed, expected
):
    """delta_from_init only trusts a random init it can prove is the run's
    own (ssl-diag-01 task 8.1), so the checkpoint names the seed -- the
    default 42 when cfg.train.seed is unset, as the manual_seed call uses."""
    _, checkpoint_root = _run_with_step_capture(
        tmp_path,
        monkeypatch,
        mesh_cfg=dict(_BUDGET_MESH_CFG),
        rates=[10.0, 10.0],
        seed=seed,
    )
    payload = torch.load(checkpoint_root / "run1.pt", weights_only=False)
    assert payload["init_seed"] == expected


def _tiny_levjepa_model_cfg() -> DictConfig:
    return DictConfig(
        {
            "encoder": "levjepa",
            "num_frames": 16,
            "image_size": 32,
            "local_image_size": 16,
            "in_channels": 3,
            "patch_size": 16,
            "embed_dim": 12,
            "depth": 1,
            "num_heads": 2,
            "mlp_ratio": 2.0,
            "dropout": 0.0,
            "drop_path_rate": 0.0,
            "drop_ratio": 0.5,
            "num_local_views": 2,
            "sigreg_lambda": 0.02,
            "sigreg_directions": 8,
            "projector_hidden": 16,
            "embed_dim_k": 8,
            "polyak_decay": 0.9,
            "polyak_update_every": 1,
        }
    )


class _FakeRawIterableDataset(torch.utils.data.IterableDataset):
    """Mimics MeshShardIterableDataset(raw_frames=True): streams (T, H, W,
    3) uint8 clips, with a no-op set_round (real dataset has one too)."""

    def __init__(self, n=4, num_frames=16, size=32):
        self._n = n
        self._num_frames = num_frames
        self._size = size

    def set_round(self, round_num: int) -> None:
        pass

    def __iter__(self):
        for _ in range(self._n):
            yield torch.randint(
                0, 256, (self._num_frames, self._size, self._size, 3), dtype=torch.uint8
            )


def test_train_one_round_and_evaluate_val_loss_dispatch_to_levjepa(monkeypatch):
    """train_one_round/evaluate_val_loss must route a _LeVJEPAPretrainModule
    through _train_epoch_levjepa/_evaluate_val_loss_levjepa, never touching
    the VideoMAE masker/token_grid/compute_loss path -- the exact crash the
    prior dead-branch bug produced (encoder='levjepa' constructed a
    _LeVJEPAPretrainModule but the round loop still assumed VideoMAE's
    interface, so round 1 crashed immediately)."""
    from video_encoding.training.diloco.mesh_round import (
        LeVJEPAViewIterableDataset,
        _LeVJEPAPretrainModule,
    )

    def _boom(*args, **kwargs):
        raise AssertionError("build_masker must not be called on the LeVJEPA path")

    monkeypatch.setattr(mesh_round, "build_masker", _boom)

    model_cfg = _tiny_levjepa_model_cfg()
    model = _LeVJEPAPretrainModule(model_cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    device = torch.device("cpu")

    raw = _FakeRawIterableDataset(n=4, num_frames=16, size=32)
    train_dataset = LeVJEPAViewIterableDataset(
        raw, num_local=2, global_size=32, local_size=16
    )

    train_stats = train_one_round(
        model,
        optimizer,
        device,
        None,  # masker: unused on the LeVJEPA path
        train_dataset,
        steps_per_round=2,
        amp="off",
        accum_steps=1,
        grad_clip=0.0,
        round_num=0,
        micro_batch_size=2,
        sigreg_lambda=float(model_cfg.sigreg_lambda),
        sigreg_directions=int(model_cfg.sigreg_directions),
    )

    assert train_stats["steps"] == 2
    assert train_stats["samples"] == 4  # 2 steps x micro_batch_size=2
    assert not math.isnan(train_stats["loss"])
    assert not train_stats["abandoned"]

    val_raw = _FakeRawIterableDataset(n=2, num_frames=16, size=32)
    val_dataset = LeVJEPAViewIterableDataset(
        val_raw, num_local=2, global_size=32, local_size=16
    )
    val_loss = evaluate_val_loss(
        model,
        val_dataset,
        device,
        None,
        "off",
        sigreg_lambda=float(model_cfg.sigreg_lambda),
        sigreg_directions=int(model_cfg.sigreg_directions),
    )
    assert not math.isnan(val_loss)
    assert model.training  # left in train mode, matching train_one_round's expectation
