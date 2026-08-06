"""Tests for `video_encoding.training.mesh_round` (Task 2 of the mesh
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

import video_encoding.training.mesh_round as mesh_round
from video_encoding.data.mesh_shard_dataset import MeshShardIterableDataset
from video_encoding.models.videomaev2.mae import VideoMAEForPreTraining
from video_encoding.models.videomaev2.masking import RandomTubeMasking
from video_encoding.scripts.build_wds_shards import (
    BalancedShardWriter,
    encode_frames_jpeg,
)
from video_encoding.training.mesh_round import (
    evaluate_val_loss,
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
    pending = tmp_path / "checkpoints" / "pending" / "r1_nodeA.pt"
    assert pending.exists()


def test_push_checkpoint_and_wait_times_out_and_exits(tmp_path, monkeypatch):
    monkeypatch.setattr(mesh_round.time, "sleep", lambda s: None)

    with pytest.raises(SystemExit):
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
        tmp_path / "checkpoints" / "pending" / "r1_nodeA.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert shared["model_state_dict"]["w"].dtype == torch.float16
    assert shared["model_state_dict"]["b"].dtype == torch.float16
    # Caller's own dict must be untouched: local checkpoint saves (which
    # happen before this call in run_mesh_pretrain_stage) must stay fp32.
    assert payload["model_state_dict"]["w"].dtype == torch.float32


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
        mesh_shared / "checkpoints" / "pending" / f"run1_{hostname}.pt",
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


def test_shared_payload_drop_keys_is_exactly_optimizer_state():
    assert mesh_round._SHARED_PAYLOAD_DROP_KEYS == (
        "optimizer_state_dict",
        "optimizer_name",
    )


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
        mesh_shared, "run1", "node001", 1, 0.5, 0.4, 12.0, 3
    )  # must not raise

    assert (mesh_shared / "metrics").is_file()  # untouched, still a plain file
