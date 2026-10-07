"""train.rng_split / model.init_seed: an arm that changes model init must
still see the control's exact views, token-drop masks and val views."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import video_encoding.training.diloco.mesh_round as mesh_round
import video_encoding.training.pretrain_levjepa as pretrain_levjepa
from video_encoding.models.levjepa.encoder import LeVJEPAEncoder
from video_encoding.models.levjepa.views import build_views, derive_seed
from video_encoding.scripts.build_wds_shards import (
    BalancedShardWriter,
    encode_frames_jpeg,
)
from video_encoding.training.pretrain_levjepa import LeVJEPAViewIterableDataset


def _noise_clip(seed: int, t: int = 4, size: int = 48) -> torch.Tensor:
    # Noise, not solid colour: on a solid frame every crop has the same
    # pixels, so "views are equal" would hold even if the crops differed.
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (t, size, size, 3), generator=g, dtype=torch.uint8)


class _Clips(torch.utils.data.IterableDataset):
    def __iter__(self):
        return iter([_noise_clip(i) for i in range(3)])


def _views(ds) -> list[torch.Tensor]:
    return [
        torch.cat([g.flatten(), loc.flatten(), box.flatten()]) for g, loc, box in ds
    ]


def _equal(a: list[torch.Tensor], b: list[torch.Tensor]) -> bool:
    return all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


def test_derive_seed_is_stable_across_processes():
    # Python's hash() is salted per process; a seed derived from it would
    # silently differ between the control's and the ablation's workers.
    code = "from video_encoding.models.levjepa.views import derive_seed; print(derive_seed(3, 'views', 0))"
    outs = {
        subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ, "PYTHONHASHSEED": salt},
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        for salt in ("1", "2")
    }
    assert outs == {str(derive_seed(3, "views", 0))}
    seeds = {derive_seed(3, "views", 0), derive_seed(3, "views", 1), derive_seed(3, "drop", 0)}
    assert len(seeds) == 3


def test_build_views_with_generator_ignores_and_preserves_global_rng():
    clip = _noise_clip(0)
    torch.manual_seed(1)
    a = build_views(
        clip,
        num_local=2,
        global_size=32,
        local_size=16,
        generator=torch.Generator().manual_seed(5),
    )
    state = torch.get_rng_state()
    torch.manual_seed(2)
    b = build_views(
        clip,
        num_local=2,
        global_size=32,
        local_size=16,
        generator=torch.Generator().manual_seed(5),
    )
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    torch.manual_seed(1)
    build_views(
        clip,
        num_local=2,
        global_size=32,
        local_size=16,
        generator=torch.Generator().manual_seed(5),
    )
    assert torch.equal(torch.get_rng_state(), state)  # global stream not consumed
    c = build_views(
        clip,
        num_local=2,
        global_size=32,
        local_size=16,
        generator=torch.Generator().manual_seed(6),
    )
    assert not torch.equal(a[2], c[2])  # a different seed gives different crop boxes


def test_view_dataset_seed_is_round_keyed_and_global_rng_independent():
    ds = LeVJEPAViewIterableDataset(
        _Clips(), num_local=2, global_size=32, local_size=16, view_seed=9
    )
    torch.manual_seed(1)
    r0 = _views(ds)
    torch.manual_seed(2)
    same = [torch.equal(x, y) for x, y in zip(r0, _views(ds), strict=True)]
    assert same == [True] * len(r0)  # same round, different global state
    ds.set_round(1)
    assert not _equal(r0, _views(ds))  # a new round draws new views


def test_view_dataset_without_seed_still_uses_global_rng():
    ds = LeVJEPAViewIterableDataset(
        _Clips(), num_local=2, global_size=32, local_size=16
    )
    torch.manual_seed(1)
    a = _views(ds)
    torch.manual_seed(1)
    same = [torch.equal(x, y) for x, y in zip(a, _views(ds), strict=True)]
    assert same == [True] * len(a)
    torch.manual_seed(2)
    assert not _equal(a, _views(ds))


def _tiny_encoder() -> LeVJEPAEncoder:
    cfg = OmegaConf.create(
        {
            "num_frames": 2,
            "image_size": 32,
            "patch_size": 8,
            "embed_dim": 12,
            "depth": 1,
            "num_heads": 2,
            "drop_ratio": 0.5,
        }
    )
    return LeVJEPAEncoder(cfg)


def test_reseed_drop_fixes_mask_and_leaves_global_rng_alone():
    enc = _tiny_encoder().train()
    tokens = torch.randn(3, 32, 12)
    coords = torch.arange(32).view(1, 32, 1).expand(3, 32, 3).float()
    masks = []
    for global_seed in (1, 2):
        torch.manual_seed(global_seed)
        enc.reseed_drop(4)
        state = torch.get_rng_state()
        _, kept = enc._drop_tokens(tokens, coords)
        assert torch.equal(torch.get_rng_state(), state)
        masks.append(kept)
    assert torch.equal(masks[0], masks[1])
    enc.reseed_drop(5)
    assert not torch.equal(enc._drop_tokens(tokens, coords)[1], masks[0])


def _write_noise_shard(root, n: int, rng: np.random.Generator) -> str:
    writer = BalancedShardWriter(root, n_shards=1)
    for i in range(n):
        frames = [rng.integers(0, 256, (32, 32, 3), dtype=np.uint8) for _ in range(16)]
        writer.write(f"id{i:05d}/v{i}/{i:05d}", encode_frames_jpeg(frames))
    writer.close()
    return "shard-000000.tar"


def _h(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().float().contiguous().numpy().tobytes()).hexdigest()


def _run_stage(tmp_path, monkeypatch, **overrides) -> dict:
    """Run the real mesh stage on tiny noise shards; trace the views each
    encoder call sees, the kept-token coordinates, and the outputs."""
    trace = {"pix": [], "mask": [], "out": [], "drop_private": [], "view_private": []}
    orig_forward, orig_drop = LeVJEPAEncoder.forward, LeVJEPAEncoder._drop_tokens
    orig_build_views = pretrain_levjepa.build_views

    def build_views(*args, generator=None, **kwargs):
        trace["view_private"].append(generator is not None)
        return orig_build_views(*args, generator=generator, **kwargs)

    def forward(self, pixel_values, crop_box=None):
        out = orig_forward(self, pixel_values, crop_box=crop_box)
        trace["pix"].append((self.training, _h(pixel_values)))
        trace["out"].append(_h(out))
        return out

    def drop(self, tokens, coords):
        kept_tokens, kept_coords = orig_drop(self, tokens, coords)
        if self.training:
            trace["mask"].append(_h(kept_coords))
            trace["drop_private"].append(self.drop_generator is not None)
        return kept_tokens, kept_coords

    # Patched and restored per run (not via monkeypatch, which only undoes at
    # test end): a second run in the same test would otherwise wrap the first
    # run's wrapper and record into both traces.
    LeVJEPAEncoder.forward, LeVJEPAEncoder._drop_tokens = forward, drop
    pretrain_levjepa.build_views = build_views

    rng = np.random.default_rng(0)
    mesh_root = tmp_path / "mesh_root"
    shard = _write_noise_shard(mesh_root, 8, rng)
    _write_noise_shard(mesh_root / "val", 4, rng)
    meta = tmp_path / "shared" / "metadata"
    meta.mkdir(parents=True)
    host = socket.gethostname().split(".", 1)[0]
    (meta / "a.json").write_text(json.dumps({host: [shard]}))
    (meta / "m.json").write_text("{}")
    for k, v in {
        "MESH_RUN_ID": "run1",
        "MESH_SHARED": tmp_path / "shared",
        "MESH_ASSIGNMENTS": meta / "a.json",
        "MESH_SHARD_MAP": meta / "m.json",
        "MESH_ROOT": mesh_root,
        "STERN_CHECKPOINT_ROOT": tmp_path / "ckpt",
    }.items():
        monkeypatch.setenv(k, str(v))
    cfg = OmegaConf.create(
        {
            "model": {
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
            },
            "train": {
                "amp": "off",
                "accum_steps": 1,
                "grad_clip": 0.0,
                "lr": 1e-3,
                "weight_decay": 0.0,
                "batch_size": 2,
                "seed": 3,
            },
            "preprocess": {"window": {"overlap": 0.0}},
            "mesh": {
                "rounds": 2,
                "steps_per_round": 3,
                "merge": False,
                "outer_lr": 1.0,
                "outer_momentum": 0.0,
                "sync_timeout_s": 5.0,
                "micro_batch_size": 2,
            },
        }
    )
    OmegaConf.set_struct(cfg, False)
    cfg.merge_with_dotlist(
        [f"{k.replace('__', '.')}={v}" for k, v in overrides.items()]
    )
    try:
        mesh_round.run_mesh_pretrain_stage(cfg, device=torch.device("cpu"))
    finally:
        LeVJEPAEncoder.forward, LeVJEPAEncoder._drop_tokens = orig_forward, orig_drop
        pretrain_levjepa.build_views = orig_build_views
    return trace


def _by_mode(trace) -> tuple[list[str], list[str]]:
    train = [p for t, p in trace["pix"] if t]
    val = [p for t, p in trace["pix"] if not t]
    return train, val


@pytest.mark.integration
def test_init_seed_changes_weights_but_not_views_or_masks(tmp_path, monkeypatch):
    split = {"train__rng_split": "true", "model__sigreg_seed": 0}
    a = _run_stage(tmp_path / "a", monkeypatch, **split, model__init_seed=1)
    b = _run_stage(tmp_path / "b", monkeypatch, **split, model__init_seed=2)
    assert a["pix"] == b["pix"] and a["mask"] == b["mask"]
    # 2 rounds x 3 steps x 2 encoder calls (global, then the stacked locals)
    assert len(a["mask"]) == 12 and all(a["drop_private"]) and all(a["view_private"])
    assert all(x != y for x, y in zip(a["out"], b["out"], strict=True))  # the init really differed


@pytest.mark.integration
def test_rng_split_val_views_fixed_across_rounds_train_views_not(tmp_path, monkeypatch):
    t = _run_stage(tmp_path, monkeypatch, train__rng_split="true", model__sigreg_seed=0)
    train, val = _by_mode(t)
    # 4 val clips fit one val batch: 2 encoder calls per val pass, one pass per round
    assert len(val) == 4 and val[:2] == val[2:]
    assert len(train) == 12 and train[:6] != train[6:]
    assert len(set(t["mask"][:6]) & set(t["mask"][6:])) == 0  # round-keyed drop stream


@pytest.mark.integration
def test_flags_off_uses_no_private_generator(tmp_path, monkeypatch):
    t = _run_stage(tmp_path, monkeypatch)
    assert set(t["drop_private"]) == {False}
    assert set(t["view_private"]) == {False}



def test_levjepa_compile_without_rng_split_still_runs(tmp_path, monkeypatch):
    # The rng_split/levjepa_compile guard must not catch compile on its own.
    t = _run_stage(tmp_path, monkeypatch, train__levjepa_compile="true")
    assert t["mask"]
    assert set(t["drop_private"]) == {False}

@pytest.mark.integration
def test_without_rng_split_a_seed_change_moves_views_and_masks(tmp_path, monkeypatch):
    # Negative control for the test above: the harness can see a difference.
    a = _run_stage(tmp_path / "a", monkeypatch, train__seed=3)
    b = _run_stage(tmp_path / "b", monkeypatch, train__seed=4)
    assert a["mask"] != b["mask"]
    assert [p for t, p in a["pix"] if t] != [p for t, p in b["pix"] if t]


@pytest.mark.integration
@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"model__init_seed": 1}, "init_seed without train.rng_split"),
        ({"model__init_seed": 0}, "init_seed without train.rng_split"),  # 0 is not "unset"
        ({"train__rng_split": "true"}, "requires \\+model.sigreg_seed"),
        (
            {"train__rng_split": "true", "model__sigreg_seed": 0, "model__encoder": "videomae"},
            "only implemented for model.encoder=levjepa",
        ),
        (
            {"train__rng_split": "true", "model__sigreg_seed": 0, "train__levjepa_compile": "true"},
            "incompatible with train.levjepa_compile",
        ),
    ],
)
def test_misconfigured_split_refuses_to_run(tmp_path, monkeypatch, overrides, message):
    with pytest.raises(ValueError, match=message):
        _run_stage(tmp_path, monkeypatch, **overrides)


def test_compiled_forward_cls_survives_a_second_batch_shape_with_flags_off():
    # Regression: an explicit generator=None made dynamo fail on the
    # recompile a second batch shape triggers (global B, then B*V locals).
    cfg = OmegaConf.create(
        dict(num_frames=16, image_size=32, local_image_size=16, in_channels=3,
             patch_size=16, embed_dim=12, depth=1, num_heads=2, mlp_ratio=2.0,
             dropout=0.0, drop_path_rate=0.0, drop_ratio=0.5)
    )
    torch.manual_seed(0)
    enc = LeVJEPAEncoder(cfg).train()
    enc.forward_cls = torch.compile(enc.forward_cls)
    for b in (4, 8, 12):
        out = enc.forward_cls(torch.randn(b, 3, 16, 32, 32))
        assert out.shape == (b, 12)
        out.sum().backward()
