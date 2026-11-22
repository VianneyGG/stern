"""Tests for `stern.training.compile_utils.maybe_compile`.

CPU-only, tiny modules -- torch.compile works on CPU (verified: ~6s first-call
cost for a single nn.Linear on this environment), no GPU needed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig

from stern.training import checkpoint as ckpt_io
from stern.training.compile_utils import maybe_compile
from stern.training.pretrain import _payload
from stern.models.videomae.mae import VideoMAEForPreTraining


def _tiny_model_cfg() -> DictConfig:
    # Same tiny shape used by test_mesh_round.py / test_pretrain_loop.py.
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


def test_maybe_compile_noop_when_disabled():
    model = nn.Linear(4, 4)
    cfg = DictConfig({"train": {"compile": False}})
    assert maybe_compile(model, cfg) is model


def test_maybe_compile_noop_when_unset():
    model = nn.Linear(4, 4)
    cfg = DictConfig({"train": {}})
    assert maybe_compile(model, cfg) is model


def test_maybe_compile_wraps_when_enabled():
    model = nn.Linear(4, 4)
    cfg = DictConfig({"train": {"compile": True, "compile_mode": "default"}})
    compiled = maybe_compile(model, cfg)
    assert compiled is not model
    assert hasattr(compiled, "_orig_mod")
    assert compiled._orig_mod is model


def test_unwrap_strips_compile_wrapper():
    model = nn.Linear(4, 4)
    cfg = DictConfig({"train": {"compile": True}})
    compiled = maybe_compile(model, cfg)
    assert ckpt_io.unwrap(compiled) is model


def test_compiled_checkpoint_keys_match_uncompiled():
    """DiLoCo's outer step merges worker checkpoints by key name
    (mesh/outer_step.py::pseudo_gradient). If a compiled model's saved
    state_dict keys ever diverged from an uncompiled one's, mixed-fleet
    merges would silently break (strict=False on the load side)."""
    model_cfg = _tiny_model_cfg()
    torch.manual_seed(0)
    plain_model = VideoMAEForPreTraining(model_cfg)
    torch.manual_seed(0)
    compiled_model = maybe_compile(
        VideoMAEForPreTraining(model_cfg),
        DictConfig({"train": {"compile": True}}),
    )

    plain_optimizer = torch.optim.AdamW(plain_model.parameters(), lr=1e-4)
    compiled_optimizer = torch.optim.AdamW(compiled_model.parameters(), lr=1e-4)

    plain_payload = _payload(
        model=plain_model,
        optimizer=plain_optimizer,
        scheduler=None,
        epoch=0,
        best_loss=1.0,
    )
    compiled_payload = _payload(
        model=compiled_model,
        optimizer=compiled_optimizer,
        scheduler=None,
        epoch=0,
        best_loss=1.0,
    )

    assert (
        plain_payload["model_state_dict"].keys()
        == compiled_payload["model_state_dict"].keys()
    )
    assert not any(
        k.startswith("_orig_mod.") for k in compiled_payload["model_state_dict"]
    )
