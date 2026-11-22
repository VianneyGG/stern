"""Tests for `stern/evaluation/diagnostics/representational.py` --
per-block attention-map entropy, computed off the shared residual-stream
traversal (`_diag_common.forward_with_hidden_states`), never a hand-mirrored
copy of `ViTBlock.forward`.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from stern.models.videomae.encoder import VideoMAEEncoder
from stern.evaluation.diagnostics.common import DiagBundle
from stern.evaluation.diagnostics.representational import (
    _entropy_of_attention_map,
    attention_entropy_by_block,
    check,
)


def _tiny_model_cfg(**overrides):
    cfg = {
        "num_frames": 4,
        "image_size": 32,
        "in_channels": 3,
        "tubelet_size": 2,
        "patch_size": 16,
        "embed_dim": 16,
        "depth": 2,
        "num_heads": 2,
        "mlp_ratio": 2.0,
        "dropout": 0.0,
        "drop_path_rate": 0.0,
    }
    cfg.update(overrides)
    return OmegaConf.create(cfg)


def _bundle(cfg) -> DiagBundle:
    encoder = VideoMAEEncoder(cfg).eval()
    return DiagBundle(encoder=encoder, model=None, run_id="r", round=None)


# --- _entropy_of_attention_map ------------------------------------------


def test_attention_entropy_maximal_on_uniform_map():
    n = 8
    uniform = np.full((n, n), 1.0 / n)
    assert _entropy_of_attention_map(uniform) == pytest.approx(np.log(n), rel=1e-4)


def test_attention_entropy_near_zero_on_permutation_map():
    n = 6
    perm = np.eye(n)[np.random.default_rng(0).permutation(n)]
    assert _entropy_of_attention_map(perm) == pytest.approx(0.0, abs=1e-6)


def test_attention_entropy_invariant_to_joint_permutation():
    rng = np.random.default_rng(0)
    n = 10
    logits = rng.standard_normal((n, n))
    attn = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
    perm = rng.permutation(n)
    permuted = attn[perm][:, perm]
    assert _entropy_of_attention_map(attn) == pytest.approx(
        _entropy_of_attention_map(permuted), rel=1e-6
    )


# --- attention_entropy_by_block -----------------------------------------


def test_attention_entropy_by_block_returns_one_value_per_block():
    cfg = _tiny_model_cfg(depth=3)
    bundle = _bundle(cfg)
    pixel_values = torch.randn(2, 3, 4, 32, 32)

    result = attention_entropy_by_block(bundle, pixel_values)

    assert set(result.keys()) == set(range(3))
    for value in result.values():
        assert value.deterministic is True
        assert np.isfinite(value.mean)


# --- check ----------------------------------------------------------------


def test_check_reports_entropy_for_both_real_and_random_inputs():
    cfg = _tiny_model_cfg(depth=2)
    bundle = _bundle(cfg)
    real = torch.randn(2, 3, 4, 32, 32)
    random_batch = torch.randn(2, 3, 4, 32, 32)

    reports = check(bundle, {"real": real, "random": random_batch})

    assert reports.keys() == {"attention_entropy_real", "attention_entropy_random"}
    for key, report in reports.items():
        assert report.meta["input_kind"] == key.removeprefix("attention_entropy_")
        assert set(report.metrics.keys()) == {f"block{i}" for i in range(cfg.depth)}
