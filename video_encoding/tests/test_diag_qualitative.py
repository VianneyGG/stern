"""Tests for `video_encoding/scripts/diag_qualitative.py` -- four Tier-C
visual panels (embedding scatter, retrieval grid, mask-ratio grid, attention
rollout). None of these functions returns a decision rule; `attention_rollout`
is Tier C permanently, not pending calibration.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from video_encoding.models.videomaev2.encoder import VideoMAEEncoder
from video_encoding.models.videomaev2.mae import VideoMAEForPreTraining
from video_encoding.scripts._diag_common import DiagBundle
from video_encoding.scripts.diag_qualitative import (
    attention_rollout,
    check,
    mask_ratio_grid,
    retrieval_grid,
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
        "decoder_depth": 1,
        "decoder_embed_dim": 8,
        "decoder_num_heads": 2,
        "mask_ratio": 0.5,
        "decoder_mask_ratio": 0.0,
    }
    cfg.update(overrides)
    return OmegaConf.create(cfg)


def _encoder_only_bundle(cfg) -> DiagBundle:
    encoder = VideoMAEEncoder(cfg).eval()
    return DiagBundle(encoder=encoder, model=None, run_id="r", round=None)


def _full_model_bundle(cfg) -> DiagBundle:
    model = VideoMAEForPreTraining(cfg).eval()
    return DiagBundle(encoder=model.encoder, model=model, run_id="r", round=None)


# --- attention_rollout ---------------------------------------------------


def test_attention_rollout_identity_at_every_layer_returns_identity():
    n = 6
    layers = [np.eye(n) for _ in range(4)]
    rollout = attention_rollout(layers)
    assert np.allclose(rollout, np.eye(n), atol=1e-6)


def test_attention_rollout_rows_sum_to_one_at_every_layer():
    rng = np.random.default_rng(0)
    n = 7
    layers = []
    for _ in range(3):
        logits = rng.standard_normal((n, n))
        layers.append(np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True))
    rollout = attention_rollout(layers)
    assert np.allclose(rollout.sum(axis=-1), 1.0, atol=1e-5)


# --- retrieval_grid --------------------------------------------------------


def test_retrieval_grid_returns_query_first_when_query_in_gallery():
    rng = np.random.default_rng(0)
    gallery = rng.standard_normal((10, 8))
    query = gallery[[3]]

    result = retrieval_grid(query, gallery, k=5)

    assert result[0, 0] == 3


# --- embedding_scatter ------------------------------------------------


# --- mask_ratio_grid -----------------------------------------------------


def test_mask_ratio_grid_one_panel_per_ratio_at_token_grid_resolution():
    """num_frames=16 gives t_p=8, so temporal_split's cut stays >= 1 through
    ratio=0.9 (8 * 0.1 rounds to 1). At num_frames=4 (t_p=2) ratio=0.75
    already rounds cut to 0 -- round() is banker's rounding, round(0.5) == 0."""
    cfg = _tiny_model_cfg(num_frames=16)
    model = VideoMAEForPreTraining(cfg).eval()
    pixel_values = torch.randn(1, 3, 16, 32, 32)
    ratios = [0.5, 0.75, 0.9]

    grid = mask_ratio_grid(model, pixel_values, ratios, torch.device("cpu"))

    t_p, h_p, w_p = model.encoder.token_grid
    assert grid.shape == (len(ratios), h_p, w_p)


def test_mask_ratio_grid_raises_on_ratio_that_empties_keep():
    """At t_p=8, ratio=0.98 rounds cut to 0: predict_z would encode zero
    visible tokens. That must fail loudly, not render a meaningless panel."""
    cfg = _tiny_model_cfg(num_frames=16)
    model = VideoMAEForPreTraining(cfg).eval()
    pixel_values = torch.randn(1, 3, 16, 32, 32)

    with pytest.raises(ValueError, match="keep"):
        mask_ratio_grid(model, pixel_values, [0.98], torch.device("cpu"))


# --- check -----------------------------------------------------------------


def test_check_wires_all_four_panels_into_arrays():
    """Regression: check() must not return arrays={}. When bundle.model is
    None, mask_ratio_grid is skipped with a recorded meta reason, but the
    other three panels must still be produced."""
    cfg = _tiny_model_cfg(depth=2)
    bundle = _encoder_only_bundle(cfg)
    pixel_values = torch.randn(4, 3, 4, 32, 32)

    reports = check(bundle, {"real": pixel_values})

    assert reports.keys() == {"diag_qualitative_real"}
    report = reports["diag_qualitative_real"]
    assert report.arrays.keys() == {
        "embedding_scatter",
        "retrieval_grid",
        "attention_rollout",
    }
    assert report.meta["mask_ratio_grid_skipped"]


def test_check_wires_all_four_panels_when_full_model_available():
    cfg = _tiny_model_cfg(depth=2, num_frames=16)
    bundle = _full_model_bundle(cfg)
    pixel_values = torch.randn(4, 3, 16, 32, 32)

    reports = check(bundle, {"real": pixel_values})

    assert reports.keys() == {"diag_qualitative_real"}
    report = reports["diag_qualitative_real"]
    assert report.arrays.keys() == {
        "embedding_scatter",
        "retrieval_grid",
        "mask_ratio_grid",
        "attention_rollout",
    }
    assert "mask_ratio_grid_skipped" not in report.meta
