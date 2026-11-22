"""Four qualitative panels: embedding scatter, retrieval grid, mask-ratio grid,
attention rollout.

These are Tier C by construction and return no decision rule. Unlike the other
diagnostics, none of them is waiting on calibration. Attention rollout stays
Tier C for good: Abnar & Zuidema's rollout is not a faithful decomposition of
attribution.
"""

from __future__ import annotations

import numpy as np
import torch

from stern.evaluation.diagnostics.common import (
    DiagBundle,
    DiagReport,
    LADDER_REVISION,
    forward_with_hidden_states,
)
from stern.evaluation.diagnostics.reconstruction import predict_z, temporal_split


def embedding_scatter(tokens, clip_ids) -> "np.ndarray":
    """A 2-D PCA/SVD projection of `tokens` (one row per clip) -- not
    t-SNE, to avoid a scikit-learn (and transitive torchvision) dependency
    for a Tier-C-only panel."""
    tokens = np.asarray(tokens, dtype=np.float64)
    clip_ids = np.asarray(clip_ids)
    if tokens.shape[0] != clip_ids.shape[0]:
        raise ValueError(
            f"tokens has {tokens.shape[0]} rows but clip_ids has {clip_ids.shape[0]}"
        )
    centered = tokens - tokens.mean(axis=0, keepdims=True)
    u, s, _vt = np.linalg.svd(centered, full_matrices=False)
    n_components = min(2, s.shape[0])
    coords = u[:, :n_components] * s[:n_components]
    if n_components < 2:
        coords = np.pad(coords, ((0, 0), (0, 2 - n_components)))
    return coords


def retrieval_grid(query, gallery, k: int = 5) -> "np.ndarray":
    """Top-`k` cosine-similarity nearest neighbours in `gallery` per row of
    `query`, as gallery row indices."""
    query = np.asarray(query, dtype=np.float64)
    gallery = np.asarray(gallery, dtype=np.float64)
    q_norm = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-12)
    g_norm = gallery / (np.linalg.norm(gallery, axis=1, keepdims=True) + 1e-12)
    sim = q_norm @ g_norm.T
    k = min(k, gallery.shape[0])
    return np.argsort(-sim, axis=1, kind="stable")[:, :k]


@torch.no_grad()
def mask_ratio_grid(model, pixel_values, ratios, device) -> "np.ndarray":
    """One coarse per-patch z-value heatmap at token-grid resolution
    `(h_p, w_p)` per mask ratio. This is NOT a pixel-space reconstruction:
    `reconstruction.py` has no unpatchify inverse."""
    grid = model.encoder.token_grid
    t_p, h_p, w_p = grid
    pixel_values = pixel_values.to(device)

    panels = []
    for ratio in ratios:
        keep, mask, _cut = temporal_split(grid, device, keep_frac=1.0 - float(ratio))
        if keep.numel() == 0:
            raise ValueError(
                f"ratio={ratio} leaves an empty keep set at grid={grid} "
                "(temporal_split's cut rounded to 0) -- lower the ratio or "
                "grow num_frames so cut >= 1"
            )
        pred = predict_z(model, pixel_values, keep, mask)  # (B, N, D)
        magnitude = pred.abs().mean(dim=-1).mean(dim=0)  # (N,)
        panel = magnitude.reshape(t_p, h_p, w_p).mean(dim=0)  # (h_p, w_p)
        panels.append(panel.detach().cpu().numpy())
    return np.stack(panels, axis=0)


def attention_rollout(attn_per_layer) -> "np.ndarray":
    """Abnar & Zuidema rollout: `R^(l) = (0.5*A^(l) + 0.5*I) @ R^(l-1)`,
    `R^(0) = I`. Tier C permanently -- see module docstring."""
    first = np.asarray(attn_per_layer[0], dtype=np.float64)
    n = first.shape[-1]
    rollout = np.eye(n)
    for attn in attn_per_layer:
        a = np.asarray(attn, dtype=np.float64)
        combined = 0.5 * a + 0.5 * np.eye(n)
        rollout = combined @ rollout
    return rollout


def _attention_maps_by_block(
    bundle: DiagBundle, pixel_values: torch.Tensor
) -> list["np.ndarray"]:
    """Per-block attention maps, averaged over batch and heads -- `(N, N)`
    each. Uses `forward_with_hidden_states` for the residual-stream
    traversal; only the attention-weight extraction is block-local
    (`block.attn(h, need_weights=True)`), same pattern as
    `diag_representational.attention_entropy_by_block`."""
    states = forward_with_hidden_states(bundle.encoder, pixel_values)
    maps = []
    for i, block in enumerate(bundle.encoder.blocks):
        h = block.norm1(states[i])
        with torch.no_grad():
            _, attn_weights = block.attn(h, need_weights=True)
        maps.append(attn_weights.mean(dim=(0, 1)).detach().cpu().numpy())
    return maps


def check(bundle: DiagBundle, inputs: dict) -> dict[str, DiagReport]:
    """Wires all four panels into one `DiagReport`'s `arrays`, keyed
    `diag_qualitative_real`. When
    `bundle.model` is `None`, `mask_ratio_grid` is skipped with an explicit
    `meta` flag recording why -- never silently dropped -- and the other
    three panels are still produced."""
    pixel_values = inputs["real"]
    encoder = bundle.encoder

    with torch.no_grad():
        pooled = encoder(pixel_values).detach().cpu().numpy()
    clip_ids = np.arange(pooled.shape[0])

    arrays: dict[str, "np.ndarray"] = {
        "embedding_scatter": embedding_scatter(pooled, clip_ids),
        "retrieval_grid": retrieval_grid(pooled, pooled, k=min(5, pooled.shape[0])),
        "attention_rollout": attention_rollout(
            _attention_maps_by_block(bundle, pixel_values)
        ),
    }

    meta = {"LADDER_REVISION": LADDER_REVISION, "input_kind": "real"}
    if bundle.model is None:
        meta["mask_ratio_grid_skipped"] = "bundle.model is None (encoder-only bundle)"
    else:
        arrays["mask_ratio_grid"] = mask_ratio_grid(
            bundle.model, pixel_values, [0.5, 0.75, 0.9], pixel_values.device
        )

    return {"diag_qualitative_real": DiagReport(metrics={}, arrays=arrays, meta=meta)}
