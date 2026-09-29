"""Per-block attention-map entropy, read off the shared residual-stream
traversal (`_diag_common.forward_with_hidden_states`) so the only
block-local thing done here is the attention-weight extraction itself --
never a second hand-mirrored copy of `ViTBlock.forward`'s residual update.

`attention_nmi` is cut per the design doc: reasoned redundant with entropy
in advance rather than measured, so this module ships entropy only.
"""

from __future__ import annotations

import numpy as np
import torch

from video_encoding.scripts._diag_common import (
    DiagBundle,
    DiagReport,
    LADDER_REVISION,
    MetricValue,
    forward_with_hidden_states,
)


def _entropy_of_attention_map(attn_map) -> float:
    """Mean Shannon entropy of a row-stochastic attention map's rows,
    averaged over every leading (batch/head/query-row) dimension."""
    attn = np.asarray(attn_map, dtype=np.float64)
    safe = np.where(attn > 0, attn, 1.0)  # log(1) = 0, so zero rows contribute 0
    row_entropy = -np.sum(attn * np.log(safe), axis=-1)
    return float(row_entropy.mean())


def attention_entropy_by_block(
    bundle: DiagBundle, pixel_values: torch.Tensor
) -> dict[int, MetricValue]:
    """Per-block mean attention entropy. Deterministic given a fixed
    `pixel_values` and the bundle's weights -- there is no sampling axis
    here (see `check` below).

    Reuses `forward_with_hidden_states` for the residual-stream traversal:
    `states[i]` is block `i`'s input, so `block.norm1(states[i])` plus
    `block.attn(..., need_weights=True)` recovers exactly the attention map
    that block's real forward pass computed, without re-deriving the
    layer_scale/drop_path residual update by hand.
    """
    states = forward_with_hidden_states(bundle.encoder, pixel_values)
    result: dict[int, MetricValue] = {}
    for i, block in enumerate(bundle.encoder.blocks):
        h = block.norm1(states[i])
        with torch.no_grad():
            _, attn_weights = block.attn(h, need_weights=True)
        result[i] = MetricValue.deterministic_value(
            _entropy_of_attention_map(attn_weights.detach().cpu().numpy())
        )
    return result


def check(bundle: DiagBundle, inputs: dict) -> dict[str, DiagReport]:
    """One `DiagReport` per input kind present in `inputs` (`"real"`/
    `"random"`), keyed `attention_entropy_<kind>`, each carrying one metric
    per block keyed `block{i}`.

    Values are deterministic by design, not a replication gap: a fixed
    `pixel_values` on a frozen eval-mode encoder has no sampling axis, the
    same reasoning `diag_depth.residual_branch_norm` uses.
    """
    reports: dict[str, DiagReport] = {}
    for kind in ("real", "random"):
        pixel_values = inputs.get(kind)
        if pixel_values is None:
            continue
        per_block = attention_entropy_by_block(bundle, pixel_values)
        reports[f"attention_entropy_{kind}"] = DiagReport(
            metrics={f"block{i}": v for i, v in per_block.items()},
            arrays={},
            meta={"LADDER_REVISION": LADDER_REVISION, "input_kind": kind},
        )
    return reports
