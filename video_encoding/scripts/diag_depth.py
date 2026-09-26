"""Per-block depth-collapse checks, read directly against `_diag_common`'s
`load_bundle` harness (see module docstring there for why: the harness exists
so a check like this one is never written against a hand-mirrored forward
that can drift from the real model).

Three of the four checks here read weights only, which is what makes them
affordable on the *round* axis where a forward pass per round is not.
"""

from __future__ import annotations

import torch
from torch import nn

from video_encoding.models.ViT.mlp import GELUMlp, SwiGLUMlp
from video_encoding.scripts._diag_common import (
    DiagBundle,
    DiagReport,
    LADDER_REVISION,
    MetricValue,
    forward_with_hidden_states,
)


def _branch_norm(qkv_weight: torch.Tensor, proj_weight: torch.Tensor) -> float:
    embed_dim = qkv_weight.shape[1]
    w_v = qkv_weight[2 * embed_dim : 3 * embed_dim]
    return (proj_weight.norm() * w_v.norm()).item()


def _mlp_branch_norm(mlp: nn.Module) -> float:
    if isinstance(mlp, GELUMlp):
        return (mlp.net[3].weight.norm() * mlp.net[0].weight.norm()).item()
    if isinstance(mlp, SwiGLUMlp):
        return (
            mlp.down.weight.norm() * mlp.gate.weight.norm() * mlp.up.weight.norm()
        ).item()
    raise ValueError(f"unsupported MLP type {type(mlp).__name__}")


def residual_branch_norm(
    bundle: DiagBundle, ref_bundle: DiagBundle
) -> dict[int, dict[str, MetricValue]]:
    """Per-block `attn`/`mlp` residual-branch norm, normalised against
    `ref_bundle`. `ref_bundle` is required positionally -- no default, no
    inference -- because a checkpoint silently compared against itself
    reports 1.0 for every block regardless of how broken it is."""
    result: dict[int, dict[str, MetricValue]] = {}
    for i, (block, ref_block) in enumerate(
        zip(bundle.encoder.blocks, ref_bundle.encoder.blocks, strict=True)
    ):
        attn_val = _branch_norm(block.attn.qkv.weight, block.attn.proj.weight)
        ref_attn_val = _branch_norm(
            ref_block.attn.qkv.weight, ref_block.attn.proj.weight
        )
        mlp_val = _mlp_branch_norm(block.mlp)
        ref_mlp_val = _mlp_branch_norm(ref_block.mlp)
        result[i] = {
            "attn": MetricValue.deterministic_value(attn_val / ref_attn_val),
            "mlp": MetricValue.deterministic_value(mlp_val / ref_mlp_val),
        }
    return result


def _normalise_attn_output(output: torch.Tensor | tuple) -> torch.Tensor:
    """`ViTBlock.forward` does `attn_out, _ = self.attn(...)` -- a hooked
    `.attn` module can return either the raw tensor or that 2-tuple,
    depending on caller. Normalise both to the tensor."""
    return output[0] if isinstance(output, tuple) else output


def branch_norm_ratio(
    bundle: DiagBundle, pixel_values: torch.Tensor
) -> dict[int, dict[str, MetricValue]]:
    """Per-block attention-branch, MLP-branch, and residual-stream norms, plus
    each branch's ratio to the residual -- the quantity the depth-collapse
    reading is made on. Captures branch outputs via temporary forward hooks
    on the real `.attn`/`.mlp` submodules rather than re-deriving
    `ViTBlock.forward` by hand, which is what let `export_vit_internals`
    drift from the model it described."""
    encoder = bundle.encoder
    attn_outputs: list[torch.Tensor] = []
    mlp_outputs: list[torch.Tensor] = []
    handles = []

    def _attn_hook(_module, _inputs, output):
        attn_outputs.append(_normalise_attn_output(output))

    def _mlp_hook(_module, _inputs, output):
        mlp_outputs.append(output)

    try:
        for block in encoder.blocks:
            handles.append(block.attn.register_forward_hook(_attn_hook))
            handles.append(block.mlp.register_forward_hook(_mlp_hook))
        with torch.no_grad():
            states = forward_with_hidden_states(encoder, pixel_values)
    finally:
        for handle in handles:
            handle.remove()

    result: dict[int, dict[str, MetricValue]] = {}
    for i, block in enumerate(encoder.blocks):
        residual_norm = states[i + 1].norm().item()
        attn_norm = block.layer_scale1(attn_outputs[i]).norm().item()
        mlp_norm = block.layer_scale2(mlp_outputs[i]).norm().item()
        result[i] = {
            "attn": MetricValue.deterministic_value(attn_norm),
            "mlp": MetricValue.deterministic_value(mlp_norm),
            "residual": MetricValue.deterministic_value(residual_norm),
            "attn_ratio": MetricValue.deterministic_value(attn_norm / residual_norm),
            "mlp_ratio": MetricValue.deterministic_value(mlp_norm / residual_norm),
        }
    return result


def layerscale_gamma(bundle: DiagBundle) -> dict[int, float] | None:
    """Per-block RMS of LayerScale's gamma, or `None` when LayerScale is
    disabled (layer_scale1 is nn.Identity) -- never `{}`, since `{}` and
    "every block still at init" are indistinguishable downstream."""
    if isinstance(bundle.encoder.blocks[0].layer_scale1, nn.Identity):
        return None
    result: dict[int, float] = {}
    for i, block in enumerate(bundle.encoder.blocks):
        gamma = block.layer_scale1.gamma
        result[i] = torch.sqrt(torch.mean(gamma**2)).item()
    return result


def _is_valid_init_reference(bundle: DiagBundle, ref_bundle: DiagBundle) -> bool:
    """`ref_bundle` is accepted only as the run's own recorded init or its
    round n-1 -- never an arbitrary bundle. At the round counts this project
    reaches, two independent random inits are farther apart than an init is
    from its own trained checkpoint, so a mismatched reference would produce
    a plausible-looking, entirely meaningless depth curve with no signal in
    the output that would reveal the mistake.

    `bundle.meta` does not exist on today's DiagBundle (the snapshotter
    records no init_seed -- ssl-diag-01-foundation follow-up), so the
    same-init path is unreachable on real bundles and only identity and
    round n-1 work in production; `getattr` with a default keeps this ready
    for the field once it lands, rather than requiring a second edit here.
    """
    if ref_bundle is bundle:
        return True
    own_seed = getattr(bundle, "meta", {}).get("init_seed")
    ref_seed = getattr(ref_bundle, "meta", {}).get("init_seed")
    if own_seed is not None and own_seed == ref_seed:
        return True
    return (
        bundle.round is not None
        and ref_bundle.round is not None
        and ref_bundle.run_id == bundle.run_id
        and ref_bundle.round == bundle.round - 1
    )


def delta_from_init(bundle: DiagBundle, ref_bundle: DiagBundle) -> dict[int, float]:
    """Per-block L2 norm of the parameter delta against `ref_bundle`.

    This is a "did this block move" check, not a damage detector: zeroing an
    already-small tensor can remove less norm than training added to that
    block's other weights, so a damaged block is not guaranteed to separate
    from a healthy one here. `residual_branch_norm` is the mutilation
    detector; see its module docstring's rung-(-1) test.
    """
    if not _is_valid_init_reference(bundle, ref_bundle):
        raise ValueError(
            f"{bundle.run_id}: delta_from_init's reference must be the run's "
            "own recorded init (matching meta['init_seed']) or its round "
            f"n-1, not {ref_bundle.run_id!r} at round {ref_bundle.round!r}"
        )
    result: dict[int, float] = {}
    for i, (block, ref_block) in enumerate(
        zip(bundle.encoder.blocks, ref_bundle.encoder.blocks, strict=True)
    ):
        total = 0.0
        for (_, p), (_, ref_p) in zip(
            block.named_parameters(), ref_block.named_parameters(), strict=True
        ):
            total += (p - ref_p).norm().item() ** 2
        result[i] = total**0.5
    return result


def check(bundle: DiagBundle, inputs: dict) -> dict[str, DiagReport]:
    """Plan 5's orchestrator reaches every check in this module only through
    here, so a check `check()` never calls is dead code. Always emits one
    report per check -- a skip is a report with empty metrics and
    `meta["skipped"]`, never an omission, since an absent report is
    indistinguishable from a check that was never defined."""
    ref_bundle = inputs.get("ref_bundle")

    if ref_bundle is None:
        residual_report = DiagReport(
            metrics={},
            arrays={},
            meta={
                "LADDER_REVISION": LADDER_REVISION,
                "input_kind": "real",
                "skipped": "no ref_bundle supplied",
            },
        )
        delta_report = DiagReport(
            metrics={},
            arrays={},
            meta={
                "LADDER_REVISION": LADDER_REVISION,
                "input_kind": "real",
                "skipped": "no ref_bundle supplied",
            },
        )
    else:
        rbn = residual_branch_norm(bundle, ref_bundle)
        metrics = {}
        for i, block_values in rbn.items():
            metrics[f"block_{i}_attn"] = block_values["attn"]
            metrics[f"block_{i}_mlp"] = block_values["mlp"]
        residual_report = DiagReport(
            metrics=metrics,
            arrays={},
            meta={"LADDER_REVISION": LADDER_REVISION, "input_kind": "real"},
        )
        dfi = delta_from_init(bundle, ref_bundle)
        delta_report = DiagReport(
            metrics={
                f"block_{i}": MetricValue.deterministic_value(v) for i, v in dfi.items()
            },
            arrays={},
            meta={"LADDER_REVISION": LADDER_REVISION, "input_kind": "real"},
        )

    gamma = layerscale_gamma(bundle)
    if gamma is None:
        layerscale_report = DiagReport(
            metrics={},
            arrays={},
            meta={
                "LADDER_REVISION": LADDER_REVISION,
                "input_kind": "real",
                "skipped": "layerscale disabled",
            },
        )
    else:
        layerscale_report = DiagReport(
            metrics={
                f"block_{i}": MetricValue.deterministic_value(v)
                for i, v in gamma.items()
            },
            arrays={},
            meta={"LADDER_REVISION": LADDER_REVISION, "input_kind": "real"},
        )

    real_input = inputs.get("real")
    if real_input is None:
        branch_norm_report = DiagReport(
            metrics={},
            arrays={},
            meta={
                "LADDER_REVISION": LADDER_REVISION,
                "input_kind": "real",
                "skipped": "no input tensor supplied",
            },
        )
    else:
        bnr = branch_norm_ratio(bundle, real_input)
        metrics = {}
        for i, block_values in bnr.items():
            for key, value in block_values.items():
                metrics[f"block_{i}_{key}"] = value
        branch_norm_report = DiagReport(
            metrics=metrics,
            arrays={},
            meta={"LADDER_REVISION": LADDER_REVISION, "input_kind": "real"},
        )

    return {
        "residual_branch_norm": residual_report,
        "layerscale_gamma": layerscale_report,
        "branch_norm_ratio": branch_norm_report,
        "delta_from_init": delta_report,
    }
