"""Per-block depth-collapse checks, read directly against `_diag_common`'s
`load_bundle` harness (see module docstring there for why: the harness exists
so a check like this one is never written against a hand-mirrored forward
that can drift from the real model).

Three of the four checks here read weights only, which is what makes them
affordable on the *round* axis where a forward pass per round is not.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from stern.models.vit.mlp import GELUMlp, SwiGLUMlp
from stern.evaluation.diagnostics.common import (
    DiagBundle,
    DiagReport,
    LADDER_REVISION,
    MetricValue,
    forward_with_hidden_states,
    replicate,
)
from stern.evaluation.diagnostics.reconstruction import patch_stats, temporal_split

# The outlier-token count `cka_matrix`'s "trimmed" reading uses by default --
# Davari et al.'s finding is that a handful of large-norm tokens can dominate
# CKA, so this is a small constant rather than a fraction of the sequence.
DEFAULT_TRIM_OUTLIER_TOKENS = 5


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


# Two independent inits sit at a relative distance of about sqrt(2); a run's
# trained weights stay well inside 1 of their own init (rung 1: 0.04-0.11 per
# matrix against init(0), 1.41 against init(42) and init(43)).
MAX_RELATIVE_DISTANCE_FROM_OWN_INIT = 1.0


def _relative_distance(bundle: DiagBundle, ref_bundle: DiagBundle) -> float:
    """Median over weight matrices of `||W - W_ref|| / ||W_ref||`.

    Per matrix, not pooled: LayerNorm scales and a fixed sin-cos table are
    identical across inits, and pooled into one norm they would hide the
    sqrt(2) gap between two independent inits. Zero-initialised matrices
    carry no init to compare against and are left out.
    """
    ref = ref_bundle.encoder.state_dict()
    ratios = []
    for key, value in bundle.encoder.state_dict().items():
        if not value.dtype.is_floating_point or value.dim() < 2:
            continue
        ref_norm = ref[key].float().norm().item()
        if ref_norm > 0.0:
            ratios.append((value.float() - ref[key].float()).norm().item() / ref_norm)
    return float(np.median(ratios))


def _is_valid_init_reference(bundle: DiagBundle, ref_bundle: DiagBundle) -> bool:
    """`ref_bundle` is accepted only as the run's own recorded init or its
    round n-1 -- never an arbitrary bundle. At the round counts this project
    reaches, two independent random inits are farther apart than an init is
    from its own trained checkpoint, so a mismatched reference would produce
    a plausible-looking, entirely meaningless depth curve with no signal in
    the output that would reveal the mistake.

    A matching `init_seed` is not taken on trust: the seed is declared by the
    caller, so the weights must also sit inside
    `MAX_RELATIVE_DISTANCE_FROM_OWN_INIT` of the rebuilt init. A wrong seed
    lands near sqrt(2) and is refused.
    """
    if ref_bundle is bundle:
        return True
    if (
        bundle.init_seed is not None
        and bundle.init_seed == ref_bundle.init_seed
        and _relative_distance(bundle, ref_bundle) < MAX_RELATIVE_DISTANCE_FROM_OWN_INIT
    ):
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
            "own init (matching init_seed, and close enough to it) or its round "
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


def cka(x: "np.ndarray | torch.Tensor", y, *, trim_outlier_tokens: int = 0) -> float:
    """Linear Centered Kernel Alignment between two `(N, D)` token matrices.

    `trim_outlier_tokens` drops the `k` tokens with the largest combined
    (x, y) norm before scoring -- Davari et al.'s finding is that a handful
    of huge-norm tokens can dominate CKA and mask a real depth signal.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if trim_outlier_tokens > 0:
        norms = np.linalg.norm(x, axis=1) + np.linalg.norm(y, axis=1)
        n_keep = max(len(norms) - trim_outlier_tokens, 1)
        keep = np.argsort(norms)[:n_keep]
        x, y = x[keep], y[keep]

    xc = x - x.mean(axis=0, keepdims=True)
    yc = y - y.mean(axis=0, keepdims=True)
    hsic = np.linalg.norm(yc.T @ xc, ord="fro") ** 2
    norm_x = np.linalg.norm(xc.T @ xc, ord="fro")
    norm_y = np.linalg.norm(yc.T @ yc, ord="fro")
    denom = norm_x * norm_y
    if denom < 1e-12:
        # A (near-)constant representation has no similarity structure;
        # 0.0 would read as "maximally different" -- a false depth signal.
        return float("nan")
    return float(hsic / denom)


def _hidden_states(
    bundle: DiagBundle, pixel_values: torch.Tensor
) -> list[torch.Tensor]:
    """`forward_with_hidden_states`'s output, register-stripped -- CKA is
    scored over patch tokens only, never the register tokens that are never
    part of the patch grid."""
    states = forward_with_hidden_states(bundle.encoder, pixel_values)
    return [bundle.encoder.registers.strip(state) for state in states]


def _flat_tokens(state: torch.Tensor) -> "np.ndarray":
    tokens = state.detach().cpu().numpy()
    return tokens.reshape(-1, tokens.shape[-1])


def cka_matrix(
    bundle: DiagBundle, pixel_values: torch.Tensor
) -> dict[str, "np.ndarray"]:
    """Block-by-block CKA -- always `full`, `trimmed`, and `diff` together
    (design.md: reporting only one would silently discard the evidence that
    distinguishes a real depth signal from an outlier-token artifact)."""
    states = [_flat_tokens(s) for s in _hidden_states(bundle, pixel_values)]
    n = len(states)
    full = np.zeros((n, n))
    trimmed = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            full[i, j] = cka(states[i], states[j])
            trimmed[i, j] = cka(
                states[i], states[j], trim_outlier_tokens=DEFAULT_TRIM_OUTLIER_TOKENS
            )
    return {"full": full, "trimmed": trimmed, "diff": full - trimmed}


def cka_between(
    bundle_a: DiagBundle, bundle_b: DiagBundle, pixel_values: torch.Tensor
) -> "np.ndarray":
    """Depth-matched CKA between two bundles' hidden states, one score per
    shared depth index."""
    states_a = _hidden_states(bundle_a, pixel_values)
    states_b = _hidden_states(bundle_b, pixel_values)
    return np.array(
        [
            cka(_flat_tokens(a), _flat_tokens(b))
            for a, b in zip(states_a, states_b, strict=True)
        ]
    )


def _z_target(pixel_values: torch.Tensor, bundle: DiagBundle) -> torch.Tensor:
    """Time-averaged z-space patch target -- the same target
    `diag_reconstruction`'s temporal-holdout metrics score against."""
    tubelet_size = int(bundle.encoder.model_cfg.tubelet_size)
    patch_size = int(bundle.encoder.model_cfg.patch_size)
    target, mu, sigma, _grid = patch_stats(
        pixel_values, tubelet_size, patch_size, per_frame=False
    )
    return (target - mu) / (sigma + 1e-6)


def _fit_depth_head(
    bundle: DiagBundle,
    pixel_values: torch.Tensor,
    layer: int,
    *,
    keep: torch.Tensor,
    steps: int = 200,
    seed: int = 0,
) -> nn.Linear:
    """Fits a small linear head from `layer`'s token output to the
    time-averaged z-space patch target, fitting ONLY on the `keep`
    (visible) token split -- never the model's frozen final-block decoder,
    and never scored/fit on the held-out split here (that is
    `decode_from_depth`'s job).

    `keep` is required and keyword-only so a caller cannot fit on the full
    (visible + held-out) token set by omission.
    """
    torch.manual_seed(seed)
    states = _hidden_states(bundle, pixel_values)
    tokens = states[layer]
    embed_dim = tokens.shape[-1]

    target_z = _z_target(pixel_values, bundle)
    out_dim = target_z.shape[-1]

    head = nn.Linear(embed_dim, out_dim).to(tokens.device)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-2)

    x_keep = tokens[:, keep, :].reshape(-1, embed_dim).detach()
    y_keep = target_z[:, keep, :].reshape(-1, out_dim).detach()

    for _ in range(steps):
        optimizer.zero_grad()
        loss = ((head(x_keep) - y_keep) ** 2).mean()
        loss.backward()
        optimizer.step()
    return head


def decode_from_depth(
    bundle: DiagBundle,
    pixel_values: torch.Tensor,
    layers: list[int],
    *,
    n_seeds: int = 3,
) -> dict[int, MetricValue]:
    """Per-depth decodability: for each requested layer, fit a linear head
    (identical capacity/fit budget at every depth) on the visible/`keep`
    split and score its held-out MSE on `mask`, replicated over `n_seeds`
    independent fits (fresh random init per seed).

    Row-held-out, not information-held-out: `forward_with_hidden_states` is
    fully bidirectional, so every `mask` token has already attended to every
    `keep` token before the head sees it. This MSE answers "how linearly
    decodable is patch content from block k" and is NOT on
    `diag_reconstruction.holdout_mse_z`'s scale -- that metric's encoder only
    ever sees the `keep` tokens -- so never read the two side by side."""
    grid = bundle.encoder.token_grid
    keep, mask, _cut = temporal_split(grid, pixel_values.device, keep_frac=0.5)
    target_z = _z_target(pixel_values, bundle)

    result: dict[int, MetricValue] = {}
    for layer in layers:

        def _sample(seed: int, layer: int = layer) -> float:
            head = _fit_depth_head(bundle, pixel_values, layer, keep=keep, seed=seed)
            states = _hidden_states(bundle, pixel_values)
            with torch.no_grad():
                pred_mask = head(states[layer][:, mask, :])
            target_mask = target_z[:, mask, :]
            return float(((pred_mask - target_mask) ** 2).mean())

        result[layer] = replicate(_sample, n=n_seeds, seed=0)
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
        try:
            dfi = delta_from_init(bundle, ref_bundle)
        except ValueError as exc:
            # A random-init reference is valid for residual_branch_norm but
            # not here; skip this check alone rather than lose all six.
            delta_report = DiagReport(
                metrics={},
                arrays={},
                meta={
                    "LADDER_REVISION": LADDER_REVISION,
                    "input_kind": "real",
                    "skipped": str(exc),
                },
            )
        else:
            delta_report = DiagReport(
                metrics={
                    f"block_{i}": MetricValue.deterministic_value(v)
                    for i, v in dfi.items()
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

    if real_input is None:
        cka_matrix_report = DiagReport(
            metrics={},
            arrays={},
            meta={
                "LADDER_REVISION": LADDER_REVISION,
                "input_kind": "real",
                "skipped": "no input tensor supplied",
            },
        )
        decode_report = DiagReport(
            metrics={},
            arrays={},
            meta={
                "LADDER_REVISION": LADDER_REVISION,
                "input_kind": "real",
                "skipped": "no input tensor supplied",
            },
        )
    else:
        cka_matrix_report = DiagReport(
            metrics={},
            arrays=cka_matrix(bundle, real_input),
            meta={"LADDER_REVISION": LADDER_REVISION, "input_kind": "real"},
        )
        layers = list(range(1, len(bundle.encoder.blocks) + 1))
        per_layer = decode_from_depth(bundle, real_input, layers)
        decode_report = DiagReport(
            metrics={f"layer_{i}": v for i, v in per_layer.items()},
            arrays={},
            meta={"LADDER_REVISION": LADDER_REVISION, "input_kind": "real"},
        )

    return {
        "residual_branch_norm": residual_report,
        "layerscale_gamma": layerscale_report,
        "branch_norm_ratio": branch_norm_report,
        "cka_matrix": cka_matrix_report,
        "decode_from_depth": decode_report,
        "delta_from_init": delta_report,
    }
