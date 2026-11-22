"""Plan 6 DiLoCo checks 1-5: does the merge's averaged pseudo-gradient agree?

Every check here is CPU `state_dict` arithmetic over one round captured by
`snapshot_mesh_checkpoints.sh` (Plan 1): no GPU, no shard reads, no forward
pass. All merge math is imported from `stern.mesh.outer_step`, never
re-derived, so a diagnostic can only disagree with the real merge if the
captured inputs do.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from stern.mesh.outer_step import (
    StateDict,
    average_deltas,
    data_weights,
    filter_stale_deltas,
    nesterov_outer_step,
    pseudo_gradient,
    to_fp32_state_dict,
)
from stern.models.videomae.mae import VideoMAEForPreTraining
from stern.models.videomae.masking import build_masker
from stern.mesh.pending_files import (
    parse_pending_filename,
    pending_files_for_round,
)
from stern.mesh.assignment import read_my_shards
from stern.mesh.run_outer_step import _compute_alphas, filter_nonfinite
from stern.evaluation.diagnostics.common import (
    DEFAULT_SEPARATION_K,
    LADDER_REVISION,
    DiagBundle,
    DiagReport,
    MetricValue,
    _raise_on_real_mismatch,
    paired_inputs,
)

# Fixed so `diagnostics.json` stays byte-for-byte reproducible (ssl-diag-05
# compares it exactly across invocations).
_FLOOR_SEED = 0
_CONTROL_SEED = 1

# Split-control workers differ from the single-node trajectory by 1% of the
# delta's own per-element scale. Scaled to the delta, not the weights: drift
# is far smaller than the weights, so a 1%-of-weights perturbation would
# swamp it and turn the ceiling into a second noise floor.
_SPLIT_PERTURBATION = 0.01

_CHECKS = (
    "agreement_matrix",
    "effective_replicas",
    "per_block_divergence",
    "merge_magnitude",
    "momentum_alignment",
)
_FORWARD_CHECKS = ("merge_regression", "counterfactual_sweep")
_SWEEP_RULES = ("nesterov", "average")
# The per-node field each weighting reads beyond `n_samples`.
_WEIGHTING_NEEDS = {"inverse_loss": "val_losses", "token": "n_steps"}
_SWEEP_WEIGHTINGS = ("data", "inverse_loss", "token")
# Default sweep points; the configured (lr, momentum) is always added.
_SWEEP_LRS = (0.1, 0.3, 0.5, 1.0)
_SWEEP_MOMENTA = (0.0,)
_SD_SOURCE = "masking seeds on one fixed window batch; window sampling not resampled"
_BELOW_FLOOR = "inconclusive: below quantization noise floor"
_QUANT_FLOOR_MODEL = (
    "max |delta cos| over node pairs under seeded uniform noise within +/- half "
    "the fp16 spacing at each element; a model of rounding, not a measurement"
)

# Anchored on the encoder: `VideoMAEForPreTraining` also has
# `decoder_blocks.N.*`, which a bare `blocks.N.` substring would fold in.
_ENCODER_BLOCK = re.compile(r"^encoder\.blocks\.(\d+)\.")

# fp16 has a 10-bit mantissa and a minimum normal exponent of -14; below
# 2**-14 values are subnormal with a fixed spacing of 2**-24.
_FP16_MANTISSA_BITS = 10
_FP16_MIN_NORMAL_EXP = -14


@dataclass(frozen=True)
class MergeInputs:
    """Everything one merge consumed, reconstructed from a captured round.

    `worker_deltas` is always `master - worker`, whichever payload kind the
    node pushed, and covers survivors of the stale-version filter only.
    """

    master: StateDict
    master_version: int | None
    worker_deltas: dict[str, StateDict]
    alphas: dict[str, float]
    momentum_buf: StateDict | None
    val_losses: dict[str, float]
    dropped_stale: tuple[str, ...]
    quant_noise_floor: float
    n_samples: dict[str, int]
    n_steps: dict[str, int]
    payload_kinds: dict[str, str]
    gaps: tuple[str, ...]
    dropped_nonfinite: tuple[str, ...] = ()


def _manifest_record(snapshot_dir: Path, round: int) -> dict:
    # The snapshot script appends, so a re-captured round appears twice; the
    # last record describes the files currently on disk.
    records = [
        record
        for line in (snapshot_dir / "manifest.jsonl").read_text().splitlines()
        if line.strip() and (record := json.loads(line))["round"] == round
    ]
    if not records:
        raise ValueError(
            f"no manifest.jsonl record for round {round} in {snapshot_dir}"
        )
    return records[-1]


def load_merge_inputs(snapshot_dir: Path, round: int) -> MergeInputs:
    """Rebuild one round's merge inputs from its `manifest.jsonl` record.

    Each payload is dispatched on the state key it actually carries, not on
    the manifest's `payload_kind`: after the first merge a worker pushes an
    fp16 `delta_state_dict`, so assuming full weights would silently diff a
    delta against the master.
    """
    snapshot_dir = Path(snapshot_dir)
    record = _manifest_record(snapshot_dir, round)
    run_id = record["run_id"]
    if record["master"] is None:
        raise ValueError(f"round {round} has no captured master: {record['gaps']}")
    master_payload = torch.load(
        snapshot_dir / record["master"], map_location="cpu", weights_only=True
    )
    master = to_fp32_state_dict(master_payload["model_state_dict"])
    master_version = master_payload.get("master_version")

    checkpoints: dict[str, dict] = {}
    for path in sorted(
        pending_files_for_round(snapshot_dir / f"pending_round{round}", run_id, round)
    ):
        _, node = parse_pending_filename(path.stem, run_id)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if "n_samples" not in payload:
            raise ValueError(
                f"{path.name}: payload has no 'n_samples'; refusing to default a "
                "merge weight"
            )
        if "delta_state_dict" not in payload and "model_state_dict" not in payload:
            raise ValueError(
                f"{path.name}: payload has neither 'model_state_dict' nor "
                f"'delta_state_dict', found {sorted(payload)}"
            )
        checkpoints[node] = payload

    # Production's order: non-finite first (run_outer_step.merge_round), then
    # stale. A nan worker would otherwise make every replayed master nan.
    checkpoints, nonfinite = filter_nonfinite(checkpoints)
    survivors, dropped = filter_stale_deltas(checkpoints, master_version)
    deltas: dict[str, StateDict] = {}
    kinds: dict[str, str] = {}
    for node, payload in survivors.items():
        if "delta_state_dict" in payload:
            kinds[node] = "delta_state_dict"
            deltas[node] = to_fp32_state_dict(payload["delta_state_dict"])
        else:
            kinds[node] = "model_state_dict"
            deltas[node] = pseudo_gradient(
                master, to_fp32_state_dict(payload["model_state_dict"])
            )
    n_samples = {node: int(p["n_samples"]) for node, p in survivors.items()}

    momentum_buf = None
    if record.get("momentum"):
        momentum_payload = torch.load(
            snapshot_dir / record["momentum"], map_location="cpu", weights_only=True
        )
        momentum_buf = to_fp32_state_dict(momentum_payload["momentum_buf"])

    return MergeInputs(
        master=master,
        master_version=master_version,
        worker_deltas=deltas,
        alphas=data_weights(n_samples),
        momentum_buf=momentum_buf,
        val_losses={
            node: float(p["val_loss"])
            for node, p in survivors.items()
            if "val_loss" in p
        },
        dropped_stale=tuple(sorted(dropped)),
        quant_noise_floor=quantization_noise_floor(deltas),
        n_samples=n_samples,
        n_steps={
            node: int(p["n_steps"]) for node, p in survivors.items() if "n_steps" in p
        },
        payload_kinds=kinds,
        gaps=tuple(record.get("gaps", [])),
        dropped_nonfinite=tuple(sorted(nonfinite)),
    )


def _flatten(sd: StateDict, keys: list[str] | None = None) -> torch.Tensor:
    keys = (
        sorted(k for k in sd if torch.is_floating_point(sd[k]))
        if keys is None
        else keys
    )
    return torch.cat([sd[k].reshape(-1).to(torch.float32) for k in keys])


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine, or 0.0 when either side is all zeros (no direction to agree on)."""
    denom = float(a.norm()) * float(b.norm())
    return 0.0 if denom == 0.0 else float(a @ b) / denom


def _fp16_spacing(x: torch.Tensor) -> torch.Tensor:
    """Gap between adjacent fp16 values at each element's magnitude."""
    exponent = torch.floor(torch.log2(x.abs().clamp_min(2.0**_FP16_MIN_NORMAL_EXP)))
    return torch.pow(2.0, exponent - _FP16_MANTISSA_BITS)


def quantization_noise_floor(deltas: dict[str, StateDict]) -> float:
    """Largest shift fp16 rounding alone could induce in a pairwise cosine.

    Real deltas already arrive fp16, so re-casting them is the identity and
    would always report a floor of 0.0. Instead each element is perturbed by
    seeded uniform noise within +/- half the fp16 spacing at its magnitude --
    a model of rounding, not a measurement -- and the floor is the max
    |cos(original) - cos(perturbed)| over node pairs.
    """
    names = sorted(deltas)
    if len(names) < 2:
        return 0.0
    gen = torch.Generator().manual_seed(_FLOOR_SEED)
    flat = {n: _flatten(deltas[n]) for n in names}
    perturbed = {
        n: v + (torch.rand(v.shape, generator=gen) - 0.5) * _fp16_spacing(v)
        for n, v in flat.items()
    }
    return max(
        abs(_cos(flat[a], flat[b]) - _cos(perturbed[a], perturbed[b]))
        for i, a in enumerate(names)
        for b in names[i + 1 :]
    )


def _encoder_block(key: str) -> int | None:
    match = _ENCODER_BLOCK.match(key)
    return None if match is None else int(match.group(1))


def _block_keys(sd: StateDict, block: int) -> list[str]:
    return sorted(k for k in sd if _encoder_block(k) == block)


def _vectors(
    deltas: dict[str, StateDict], block: int | None = None
) -> dict[str, torch.Tensor]:
    """Each node's delta flattened, over all float keys or one encoder block."""
    return {
        n: _flatten(sd, None if block is None else _block_keys(sd, block))
        for n, sd in sorted(deltas.items())
    }


def _cos_matrix(vectors: dict[str, torch.Tensor]) -> np.ndarray:
    names = list(vectors)
    return np.array([[_cos(vectors[a], vectors[b]) for b in names] for a in names])


def _mean_offdiag(matrix: np.ndarray) -> float:
    n = matrix.shape[0]
    if n < 2:
        return float("nan")
    return float(matrix[~np.eye(n, dtype=bool)].mean())


def _rho(vectors: dict[str, torch.Tensor], alphas: dict[str, float]) -> float:
    """`||sum a_i d_i|| / sum a_i ||d_i||`: 1.0 when workers agree exactly."""
    merged = sum(alphas[n] * v for n, v in vectors.items())
    denom = sum(alphas[n] * float(v.norm()) for n, v in vectors.items())
    return float("nan") if denom == 0.0 else float(merged.norm()) / denom


def _m_eff(rho: float) -> float:
    return math.inf if rho == 0.0 else 1.0 / rho**2


def agreement_matrix(deltas: dict[str, StateDict], *, block: int | None = None) -> np.ndarray:
    """Pairwise cosine of pseudo-gradients, rows/columns in sorted node order."""
    return _cos_matrix(_vectors(deltas, block))


def effective_replicas(
    deltas: dict[str, StateDict], alphas: dict[str, float]
) -> tuple[float, float]:
    """`(M_eff, norm_spread)`, with `M_eff = 1/rho**2`.

    `norm_spread` is the coefficient of variation of `||d_i||`: an M_eff near
    1 can mean agreement or one worker dominating the average by norm, and
    only the spread tells those apart.
    """
    vectors = _vectors(deltas)
    norms = [float(v.norm()) for v in vectors.values()]
    return _m_eff(_rho(vectors, alphas)), statistics.pstdev(norms) / statistics.fmean(norms)


def per_block_divergence(
    deltas: dict[str, StateDict], master: StateDict, alphas: dict[str, float]
) -> dict[int, dict[str, float]]:
    """`rho`, `m_eff`, `rel_norm` and `mean_cos` for each encoder block."""
    blocks = sorted({b for k in master if (b := _encoder_block(k)) is not None})
    result = {}
    for block in blocks:
        vectors = _vectors(deltas, block)
        master_norm = float(_flatten(master, _block_keys(master, block)).norm())
        rho = _rho(vectors, alphas)
        result[block] = {
            "rho": rho,
            "m_eff": _m_eff(rho),
            "rel_norm": statistics.fmean(float(v.norm()) for v in vectors.values())
            / master_norm,
            "mean_cos": _mean_offdiag(_cos_matrix(vectors)),
        }
    return result


def effective_outer_lr(lr: float, momentum: float) -> float:
    """Steady-state outer stepsize: momentum's geometric series sums to `1/(1-m)`."""
    return lr / (1.0 - momentum)


def merge_magnitude(
    inputs: MergeInputs, *, outer_lr: float, outer_momentum: float
) -> dict[int, dict[str, float]]:
    """Per encoder block, the applied outer step relative to the master and to
    client drift (mean `||d_i||`).

    The step is `master - nesterov_outer_step(...)`, i.e. exactly what the merge
    applied including Nesterov's look-ahead, not a hand-rolled `lr*d + m`.
    """
    merged = average_deltas(inputs.worker_deltas, inputs.alphas)
    new_master, _ = nesterov_outer_step(
        inputs.master, merged, inputs.momentum_buf, lr=outer_lr, momentum=outer_momentum
    )
    step = {k: inputs.master[k] - new_master[k] for k in inputs.master}
    blocks = sorted({b for k in inputs.master if (b := _encoder_block(k)) is not None})
    result = {}
    for block in blocks:
        keys = _block_keys(inputs.master, block)
        step_norm = float(_flatten(step, keys).norm())
        drift = statistics.fmean(
            float(_flatten(d, keys).norm()) for d in inputs.worker_deltas.values()
        )
        result[block] = {
            "step_over_master": step_norm / float(_flatten(inputs.master, keys).norm()),
            "step_over_drift": step_norm / drift,
        }
    return result


def momentum_alignment(momentum_buf: StateDict | None, merged_delta: StateDict) -> float:
    """`cos(momentum_buf, merged_delta)`; 0.0 for a missing or all-zero buffer."""
    if momentum_buf is None:
        return 0.0
    keys = sorted(merged_delta)
    return _cos(_flatten(momentum_buf, keys), _flatten(merged_delta, keys))


def _control_inputs(master: StateDict, deltas: dict[str, StateDict]) -> MergeInputs:
    return MergeInputs(
        master=master,
        master_version=None,
        worker_deltas=deltas,
        alphas={n: 1.0 / len(deltas) for n in deltas},
        momentum_buf=None,
        val_losses={},
        dropped_stale=(),
        # Generated in fp32 and never quantized, so there is no rounding to bound.
        quant_noise_floor=0.0,
        n_samples={},
        n_steps={},
        payload_kinds={},
        gaps=(),
    )


def synthetic_controls(
    single_node_ckpt: StateDict, master: StateDict, *, n_workers: int
) -> dict[str, MergeInputs]:
    """Bracketing controls for absolute agreement/M_eff values.

    No published reference exists for any DiLoCo diagnostic on a ViT, so a
    real M_eff is only readable between these two: `split` (every worker on
    one trajectory, lightly perturbed -- the agreement ceiling) and `noise`
    (independent Gaussians at the same per-element scale -- the floor).
    Seeds are fixed so repeated calls are identical.
    """
    keys = sorted(k for k in master if torch.is_floating_point(master[k]))
    float_master = {k: master[k] for k in keys}
    base = pseudo_gradient(float_master, single_node_ckpt)
    scale = {k: float(base[k].pow(2).mean().sqrt()) for k in keys}
    split, noise = {}, {}
    for i in range(n_workers):
        gen = torch.Generator().manual_seed(_CONTROL_SEED + i)
        worker = {
            k: single_node_ckpt[k]
            + _SPLIT_PERTURBATION * scale[k] * torch.randn(base[k].shape, generator=gen)
            for k in keys
        }
        split[f"w{i}"] = pseudo_gradient(float_master, worker)
        noise[f"w{i}"] = {
            k: scale[k] * torch.randn(base[k].shape, generator=gen) for k in keys
        }
    return {
        "split": _control_inputs(master, split),
        "noise": _control_inputs(master, noise),
    }


def mae_loss_fn(
    model_cfg, windows: torch.Tensor, *, seed: int, device
) -> Callable[[StateDict], float]:
    """A scorer for full `VideoMAEForPreTraining` state dicts on `windows`.

    The mask is drawn once, here, and every state dict the returned function
    scores sees that same mask (and the same decoder subsample): the merge
    regressions are differences between models, and a fresh mask per model
    would add mask noise to every difference. Global RNG state is untouched.
    """
    device = torch.device(device)
    model = VideoMAEForPreTraining(model_cfg).to(device).eval()
    windows = windows.to(device)
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(seed)
        keep, mask = build_masker(model_cfg)(
            windows, model.encoder.token_grid, device, windows.shape[0]
        )

    def score(state_dict: StateDict) -> float:
        result = model.load_state_dict(state_dict, strict=False)
        _raise_on_real_mismatch(result, model.encoder, "diag_diloco")
        with (
            torch.random.fork_rng(devices=[device] if device.type == "cuda" else []),
            torch.no_grad(),
        ):
            torch.manual_seed(seed)
            pred, target = model(windows, keep, mask)
            return float(model.compute_loss(pred, target))

    return score


def _baseline_losses(
    inputs: MergeInputs, loss_fn: Callable[[StateDict], float]
) -> tuple[dict[str, float], float]:
    """Each contributor's and the pre-merge master's loss. The logged per-node
    `val_loss` is never used: `mesh_round.py` computes it on each node's own
    shards, so it is not comparable across nodes."""
    contributors = {
        node: loss_fn({k: inputs.master[k] - v for k, v in delta.items()})
        for node, delta in sorted(inputs.worker_deltas.items())
    }
    return contributors, loss_fn(inputs.master)


def _regression(
    post: float,
    contributors: dict[str, float],
    alphas: dict[str, float],
    master_loss: float,
) -> dict[str, float]:
    return {
        "vs_alpha_mean": post - sum(alphas[n] * loss for n, loss in contributors.items()),
        "vs_best_worker": post - min(contributors.values()),
        "vs_prev_master": post - master_loss,
    }


def merge_regression(
    inputs: MergeInputs,
    loss_fn: Callable[[StateDict], float],
    *,
    outer_lr: float,
    outer_momentum: float,
) -> dict[str, float]:
    """Post-merge loss minus each of three baselines; positive means the merge
    hurt. `inputs.master` is the pre-merge master, i.e. the previous round's
    merge result, so it is the "previous master" baseline."""
    post, _ = nesterov_outer_step(
        inputs.master,
        average_deltas(inputs.worker_deltas, inputs.alphas),
        inputs.momentum_buf,
        lr=outer_lr,
        momentum=outer_momentum,
    )
    contributors, master_loss = _baseline_losses(inputs, loss_fn)
    return _regression(loss_fn(post), contributors, inputs.alphas, master_loss)


def _sweep_alphas(
    inputs: MergeInputs, weighting: str
) -> tuple[dict[str, float] | None, str | None]:
    """`(production alphas, None)` for `weighting`, or `(None, reason)` when a
    survivor lacks the field it reads or production would refuse its value --
    defaulting either would fabricate a weighting."""
    nodes = sorted(inputs.worker_deltas)
    field = _WEIGHTING_NEEDS.get(weighting)
    missing = [n for n in nodes if field and n not in getattr(inputs, field)]
    if missing:
        return None, f"{field} missing for {missing}"
    checkpoints = {
        n: {
            "n_samples": inputs.n_samples[n],
            **({"val_loss": inputs.val_losses[n]} if n in inputs.val_losses else {}),
            **({"n_steps": inputs.n_steps[n]} if n in inputs.n_steps else {}),
        }
        for n in nodes
    }
    try:
        return _compute_alphas(checkpoints, weighting), None
    except ValueError as exc:
        return None, str(exc)


def counterfactual_sweep(
    inputs: MergeInputs,
    loss_fn: Callable[[StateDict], float],
    *,
    grid: list[tuple[float, float]],
    rules: list[str],
    weightings: list[str],
    configured: tuple[float, float],
) -> list[dict]:
    """One replayed merge per (rule, weighting, lr, momentum) cell, no retraining.

    Contributors and the pre-merge master are the same in every cell, so they
    are scored once. `average` is nesterov at (lr=1, momentum=0), where the
    momentum buffer drops out, so it is one cell per weighting rather than a
    copy per grid point.
    """
    unknown = sorted(set(rules) - set(_SWEEP_RULES))
    if unknown:
        raise ValueError(f"unknown sweep rule(s) {unknown}; supported: {_SWEEP_RULES}")
    if min(lr for lr, _ in grid) >= configured[0]:
        raise ValueError(
            f"sweep grid must reach below the configured outer_lr {configured[0]}"
        )
    if tuple(configured) not in {tuple(cell) for cell in grid}:
        raise ValueError(f"sweep grid lacks the configured cell {tuple(configured)}")

    contributors, master_loss = _baseline_losses(inputs, loss_fn)
    rows = []
    for weighting in weightings:
        alphas, _ = _sweep_alphas(inputs, weighting)
        if alphas is None:
            continue
        merged = average_deltas(inputs.worker_deltas, alphas)
        cells = []
        if "nesterov" in rules:
            cells += [("nesterov", lr, mu, inputs.momentum_buf) for lr, mu in grid]
        if "average" in rules:
            cells.append(("average", 1.0, 0.0, None))
        for rule, lr, mu, momentum_buf in cells:
            post, _ = nesterov_outer_step(
                inputs.master, merged, momentum_buf, lr=lr, momentum=mu
            )
            post_loss = loss_fn(post)
            rows.append(
                {
                    "rule": rule,
                    "weighting": weighting,
                    "outer_lr": lr,
                    "outer_momentum": mu,
                    "post_loss": post_loss,
                    **_regression(post_loss, contributors, alphas, master_loss),
                }
            )
    return rows


def assert_held_out(eval_shard: Path, assignments_path: Path, nodes) -> None:
    """Refuse an eval shard any contributor trained on.

    Mesh val shards live in `mesh_root/val/`, apart from training shards, but
    an overlap would tilt every regression toward "the merge helped", so it is
    checked rather than assumed. A node the assignments file does not know is
    an error too: reading it as "no shards" would hide exactly that overlap.
    """
    eval_shard = Path(eval_shard)
    if eval_shard.parent.name != "val":
        raise ValueError(
            f"eval shard {eval_shard} is not in a val/ directory; merge "
            "regression must be scored on held-out mesh val shards"
        )
    for node in sorted(nodes):
        try:
            shards = read_my_shards(Path(assignments_path), node)
        except KeyError as exc:
            raise ValueError(
                f"cannot establish that eval shard is held out from node {node!r}: {exc}"
            ) from exc
        if eval_shard.name in shards:
            raise ValueError(
                f"eval shard {eval_shard.name} is in node {node!r}'s training "
                f"shards ({assignments_path})"
            )


def history_only(metrics_dir: Path) -> dict[int, DiagReport]:
    """Per-round contributor count and val-loss dispersion from worker metrics.

    Reads `mesh_shared/metrics/<run_id>/<node>.jsonl` only, so it runs on any
    past run without a Plan 1 capture. Rounds are keyed as the JSONL records
    them (1-based), not as pending filenames number them.
    """
    by_round: dict[int, list[dict]] = defaultdict(list)
    for path in sorted(Path(metrics_dir).glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                by_round[int(record["round"])].append(record)
    reports = {}
    for round_num, records in sorted(by_round.items()):
        live = [r for r in records if not r.get("abandoned")]
        losses = [float(r["val_loss"]) for r in live]
        reports[round_num] = DiagReport(
            metrics={
                "n_contributors": MetricValue.deterministic_value(float(len(live))),
                "loss_dispersion": MetricValue.deterministic_value(
                    statistics.pstdev(losses) if losses else float("nan")
                ),
            },
            arrays={},
            meta={
                "LADDER_REVISION": LADDER_REVISION,
                "input_kind": "real",
                "n_abandoned": len(records) - len(live),
            },
        )
    return reports


def _verdict_meta(value: float, floor: float) -> dict:
    """Downgrade an agreement value fp16 rounding alone could produce. Never
    upgrades: a zero floor (nothing to bound) leaves the value unqualified."""
    if floor > 0.0 and abs(value) < floor:
        return {"verdict": _BELOW_FLOOR}
    return {}


def _block_metrics(per_block: dict[int, dict[str, float]]) -> dict[str, MetricValue]:
    return {
        f"block_{block}_{name}": MetricValue.deterministic_value(value)
        for block, values in per_block.items()
        for name, value in values.items()
    }


def _sampled(values: list[float]) -> MetricValue:
    return MetricValue(
        mean=statistics.fmean(values),
        sd=statistics.stdev(values),
        n=len(values),
        seeds=tuple(range(len(values))),
        deterministic=False,
    )


def _separation(value: MetricValue, *, above: str, below: str) -> str:
    band = DEFAULT_SEPARATION_K * value.sd
    if value.mean > band:
        return above
    if value.mean < -band:
        return below
    return "INCONCLUSIVE"


def _cell_key(row: dict) -> tuple:
    return (row["rule"], row["weighting"], row["outer_lr"], row["outer_momentum"])


def _forward_reports(
    merge: MergeInputs, inputs: dict, base_meta: dict
) -> dict[str, DiagReport]:
    """Checks 6-7, replicated over masking seeds. Each seed's sweep contains
    the configured cell (nesterov at the run's own weighting), which IS that
    seed's merge regression (same alphas, same step, same mask), so the
    regression is read off it rather than paid for twice."""

    def skipped(reason: str) -> dict[str, DiagReport]:
        return {
            name: DiagReport(
                metrics={}, arrays={}, meta={**base_meta, "skipped": reason}
            )
            for name in _FORWARD_CHECKS
        }

    required = ("diloco_eval_shard", "diloco_assignments", "model_cfg")
    if any(inputs.get(key) is None for key in required):
        return skipped("no diloco eval shard supplied")

    # A misconfigured eval set is the operator's error and raises; anything
    # after it (a refused weighting, an OOM) is this pair's failure alone and
    # must not take the five 12a reports down with it.
    assert_held_out(
        inputs["diloco_eval_shard"], inputs["diloco_assignments"], merge.worker_deltas
    )
    try:
        return _forward_reports_unguarded(merge, inputs, base_meta)
    except Exception as exc:  # noqa: BLE001 -- see above
        return skipped(f"forward-pass checks failed: {type(exc).__name__}: {exc}")


def _forward_reports_unguarded(
    merge: MergeInputs, inputs: dict, base_meta: dict
) -> dict[str, DiagReport]:
    n_replicates = inputs["diloco_n_replicates"]
    if n_replicates < 3:
        # MetricValue needs n >= 3; failing here names the cause instead of
        # surfacing as a StatisticsError after every forward pass has run.
        raise ValueError(f"diloco_n_replicates must be >= 3, got {n_replicates}")
    weighting = inputs.get("diloco_weighting", "data")
    unavailable = {
        w: reason
        for w in _SWEEP_WEIGHTINGS
        if (reason := _sweep_alphas(merge, w)[1]) is not None
    }
    if weighting in unavailable:
        raise ValueError(
            f"the run's own weighting {weighting!r} cannot be replayed: "
            f"{unavailable[weighting]}"
        )
    model_cfg = inputs["model_cfg"]
    windows, _ = paired_inputs(
        inputs["diloco_eval_shard"],
        n_windows=inputs["n_windows"],
        num_frames=model_cfg.num_frames,
        image_size=model_cfg.image_size,
        seed=inputs["seed"],
    )
    device = inputs.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    lr, mu = inputs["diloco_outer_lr"], inputs["diloco_outer_momentum"]
    grid = [
        (a, b)
        for a in sorted({*_SWEEP_LRS, lr})
        for b in sorted({*_SWEEP_MOMENTA, mu})
    ]
    configured = ("nesterov", weighting, lr, mu)

    per_seed = []
    for seed in range(n_replicates):
        rows = counterfactual_sweep(
            merge,
            mae_loss_fn(model_cfg, windows, seed=seed, device=device),
            grid=grid,
            rules=list(_SWEEP_RULES),
            weightings=list(_SWEEP_WEIGHTINGS),
            configured=(lr, mu),
        )
        per_seed.append({_cell_key(row): row for row in rows})

    regression = {
        key: _sampled([cells[configured][key] for cells in per_seed])
        for key in ("vs_alpha_mean", "vs_best_worker", "vs_prev_master")
    }
    cells = list(per_seed[0])
    post = {c: [seed_cells[c]["post_loss"] for seed_cells in per_seed] for c in cells}
    best = min(cells, key=lambda c: statistics.fmean(post[c]))
    gain = _sampled([b - c for b, c in zip(post[best], post[configured])])
    meta = {
        **base_meta,
        "sd_source": _SD_SOURCE,
        "eval_shard": str(inputs["diloco_eval_shard"]),
    }
    return {
        "merge_regression": DiagReport(
            metrics=regression,
            arrays={},
            meta={
                **meta,
                "verdicts": {
                    key: _separation(value, above="HURT", below="HELPED")
                    for key, value in regression.items()
                },
            },
        ),
        "counterfactual_sweep": DiagReport(
            metrics={
                "configured_post_loss": _sampled(post[configured]),
                "best_post_loss": _sampled(post[best]),
                "best_minus_configured": gain,
            },
            arrays={},
            meta={
                **meta,
                "grid": [
                    {
                        **dict(zip(("rule", "weighting", "outer_lr", "outer_momentum"), c)),
                        "post_loss_mean": statistics.fmean(post[c]),
                        "post_loss_sd": statistics.stdev(post[c]),
                        "vs_alpha_mean_mean": statistics.fmean(
                            seed_cells[c]["vs_alpha_mean"] for seed_cells in per_seed
                        ),
                    }
                    for c in cells
                ],
                "best_cell": dict(
                    zip(("rule", "weighting", "outer_lr", "outer_momentum"), best)
                ),
                "configured_cell": dict(
                    zip(("rule", "weighting", "outer_lr", "outer_momentum"), configured)
                ),
                "weightings_unavailable": unavailable,
                # best <= configured by construction, so only one side exists.
                "verdict": _separation(gain, above="INCONCLUSIVE", below="CONFIG_CHANGE"),
            },
        ),
    }


def check(bundle: DiagBundle, inputs: dict) -> dict[str, DiagReport]:
    """Checks 1-5 on one captured round, keyed like `diag_depth.check`.

    `bundle` is unused: every check reads the captured merge inputs, not the
    loaded encoder. The orchestrator calls every check on every run, so a
    missing snapshot is a skip report per check, never an exception.
    """
    if "diloco_snapshot_dir" not in inputs:
        return {
            name: DiagReport(
                metrics={},
                arrays={},
                meta={
                    "LADDER_REVISION": LADDER_REVISION,
                    "input_kind": "real",
                    "skipped": inputs.get(
                        "diloco_skip_reason", "no diloco snapshot supplied"
                    ),
                },
            )
            for name in _CHECKS + _FORWARD_CHECKS
        }

    merge = load_merge_inputs(Path(inputs["diloco_snapshot_dir"]), inputs["diloco_round"])
    outer_lr = inputs["diloco_outer_lr"]
    outer_momentum = inputs["diloco_outer_momentum"]
    base_meta = {
        "LADDER_REVISION": LADDER_REVISION,
        "input_kind": "real",
        "gaps": list(merge.gaps),
        "dropped_stale": list(merge.dropped_stale),
        "dropped_nonfinite": list(merge.dropped_nonfinite),
    }

    # D8: the lexicographically first survivor stands in for a single-node run.
    source = min(merge.worker_deltas)
    single = {k: merge.master[k] - v for k, v in merge.worker_deltas[source].items()}
    controls = synthetic_controls(single, merge.master, n_workers=len(merge.worker_deltas))
    matrix = agreement_matrix(merge.worker_deltas)
    mean_cos = _mean_offdiag(matrix)
    m_eff, norm_spread = effective_replicas(merge.worker_deltas, merge.alphas)
    absolute_meta = {
        **base_meta,
        "quant_noise_floor": merge.quant_noise_floor,
        "quant_floor_model": _QUANT_FLOOR_MODEL,
        "synthetic_split_m_eff": effective_replicas(
            controls["split"].worker_deltas, controls["split"].alphas
        )[0],
        "synthetic_noise_m_eff": effective_replicas(
            controls["noise"].worker_deltas, controls["noise"].alphas
        )[0],
        "control_source_node": source,
        **_verdict_meta(mean_cos, merge.quant_noise_floor),
    }

    if "delta_state_dict" in merge.payload_kinds.values():
        # Error-feedback residuals correlate round to round by construction --
        # the very axis momentum alignment measures -- so a value here would
        # be an artefact of the push path, not of the merge.
        momentum_report = DiagReport(
            metrics={},
            arrays={},
            meta={
                **base_meta,
                "skipped": "momentum alignment unresolvable under error feedback",
            },
        )
    else:
        alignment = momentum_alignment(
            merge.momentum_buf, average_deltas(merge.worker_deltas, merge.alphas)
        )
        momentum_report = DiagReport(
            metrics={"momentum_alignment": MetricValue.deterministic_value(alignment)},
            arrays={},
            meta=dict(base_meta),
        )

    return {
        "agreement_matrix": DiagReport(
            metrics={"mean_cos": MetricValue.deterministic_value(mean_cos)},
            arrays={"agreement": matrix},
            meta={**absolute_meta, "nodes": sorted(merge.worker_deltas)},
        ),
        "effective_replicas": DiagReport(
            metrics={
                "m_eff": MetricValue.deterministic_value(m_eff),
                "norm_spread": MetricValue.deterministic_value(norm_spread),
                "n_workers": MetricValue.deterministic_value(
                    float(len(merge.worker_deltas))
                ),
            },
            arrays={},
            meta=dict(absolute_meta),
        ),
        "per_block_divergence": DiagReport(
            metrics=_block_metrics(
                per_block_divergence(merge.worker_deltas, merge.master, merge.alphas)
            ),
            arrays={},
            meta=dict(base_meta),
        ),
        "merge_magnitude": DiagReport(
            metrics={
                "effective_outer_lr": MetricValue.deterministic_value(
                    effective_outer_lr(outer_lr, outer_momentum)
                ),
                **_block_metrics(
                    merge_magnitude(
                        merge, outer_lr=outer_lr, outer_momentum=outer_momentum
                    )
                ),
            },
            arrays={},
            meta=dict(base_meta),
        ),
        "momentum_alignment": momentum_report,
        **_forward_reports(merge, inputs, base_meta),
    }
