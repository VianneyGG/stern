"""One CLI, one output directory, three axes.

- ladder (`--ladder`): score every buildable rung, write `ladder.json`.
- run (`--run-ids`): score checkpoints against that ladder, write
  `diagnostics.json`.
- round (`--rounds` + `--snapshot-dir`): one run's trajectory across captured
  DiLoCo rounds, same `diagnostics.json` schema.

`render_report.py` turns `diagnostics.json` + `plots/` into `report.html`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pickle
import re
import traceback
from pathlib import Path
from typing import Callable

import numpy as np

from video_encoding.scripts import (
    diag_depth,
    diag_diloco,
    diag_finetune_afewva,
    diag_labelfree,
    diag_probe,
    diag_qualitative,
    diag_representational,
)
from video_encoding.scripts._diag_common import (
    LADDER,
    LADDER_REVISION,
    LADDER_RUN_IDS,
    DiagBundle,
    DiagReport,
    MetricValue,
    Rung,
    RungUnavailable,
    ladder_tau,
    load_bundle,
    load_ladder_rung,
    load_snapshot_bundle,
    paired_inputs,
    random_init_bundle,
    run_hyperparams,
)
from video_encoding.scripts.render_report import (
    CONTROLLED,
    NO_VERDICT,
    PANEL_OF,
    bad_direction,
    good_direction,
    metric_key,
)


def _as_list(check_fn: Callable) -> Callable:
    """Adapt a check whose dict keys only encode the input kind (e.g.
    `attention_entropy_real`) into a list result, so every report is named by
    its registry key and the kind is read from `meta["input_kind"]`."""

    def wrapped(bundle: DiagBundle, inputs: dict) -> list[DiagReport]:
        return list(check_fn(bundle, inputs).values())

    wrapped.__name__ = f"{check_fn.__module__}.check"
    return wrapped


# Registry key -> check(bundle, inputs), returning dict[str, DiagReport] (keys
# are the emitted check names) or list[DiagReport] (named by the registry key).
CHECKS: dict[str, Callable] = {
    "diag_depth": diag_depth.check,
    "attention_entropy_by_block": _as_list(diag_representational.check),
    "labelfree": _as_list(diag_labelfree.check),
    "diag_qualitative": _as_list(diag_qualitative.check),
    "probe_by_depth": diag_probe.check,
    "diag_diloco": diag_diloco.check,
    "finetune_afewva": diag_finetune_afewva.check,
}

# Registry key -> every check name it may emit.
EMITS: dict[str, frozenset[str]] = {
    "diag_depth": frozenset(
        {
            "residual_branch_norm",
            "layerscale_gamma",
            "branch_norm_ratio",
            "cka_matrix",
            "decode_from_depth",
            "delta_from_init",
        }
    ),
    "attention_entropy_by_block": frozenset({"attention_entropy_by_block"}),
    "labelfree": frozenset({"labelfree"}),
    "diag_qualitative": frozenset({"diag_qualitative"}),
    "probe_by_depth": frozenset({"probe_by_depth"}),
    "diag_diloco": frozenset(
        {
            "agreement_matrix",
            "effective_replicas",
            "per_block_divergence",
            "merge_magnitude",
            "momentum_alignment",
            "merge_regression",
            "counterfactual_sweep",
        }
    ),
    "finetune_afewva": frozenset({"finetune_afewva"}),
}

# Registry keys that must return both a `real` and a `random` report -- the
# pairing that separates data-dependent behaviour from architecture.
PAIRED: frozenset[str] = frozenset({"attention_entropy_by_block", "labelfree"})

# Emitted names whose deterministic metrics read weights only, so replication
# would return the identical number: they count as replicated. Any other
# deterministic metric is written `replicated: false`.
WEIGHTS_ONLY: frozenset[str] = frozenset(
    {"residual_branch_norm", "delta_from_init", "layerscale_gamma"}
)

# Registry keys that read no drawn window: the probe fits on AFEW-VA, so a
# second input draw repeats the identical (multi-hour) fit.
DRAW_INDEPENDENT: frozenset[str] = frozenset({"probe_by_depth"})

# Registry keys that always run, whatever `--checks` says: no checkpoint is
# accepted on intrinsic diagnostics alone.
GATES: frozenset[str] = frozenset({"finetune_afewva"})


def validate_registry(
    checks: dict[str, Callable], emits: dict[str, frozenset[str]]
) -> None:
    """Every registry key declares what it emits, and every emitted name has a
    control or is explicitly verdict-free -- a registration bug caught at
    import, not as a struck-through cell in a rendered report."""
    missing = sorted(set(checks) - set(emits))
    if missing:
        raise ValueError(f"CHECKS keys with no EMITS entry: {missing}")
    uncontrolled = sorted(
        name
        for key in checks
        for name in emits[key]
        if name not in CONTROLLED and name not in NO_VERDICT
    )
    if uncontrolled:
        raise ValueError(
            f"emitted check names with no configured control: {uncontrolled} -- "
            "add each to render_report.CONTROLLED, or NO_VERDICT if it is "
            "qualitative-only"
        )


validate_registry(CHECKS, EMITS)


def normalise(
    registry_key: str,
    result: dict[str, DiagReport] | list[DiagReport],
    emits: frozenset[str],
) -> list[tuple[str, DiagReport]]:
    """A check's result as `(emitted name, report)` pairs."""
    if isinstance(result, dict):
        pairs = list(result.items())
    else:
        pairs = [(registry_key, report) for report in result]
    undeclared = sorted({name for name, _ in pairs} - emits)
    if undeclared:
        raise ValueError(
            f"{registry_key} emitted undeclared check name(s) {undeclared}; "
            f"EMITS[{registry_key!r}] declares {sorted(emits)}"
        )
    return pairs


def build_inputs(
    bundle: DiagBundle,
    *,
    shard_path: Path,
    model_cfg,
    seed: int,
    n_windows: int,
    ref_bundle: DiagBundle | None,
) -> dict:
    """Every input key a landed check reads. `ref_bundle` is the round n-1
    bundle on the round axis and None elsewhere."""
    real, random_batch = paired_inputs(
        shard_path,
        n_windows=n_windows,
        num_frames=model_cfg.num_frames,
        image_size=model_cfg.image_size,
        seed=seed,
    )
    return {
        "real": real,
        "random": random_batch,
        "shard_path": shard_path,
        "n_windows": n_windows,
        "seed": seed,
        "control_bundle": random_init_bundle(model_cfg, seed=seed),
        "ref_bundle": ref_bundle,
    }


# The run's merge settings a replay needs; each run's mesh_run_configs.json
# entry carries all three.
DILOCO_HYPERPARAMS = ("outer_lr", "outer_momentum", "weighting")
_DILOCO_NO_HYPERPARAMS = "no outer hyperparameters for this run (--mesh-run-configs)"


def diloco_inputs(
    run_id: str,
    round: int,
    *,
    snapshot_dir: Path,
    configs_path: Path | None,
    eval_shard: Path | None,
    assignments_path: Path | None,
    model_cfg,
    n_replicates: int,
) -> dict:
    """The `diag_diloco` input keys for one captured round. `build_inputs`
    stays check-agnostic; only the round axis has a merge to diagnose. The
    outer hyperparameters are required: the replayed step is meaningless
    without the ones the run actually used."""
    if (eval_shard is None) != (assignments_path is None):
        raise ValueError(
            "--diloco-eval-shard and --diloco-assignments go together: the held-out "
            "guard needs the assignments to vouch for the shard"
        )
    hyperparams = run_hyperparams(run_id, configs_path) if configs_path else {}
    missing = [k for k in DILOCO_HYPERPARAMS if k not in hyperparams]
    if missing:
        raise ValueError(
            f"{run_id}: diag_diloco needs {missing} from its mesh_run_configs.json "
            f"entry (--mesh-run-configs, got {configs_path})"
        )
    inputs = {
        "diloco_snapshot_dir": Path(snapshot_dir) / run_id,
        "diloco_round": round,
        "diloco_outer_lr": float(hyperparams["outer_lr"]),
        "diloco_outer_momentum": float(hyperparams["outer_momentum"]),
        "diloco_weighting": hyperparams["weighting"],
        "model_cfg": model_cfg,
        "diloco_n_replicates": n_replicates,
    }
    if eval_shard is not None:
        inputs["diloco_eval_shard"] = Path(eval_shard)
        inputs["diloco_assignments"] = Path(assignments_path)
    return inputs


SCHEMA = 1


def _metric_json(check: str, value: MetricValue) -> dict:
    """A metric as written to diagnostics.json. A deterministic value read off
    activations could have been resampled over inputs and wasn't, so it is
    `replicated: false`; only a weights-only one has nothing to resample."""
    return {
        "mean": value.mean,
        "sd": value.sd,
        "n": value.n,
        "seeds": list(value.seeds),
        "deterministic": value.deterministic,
        "replicated": not value.deterministic or check in WEIGHTS_ONLY,
    }


def call_checks(
    bundle: DiagBundle,
    inputs: dict,
    checks: list[str],
    *,
    run_id: str,
    round: int | None = None,
    rung: str | None = None,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Run each requested check once on one bundle.

    Returns `(envelopes, arrays, errors, skipped)`; `arrays[i]` holds the bulk
    arrays of `envelopes[i]`, which never go into JSON. A check that raises, or
    a PAIRED check missing an input kind, is an error for that check alone.
    """
    where = {"run_id": run_id, "round": round, "rung": rung}
    envelopes: list[dict] = []
    arrays: list[dict] = []
    errors: list[dict] = []
    skipped: list[dict] = []
    for key in checks:
        try:
            pairs = normalise(key, CHECKS[key](bundle, inputs), EMITS[key])
        except Exception as exc:  # noqa: BLE001 -- one bad check must not abort the rest
            errors.append(
                {
                    "check": key,
                    **where,
                    "reason": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
            continue
        if key in PAIRED:
            kinds = {report.meta.get("input_kind") for _, report in pairs}
            absent = sorted({"real", "random"} - kinds)
            if absent:
                errors.append(
                    {
                        "check": key,
                        **where,
                        "reason": f"PAIRED check returned no {' / '.join(absent)} "
                        "input_kind report -- the real/random pairing is broken",
                    }
                )
                continue
        for name, report in pairs:
            if "skipped" in report.meta:
                skipped.append(
                    {"check": name, **where, "reason": report.meta["skipped"]}
                )
                continue
            envelopes.append(
                {
                    "check": name,
                    **where,
                    "input_kind": report.meta["input_kind"],
                    "metrics": {
                        m: _metric_json(name, v) for m, v in report.metrics.items()
                    },
                    "meta": dict(report.meta),
                }
            )
            arrays.append(report.arrays)
    return envelopes, arrays, errors, skipped


def _resolve_checks(checks: list[str], n_replicates: int) -> list[str]:
    """Startup validation shared by every axis; returns the checks to run,
    GATES included whether requested or not."""
    if n_replicates < 3:
        raise ValueError(f"n_replicates must be >= 3, got {n_replicates}")
    unknown = [c for c in checks if c not in CHECKS]
    if unknown:
        raise ValueError(f"unknown check(s) {unknown}; valid names: {sorted(CHECKS)}")
    resolved = list(dict.fromkeys(checks))
    return resolved + sorted(g for g in GATES if g not in resolved)


def _read_ladder(ladder_path: Path | None, skip_ladder_check: bool) -> dict:
    """The `ladder_tiers` a run/round invocation is read against. Without a
    ladder a cell is as uninterpretable as a lone checkpoint's number, so its
    absence is refused unless explicitly skipped."""
    if ladder_path is None:
        if skip_ladder_check:
            return {}
        raise ValueError(
            "no ladder on record: pass ladder_path (--ladder-path) from a prior "
            "--ladder invocation, or skip_ladder_check (--skip-ladder-check) for "
            "a deliberate uncalibrated run"
        )
    ladder = json.loads(Path(ladder_path).read_text())
    if ladder["ladder_revision"] != LADDER_REVISION:
        raise ValueError(
            f"{ladder_path} was written under ladder revision "
            f"{ladder['ladder_revision']}, current is {LADDER_REVISION} -- re-run --ladder"
        )
    return ladder["ladder_tiers"]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _diagnostics_payload(
    envelopes: list[dict],
    errors: list[dict],
    skipped: list[dict],
    ladder_tiers: dict,
    hyperparams: dict,
) -> dict:
    return {
        "schema": SCHEMA,
        "ladder_revision": LADDER_REVISION,
        "ladder_tiers": ladder_tiers,
        "hyperparams": hyperparams,
        "reports": envelopes,
        "errors": errors,
        "skipped": skipped,
    }


def _fmt(entry: dict) -> str:
    if entry.get("sd") is None:
        return f"{entry['mean']:.3g}"
    return f"{entry['mean']:.3g} ± {entry['sd']:.3g}"


def _separated(value: dict, rung: dict) -> bool:
    """`MetricValue.separated_from`, with its exact-comparison rule for a pair
    of deterministic values. A band that cannot be formed -- mixed kinds, or
    zero observed variance -- resolves nothing, so it is not a separation."""
    if value["deterministic"] and rung["deterministic"]:
        return value["mean"] != rung["mean"]
    if value["deterministic"] or rung["deterministic"]:
        return False
    try:
        return _metric_value({**value, "seeds": ()}).separated_from(
            _metric_value({**rung, "seeds": ()})
        )
    except ValueError:
        return False


def stamp_verdicts(envelopes: list[dict], ladder_tiers: dict) -> None:
    """Set `bracket`, `verdict` and `red` on every metric, in place, so the
    renderer computes nothing. The verdict tests separation against the
    nearest calibrated rung only -- the bracket the value sits in is the one
    comparison that means something."""
    for e in envelopes:
        for name, m in e["metrics"].items():
            m.update(bracket=None, verdict=None, red=False)
            entry = ladder_tiers.get(metric_key(e["check"], e["input_kind"], name))
            rungs = sorted(
                (entry or {}).get("rungs", {}).items(), key=lambda kv: kv[1]["mean"]
            )
            if len(rungs) < 2:
                continue
            (lo_name, lo), (hi_name, hi) = rungs[0], rungs[-1]
            if not lo["mean"] <= m["mean"] <= hi["mean"]:
                m["bracket"] = (
                    f"{_fmt(m)} · outside ladder range {lo_name}={lo['mean']:.3g} "
                    f"to {hi_name}={hi['mean']:.3g}"
                )
                if m["replicated"]:
                    m["verdict"] = "OUT_OF_RANGE"
                continue
            i = max(i for i, (_, r) in enumerate(rungs) if r["mean"] <= m["mean"])
            (below, b), (above, a) = rungs[
                min(i, len(rungs) - 2) : min(i, len(rungs) - 2) + 2
            ]
            m["bracket"] = (
                f"{_fmt(m)} · between {below}={b['mean']:.3g} and {above}={a['mean']:.3g}"
            )
            if not m["replicated"]:
                continue
            nearest = min(rungs, key=lambda kv: abs(kv[1]["mean"] - m["mean"]))[1]
            if not _separated(m, nearest):
                m["verdict"] = "INCONCLUSIVE"
                continue
            m["verdict"] = "SEPARATED"
            bad = bad_direction(e["check"], name)
            below_nearest = m["mean"] < nearest["mean"]
            m["red"] = entry["tier"] in ("A", "B") and (
                (bad == "low" and below_nearest)
                or (bad == "high" and not below_nearest)
            )


def _plot_label(e: dict) -> str:
    label = e["rung"] or e["run_id"] or "run"
    if e["round"] is not None:
        label += f"_round{e['round']}"
    return label


def write_plots(
    envelopes: list[dict], arrays: list[dict], plots_dir: Path
) -> list[dict]:
    """One PNG per plottable array: 2-D as a heatmap, `(N, 2)` as a scatter,
    1-D as a line, 3-D `(K, H, W)` as its K heatmaps side by side (the mask-
    ratio grid: one panel per ratio). Sets `meta["plot"]` (the file names) and `meta["panel"]`;
    returns a `skipped` entry for each array of any other shape. PNG metadata
    is pinned, so identical arrays give identical bytes."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    skipped: list[dict] = []
    for e, named in zip(envelopes, arrays, strict=True):
        plots = []
        for array_name in sorted(named):
            arr = np.asarray(named[array_name])
            if arr.ndim == 3:
                arr = np.concatenate(list(arr), axis=1)
            if arr.ndim not in (1, 2):
                skipped.append(
                    {
                        "check": e["check"],
                        "run_id": e["run_id"],
                        "round": e["round"],
                        "rung": e["rung"],
                        "reason": f"array {array_name} has unsupported shape {arr.shape}; no plot written",
                    }
                )
                continue
            fig = Figure(figsize=(4, 3.2))
            FigureCanvasAgg(fig)
            ax = fig.subplots()
            if arr.ndim == 1:
                ax.plot(arr, marker="o")
            elif arr.shape[1] == 2:
                ax.scatter(arr[:, 0], arr[:, 1], s=10)
            else:
                fig.colorbar(ax.imshow(arr, cmap="viridis"), ax=ax)
            ax.set_title(f"{e['check']} {array_name} ({_plot_label(e)})", fontsize=8)
            fig.tight_layout()
            filename = re.sub(
                r"[^A-Za-z0-9._-]",
                "_",
                f"{_plot_label(e)}__{e['check']}__{e['input_kind']}__{array_name}.png",
            )
            plots_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(plots_dir / filename, format="png", metadata={"Software": None})
            plots.append(filename)
        if plots:
            e["meta"]["plot"] = plots
            e["meta"]["panel"] = PANEL_OF.get(e["check"], "qualitative")
    return skipped


def _gate_ceiling(model_cfg, public_videomaev2_ckpt: Path | None) -> DiagBundle | None:
    """The gate's ceiling arm: rung 3, the one ladder rung sharing the
    checkpoint's architecture. Absent is not an error -- the gate still
    classifies against the floor and records why it has no position."""
    try:
        return load_ladder_rung(
            Rung.PUBLIC_VIDEOMAEV2,
            model_cfg,
            public_videomaev2_ckpt_path=public_videomaev2_ckpt,
        )
    except RungUnavailable:
        return None


def run(
    *,
    run_ids: list[str],
    checks: list[str],
    mesh_shared: Path,
    shard_path: Path,
    model_cfg,
    out_dir: Path,
    ladder_path: Path | None,
    n_replicates: int = 3,
    skip_ladder_check: bool = False,
    configs_path: Path | None = None,
    seed: int = 0,
    n_windows: int = 4,
    min_epochs: int = diag_finetune_afewva.GATE_MIN_EPOCHS,
    gate_seeds: int = diag_finetune_afewva.GATE_SEEDS,
    gate_cache_dir: Path | None = None,
    public_videomaev2_ckpt: Path | None = None,
) -> None:
    """The run axis: every check on each run's `master.pt`, against a ladder.
    A run whose checkpoint fails to load is an error for that run alone.

    The only axis that supplies the AFEW-VA gate's inputs: fine-tuning per
    ladder rung or per captured round is not what the gate measures."""
    checks = _resolve_checks(checks, n_replicates)
    ladder_tiers = _read_ladder(ladder_path, skip_ladder_check)
    out_dir = Path(out_dir)
    gate_inputs = {
        "model_cfg": model_cfg,
        "min_epochs": min_epochs,
        "n_seeds": gate_seeds,
        "cache_dir": Path(gate_cache_dir) if gate_cache_dir else out_dir / "gate_cache",
        "ceiling_bundle": _gate_ceiling(model_cfg, public_videomaev2_ckpt),
    }

    envelopes: list[dict] = []
    arrays: list[dict] = []
    errors: list[dict] = []
    skipped: list[dict] = []
    for run_id in run_ids:
        try:
            bundle = load_bundle(run_id, mesh_shared, model_cfg)
            inputs = build_inputs(
                bundle,
                shard_path=shard_path,
                model_cfg=model_cfg,
                seed=seed,
                n_windows=n_windows,
                ref_bundle=None,
            )
            inputs["gate"] = gate_inputs
        except Exception as exc:  # noqa: BLE001 -- one bad run must not abort the rest
            errors.append(
                {
                    "check": None,
                    "run_id": run_id,
                    "round": None,
                    "rung": None,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        e, a, err, sk = call_checks(bundle, inputs, checks, run_id=run_id)
        envelopes += e
        arrays += a
        errors += err
        skipped += sk

    stamp_verdicts(envelopes, ladder_tiers)
    skipped += write_plots(envelopes, arrays, out_dir / "plots")
    hyperparams = (
        {rid: run_hyperparams(rid, configs_path) for rid in run_ids}
        if configs_path
        else {}
    )
    _write_json(
        out_dir / "diagnostics.json",
        _diagnostics_payload(envelopes, errors, skipped, ladder_tiers, hyperparams),
    )


def _metric_value(entry: dict) -> MetricValue:
    return MetricValue(
        mean=entry["mean"],
        sd=entry["sd"],
        n=entry["n"],
        seeds=tuple(entry["seeds"]),
        deterministic=entry["deterministic"],
    )


def _ladder_tiers(per_rung: dict[Rung, list[dict]]) -> dict:
    """Each metric's ladder_tau over the rungs that produced it, with the
    per-rung values kept so a later cell can be bracketed between them."""
    values: dict[str, dict[Rung, dict]] = {}
    directions: dict[str, str] = {}
    for rung, envelopes in per_rung.items():
        for e in envelopes:
            for name, entry in e["metrics"].items():
                key = metric_key(e["check"], e["input_kind"], name)
                values.setdefault(key, {})[rung] = entry
                directions[key] = good_direction(e["check"], name)
    tiers = {}
    for key, by_rung in values.items():
        direction = directions[key]
        result = ladder_tau(
            {r: _metric_value(v) for r, v in by_rung.items()},
            direction=direction or "rising",
        )
        tiers[key] = {
            "tau": result.tau,
            "tier": result.tier.value,
            "direction": direction or "undeclared",
            # With no declared direction there is nothing to be inverted against.
            "inverted": result.inverted and direction is not None,
            "present_rungs": result.present_rungs,
            "rungs": {
                r.name.lower(): {k: v[k] for k in ("mean", "sd", "n", "deterministic")}
                for r in LADDER
                if (v := by_rung.get(r)) is not None
            },
        }
    return tiers


def _merge_metric(entries: list[dict], seeds: list[int]) -> dict:
    """One metric's value over independent input draws.

    Identical deterministic values read weights only and stay deterministic.
    Deterministic values that move with the input become a sampled value over
    the draws -- the band they never had. A sampled value already carries its
    own band and keeps the first draw's: its replicate seeds (and, for the
    probe, its data) repeat across draws, so pooling would count copies as
    independent samples and narrow the band.
    """
    if not all(e["deterministic"] for e in entries):
        if any(e["deterministic"] for e in entries):
            raise ValueError("a metric cannot be deterministic in one draw only")
        return dict(entries[0])
    means = np.array([e["mean"] for e in entries], dtype=float)
    if np.all(means == means[0]):
        return dict(entries[0])
    return {
        "mean": float(means.mean()),
        "sd": float(means.std(ddof=1)),
        "n": len(entries),
        "seeds": list(seeds),
        "deterministic": False,
        "replicated": True,
    }


def _redrawn(checks: list[str], first: list[dict]) -> list[str]:
    """The registry keys worth another input draw: those that emitted a
    deterministic metric read off activations. A sampled metric has its own
    band, and a weights-only one would return the same number again."""
    needs = {
        e["check"]
        for e in first
        if e["check"] not in WEIGHTS_ONLY | DRAW_INDEPENDENT
        and any(m["deterministic"] for m in e["metrics"].values())
    }
    return [k for k in checks if EMITS.get(k, frozenset({k})) & needs]


def _merge_draws(
    first: list[dict],
    arrays: list[dict],
    extra: list[list[dict]],
    redrawn: set[str],
    seeds: list[int],
) -> tuple[list[dict], list[dict]]:
    """A rung's envelopes merged over its input draws, with `arrays` (the
    first draw's, the ones plotted) kept aligned.

    Only envelopes whose emitted check is in `redrawn` are merged. One that
    some extra draw lacks -- its check raised there -- is dropped, as is a
    metric some draw lacks (labelfree omits a degenerate one): the rung is
    then absent for that metric, rather than a single-draw value mixed into
    a ladder whose other rungs carry three.
    """

    def key(e: dict) -> tuple:
        return e["check"], e["input_kind"]

    by_key = [{key(e): e for e in draw} for draw in extra]
    merged, kept_arrays = [], []
    for e, arr in zip(first, arrays):
        if e["check"] not in redrawn:
            merged.append(e)
            kept_arrays.append(arr)
            continue
        matches = [e] + [d.get(key(e)) for d in by_key]
        if any(m is None for m in matches):
            continue
        metrics = {
            name: _merge_metric([m["metrics"][name] for m in matches], seeds)
            for name in e["metrics"]
            if all(name in m["metrics"] for m in matches)
        }
        merged.append({**e, "metrics": metrics})
        kept_arrays.append(arr)
    return merged, kept_arrays


def _to_device(bundle: DiagBundle, device: str) -> DiagBundle:
    """The bundle with its modules on `device` (moved in place, as nn.Module
    does). Every check derives its own tensors from the inputs' device."""
    bundle.encoder.to(device)
    if bundle.model is not None:
        bundle.model.to(device)
    return bundle


def run_ladder(
    *,
    checks: list[str],
    mesh_shared: Path,
    shard_path: Path,
    model_cfg,
    out_dir: Path,
    ladder_run_ids: dict[Rung, "str | dict"],
    public_videomaev2_ckpt_path: Path | None = None,
    supervised_va_ckpt_path: Path | None = None,
    supervised_va_model_cfg=None,
    n_replicates: int = 3,
    seed: int = 0,
    n_windows: int = 4,
    n_draws: int = 3,
    ladder_init_seeds: dict[Rung, int] | None = None,
    device: str = "cpu",
    probe_layers: list[int] | None = None,
    max_new_rungs: int | None = None,
) -> None:
    """The ladder axis: score every buildable rung on identical inputs and
    write `ladder.json` -- and only that, so no later run/round invocation in
    the same out_dir can clobber the calibration it is read against. A rung
    that cannot be built is recorded as unavailable with its reason and left
    out of tau, never silently dropped."""
    checks = _resolve_checks(checks, n_replicates)
    if n_draws < 1 or n_draws == 2:
        # Two draws would make a sampled value with n=2, which MetricValue
        # rejects: one draw is the unbanded reading, three the minimum band.
        raise ValueError(f"n_draws must be 1 or >= 3, got {n_draws}")
    out_dir = Path(out_dir)
    # Each finished rung is cached, so a rerun after a walltime kill resumes at
    # the first unfinished one. The settings key keeps a rerun with other
    # settings from reusing a stale rung.
    rung_dir = out_dir / "rungs"
    rung_dir.mkdir(parents=True, exist_ok=True)
    settings = repr(
        (checks, seed, n_windows, n_draws, n_replicates, device, probe_layers)
    )
    # A walltime-bound job computes at most this many rungs and exits without
    # ladder.json, rather than dying inside a rung it cannot finish.
    new_rungs = 0

    per_rung: dict[Rung, list[dict]] = {}
    rung_status: list[dict] = []
    envelopes: list[dict] = []
    arrays: list[dict] = []
    errors: list[dict] = []
    skipped: list[dict] = []
    for rung in LADDER:
        name = rung.name.lower()
        if rung in (Rung.ROUND, Rung.OVERFIT) and rung not in ladder_run_ids:
            rung_status.append(
                {
                    "rung": name,
                    "status": "unavailable",
                    "reason": f"no ladder_run_ids entry for {name}",
                }
            )
            continue
        if rung is Rung.SUPERVISED_VA and supervised_va_ckpt_path is None:
            rung_status.append(
                {
                    "rung": name,
                    "status": "unavailable",
                    "reason": "no supervised_va checkpoint path supplied",
                }
            )
            continue
        cache = rung_dir / f"{name}.pkl"
        key = repr(
            (settings, ladder_run_ids.get(rung), (ladder_init_seeds or {}).get(rung))
        )
        if cache.exists():
            done = pickle.loads(cache.read_bytes())
            if done["settings"] == key:
                rung_status.append({"rung": name, "status": "built"})
                per_rung[rung] = done["envelopes"]
                envelopes += done["envelopes"]
                arrays += done["arrays"]
                errors += done["errors"]
                skipped += done["skipped"]
                print(f"rung {name}: cached", flush=True)
                continue
        n_errors, n_skipped = len(errors), len(skipped)
        try:
            # full_model: a rung whose checkpoint carries the MAE decoder
            # (random init, round, overfit) gets diag_qualitative's
            # mask-ratio grid; public and supervised rungs stay encoder-only.
            bundle = load_ladder_rung(
                rung,
                model_cfg,
                full_model=True,
                mesh_shared=mesh_shared,
                ladder_run_ids=ladder_run_ids,
                public_videomaev2_ckpt_path=public_videomaev2_ckpt_path,
                supervised_va_ckpt_path=supervised_va_ckpt_path,
                supervised_va_model_cfg=supervised_va_model_cfg,
                seed=seed,
            )
        except (RungUnavailable, NotImplementedError, FileNotFoundError) as exc:
            rung_status.append(
                {"rung": name, "status": "unavailable", "reason": str(exc)}
            )
            continue
        if max_new_rungs is not None and new_rungs >= max_new_rungs:
            print(f"rung {name}: left for the next run", flush=True)
            return
        rung_status.append({"rung": name, "status": "built"})
        init_seed = (ladder_init_seeds or {}).get(rung)
        if init_seed is not None:
            bundle = dataclasses.replace(bundle, init_seed=init_seed)
        # Every rung is read against a random init: its own when the seed is
        # known (delta_from_init then measures how far training moved it),
        # the floor's otherwise (residual_branch_norm only needs an untrained
        # reference; delta_from_init refuses one that is not the rung's own).
        ref = random_init_bundle(
            model_cfg, seed=bundle.init_seed if bundle.init_seed is not None else seed
        )
        bundle, ref = _to_device(bundle, device), _to_device(ref, device)
        # n_draws independent input draws (windows, controls, check seeds):
        # a value read off one fixed batch has no band, so any difference
        # between rungs would count as separated.
        draw_seeds = [seed + d for d in range(n_draws)]
        first: list[dict] = []
        extra: list[list[dict]] = []
        redrawn: list[str] = checks
        for d, draw_seed in enumerate(draw_seeds):
            inputs = build_inputs(
                bundle,
                shard_path=shard_path,
                model_cfg=model_cfg,
                seed=draw_seed,
                n_windows=n_windows,
                ref_bundle=ref,
            )
            for kind in ("real", "random"):
                inputs[kind] = inputs[kind].to(device)
            inputs["control_bundle"] = _to_device(inputs["control_bundle"], device)
            if probe_layers is not None:
                inputs["layers"] = list(probe_layers)
            e, a, err, sk = call_checks(
                bundle, inputs, redrawn, run_id=bundle.run_id, rung=name
            )
            errors += [{**x, "draw": d} for x in err]
            if d == 0:
                first, first_arrays = e, a
                skipped += sk
                redrawn = _redrawn(checks, first)
                if not redrawn:
                    break
            else:
                extra.append(e)
        e, a = _merge_draws(
            first,
            first_arrays,
            extra,
            {c for k in redrawn for c in EMITS.get(k, frozenset({k}))},
            draw_seeds,
        )
        arrays += a
        per_rung[rung] = e
        envelopes += e
        cache.write_bytes(
            pickle.dumps(
                {
                    "settings": key,
                    "envelopes": e,
                    "arrays": a,
                    "errors": errors[n_errors:],
                    "skipped": skipped[n_skipped:],
                }
            )
        )
        new_rungs += 1
        print(f"rung {name}: done", flush=True)

    ladder_tiers = _ladder_tiers(per_rung)
    stamp_verdicts(envelopes, ladder_tiers)
    skipped += write_plots(envelopes, arrays, out_dir / "plots")

    _write_json(
        out_dir / "ladder.json",
        {
            "schema": SCHEMA,
            "ladder_revision": LADDER_REVISION,
            "ladder_tiers": ladder_tiers,
            "rung_status": rung_status,
            "reports": envelopes,
            "errors": errors,
            "skipped": skipped,
        },
    )
    # ladder.json is the only output; the rung cache only bridges reruns.
    for cache in rung_dir.glob("*.pkl"):
        cache.unlink()
    rung_dir.rmdir()


def parse_ladder_run_ids(text: str) -> dict[Rung, "str | dict"]:
    """`round=r7,overfit=ssl-diag-overfit50` -> `{Rung.ROUND: "r7", ...}`.

    A run id matching a recorded solo route in `LADDER_RUN_IDS` resolves to
    that route: a merge=false run writes no master.pt, so a bare id would
    name a file that does not exist."""
    parsed: dict[Rung, "str | dict"] = {}
    for pair in filter(None, (p.strip() for p in text.split(","))):
        name, _, run_id = pair.partition("=")
        try:
            rung = Rung[name.strip().upper()]
        except KeyError:
            raise ValueError(
                f"unknown rung {name.strip()!r} in --ladder-run-ids; valid: "
                f"{[r.name.lower() for r in LADDER]}"
            ) from None
        route = LADDER_RUN_IDS.get(rung)
        if route is not None and route["run_id"] == run_id.strip():
            parsed[rung] = dict(route)
        else:
            parsed[rung] = run_id.strip()
    return parsed


def _validate_rounds_against_manifest(
    manifest_lines: list[str], requested_rounds: list[int]
) -> list[str]:
    """One problem string per requested round the capture manifest does not
    fully cover: absent, or recorded with a non-empty `gaps` list. A thin
    reader -- whether the manifest itself is right is the capture's concern."""
    records = {
        rec["round"]: rec
        for rec in (json.loads(ln) for ln in manifest_lines if ln.strip())
    }
    problems = []
    for r in requested_rounds:
        if r not in records:
            problems.append(f"round {r}: not captured")
        elif records[r].get("gaps"):
            problems.append(f"round {r}: {'; '.join(records[r]['gaps'])}")
    return problems


def run_rounds(
    *,
    run_id: str,
    rounds: list[int],
    snapshot_dir: Path | None,
    checks: list[str],
    shard_path: Path,
    model_cfg,
    out_dir: Path,
    ladder_path: Path | None,
    n_replicates: int = 3,
    skip_ladder_check: bool = False,
    configs_path: Path | None = None,
    seed: int = 0,
    n_windows: int = 4,
    diloco_eval_shard: Path | None = None,
    diloco_assignments: Path | None = None,
    diloco_optional: bool = False,
) -> None:
    """The round axis: one run's captured round masters, each with the round
    n-1 snapshot as its reference. Every requested round is validated against
    the capture manifest before anything loads."""
    checks = _resolve_checks(checks, n_replicates)
    if snapshot_dir is None:
        raise ValueError("the round axis requires snapshot_dir (--snapshot-dir)")
    manifest_path = Path(snapshot_dir) / run_id / "manifest.jsonl"
    if not manifest_path.exists():
        raise ValueError(f"{run_id}: no capture manifest at {manifest_path}")
    manifest_lines = manifest_path.read_text().splitlines()
    problems = _validate_rounds_against_manifest(manifest_lines, rounds)
    if problems:
        raise ValueError(f"{run_id}: requested rounds unavailable -- {problems}")
    captured = {json.loads(ln)["round"] for ln in manifest_lines if ln.strip()}
    ladder_tiers = _read_ladder(ladder_path, skip_ladder_check)
    # Built up front so a missing outer hyperparameter fails before any load.
    # `diloco_optional` (diag_diloco only there because --checks defaulted)
    # turns that failure into a skip, so depth-only command lines keep working.
    diloco: dict[int, dict] = {}
    if "diag_diloco" in checks:
        hyperparams = run_hyperparams(run_id, configs_path) if configs_path else {}
        if diloco_optional and not set(DILOCO_HYPERPARAMS) <= set(hyperparams):
            diloco = {r: {"diloco_skip_reason": _DILOCO_NO_HYPERPARAMS} for r in rounds}
        else:
            diloco = {
                r: diloco_inputs(
                    run_id,
                    r,
                    snapshot_dir=snapshot_dir,
                    configs_path=configs_path,
                    eval_shard=diloco_eval_shard,
                    assignments_path=diloco_assignments,
                    model_cfg=model_cfg,
                    n_replicates=n_replicates,
                )
                for r in rounds
            }

    envelopes: list[dict] = []
    arrays: list[dict] = []
    errors: list[dict] = []
    skipped: list[dict] = []
    for r in rounds:
        try:
            bundle = load_snapshot_bundle(run_id, snapshot_dir, model_cfg, round=r)
            ref = (
                load_snapshot_bundle(run_id, snapshot_dir, model_cfg, round=r - 1)
                if r - 1 in captured
                else None
            )
            inputs = build_inputs(
                bundle,
                shard_path=shard_path,
                model_cfg=model_cfg,
                seed=seed,
                n_windows=n_windows,
                ref_bundle=ref,
            ) | diloco.get(r, {})
        except Exception as exc:  # noqa: BLE001 -- one bad round must not abort the rest
            errors.append(
                {
                    "check": None,
                    "run_id": run_id,
                    "round": r,
                    "rung": None,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        e, a, err, sk = call_checks(bundle, inputs, checks, run_id=run_id, round=r)
        if ref is None:
            for entry in sk:
                if entry["reason"] == "no ref_bundle supplied":
                    entry["reason"] += (
                        f": round {r - 1} reference snapshot not captured"
                    )
        envelopes += e
        arrays += a
        errors += err
        skipped += sk

    stamp_verdicts(envelopes, ladder_tiers)
    skipped += write_plots(envelopes, arrays, Path(out_dir) / "plots")
    hyperparams = (
        {run_id: run_hyperparams(run_id, configs_path)} if configs_path else {}
    )
    _write_json(
        Path(out_dir) / "diagnostics.json",
        _diagnostics_payload(envelopes, errors, skipped, ladder_tiers, hyperparams),
    )


def _csv(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ladder", action="store_true", help="the ladder axis")
    parser.add_argument(
        "--ladder-run-ids", default="", help="e.g. round=r7,overfit=ssl-diag-overfit50"
    )
    parser.add_argument("--run-ids", default="")
    parser.add_argument(
        "--rounds", default="", help="comma-separated; needs --snapshot-dir"
    )
    parser.add_argument("--snapshot-dir", type=Path, default=None)
    parser.add_argument("--mesh-shared", type=Path, default=None)
    parser.add_argument("--shard-path", type=Path, required=True)
    parser.add_argument(
        "--checks", default=None, help="comma-separated; default: every check"
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--ladder-path", type=Path, default=None)
    parser.add_argument(
        "--skip-ladder-check",
        action="store_true",
        help="explicit opt-out of the ladder-on-record requirement",
    )
    parser.add_argument("--n-replicates", type=int, default=3)
    parser.add_argument("--n-windows", type=int, default=4)
    parser.add_argument(
        "--device",
        default="cpu",
        help="ladder axis: where rungs and inputs live (e.g. cuda); the CPU "
        "ladder takes ~30 h at three draws",
    )
    parser.add_argument(
        "--ladder-init-seeds",
        default="",
        help="ladder axis: the train.seed each trained rung was initialised "
        "from, e.g. round=0,overfit=0 -- lets delta_from_init read them",
    )
    parser.add_argument(
        "--ladder-draws",
        type=int,
        default=3,
        help="ladder axis: independent input draws merged into each rung's values",
    )
    parser.add_argument(
        "--probe-layers",
        default="",
        help="ladder axis: probe_by_depth's layers, e.g. 12 (default: its sweep)",
    )
    parser.add_argument(
        "--ladder-max-new-rungs",
        type=int,
        default=None,
        help="ladder axis: compute at most N uncached rungs, then exit; rerun "
        "to resume (one rung per walltime-bound job)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mesh-run-configs", type=Path, default=None)
    parser.add_argument(
        "--diloco-eval-shard",
        type=Path,
        default=None,
        help="round axis: a mesh val/ shard for merge_regression/counterfactual_sweep",
    )
    parser.add_argument(
        "--diloco-assignments",
        type=Path,
        default=None,
        help="round axis: the run's shard_assignments_<run_id>.json",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("video_encoding/conf/model/videomae_base.yaml"),
    )
    parser.add_argument("--public-videomaev2-ckpt", type=Path, default=None)
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=diag_finetune_afewva.GATE_MIN_EPOCHS,
        help="run axis: AFEW-VA gate fine-tune length, identical for every arm",
    )
    parser.add_argument(
        "--gate-seeds", type=int, default=diag_finetune_afewva.GATE_SEEDS
    )
    parser.add_argument(
        "--gate-cache-dir",
        type=Path,
        default=None,
        help="run axis: shared gate arm cache; default <out-dir>/gate_cache",
    )
    parser.add_argument("--supervised-va-ckpt", type=Path, default=None)
    parser.add_argument("--supervised-va-model-config", type=Path, default=None)
    args = parser.parse_args(argv)

    run_ids = _csv(args.run_ids)
    if args.rounds and len(run_ids) != 1:
        raise ValueError(
            f"--rounds takes exactly one --run-ids entry (got {run_ids}); the round axis is per-run"
        )

    from omegaconf import OmegaConf

    model_cfg = OmegaConf.load(args.model_config)
    common = dict(
        checks=_csv(args.checks) if args.checks is not None else list(CHECKS),
        shard_path=args.shard_path,
        model_cfg=model_cfg,
        out_dir=args.out_dir,
        n_replicates=args.n_replicates,
        seed=args.seed,
        n_windows=args.n_windows,
    )
    if args.ladder:
        run_ladder(
            mesh_shared=args.mesh_shared,
            ladder_run_ids=parse_ladder_run_ids(args.ladder_run_ids),
            public_videomaev2_ckpt_path=args.public_videomaev2_ckpt,
            supervised_va_ckpt_path=args.supervised_va_ckpt,
            n_draws=args.ladder_draws,
            device=args.device,
            probe_layers=[int(x) for x in _csv(args.probe_layers)] or None,
            max_new_rungs=args.ladder_max_new_rungs,
            ladder_init_seeds={
                Rung[k.strip().upper()]: int(v)
                for k, _, v in (p.partition("=") for p in _csv(args.ladder_init_seeds))
            },
            supervised_va_model_cfg=(
                OmegaConf.load(args.supervised_va_model_config)
                if args.supervised_va_model_config
                else None
            ),
            **common,
        )
    elif args.rounds:
        run_rounds(
            run_id=run_ids[0],
            rounds=[int(r) for r in _csv(args.rounds)],
            snapshot_dir=args.snapshot_dir,
            ladder_path=args.ladder_path,
            skip_ladder_check=args.skip_ladder_check,
            configs_path=args.mesh_run_configs,
            diloco_eval_shard=args.diloco_eval_shard,
            diloco_assignments=args.diloco_assignments,
            diloco_optional=args.checks is None,
            **common,
        )
    else:
        run(
            run_ids=run_ids,
            mesh_shared=args.mesh_shared,
            ladder_path=args.ladder_path,
            skip_ladder_check=args.skip_ladder_check,
            configs_path=args.mesh_run_configs,
            min_epochs=args.min_epochs,
            gate_seeds=args.gate_seeds,
            gate_cache_dir=args.gate_cache_dir,
            public_videomaev2_ckpt=args.public_videomaev2_ckpt,
            **common,
        )


if __name__ == "__main__":
    main()
