"""AFEW-VA fine-tune acceptance gate.

The framework's one non-intrinsic check: no SSL checkpoint is accepted on
intrinsic diagnostics alone (design doc, "Acceptance gate"). The checkpoint,
a `random_init` floor and a ladder ceiling are each fine-tuned end to end on
AFEW-VA for the same fixed epoch count and the same seeds, through the
training path that ships (`training/train.py::run_train_stage`). This module
adds only the gate logic on top.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import tempfile
from pathlib import Path
from typing import Callable

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from video_encoding.scripts._diag_common import (
    LADDER_REVISION,
    DiagBundle,
    DiagReport,
    MetricValue,
)
from video_encoding.scripts.diag_probe import _resolve_device
from video_encoding.training import train as train_mod

CONF_DIR = Path(__file__).resolve().parents[1] / "conf"

# Two-sided 95% Student-t critical values by degrees of freedom (n_seeds - 1).
# Tabulated because scipy is not a dependency; the table is what bounds
# n_seeds to [3, 10].
_T975 = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}

# A non-PASS whose MDE exceeds this CCC difference says more about the
# experiment than the checkpoint: the seeds could not have resolved a gap this
# size. A judgment call, named so step 14's re-calibration can revise it.
MDE_OF_INTEREST = 0.05

# Held-out share of identities, pinned so every arm and every cached result
# scores on the same split (diag_probe's PROBE_SPLIT_FRACTION, same value).
SPLIT_FRACTION = 0.2
# The gate's fine-tune recipe departs from Stage 2's defaults in two ways, both
# from ssl-diag-08 job 362: at lr 1e-4 the regressor's tanh pre-activation hit
# ~7 within 20 steps, pinning every output at +-1 (CCC exactly 0 on every arm).
# An unbounded readout at the probe's lr (ssl-diag-03) keeps the gradient alive.
GATE_LR = 1e-5
GATE_HEAD_BOUNDED = False
# The live gate (ssl-diag-08 jobs 446, 485-489) scored the 10th epoch, but
# job 446's held-out CCC peaks at epoch 1 and then only overfits, so the last
# epoch read every weak arm at its overfit worst. One epoch scores the peak
# without selecting on the scored split. Ten seeds (the most `_T975` tabulates)
# take the paired interval's MDE from 0.25 at n=3 to roughly a quarter of it;
# a 1-epoch fine-tune costs about a tenth of a 10-epoch one, so the run is
# cheaper than the old gate despite the extra seeds.
GATE_MIN_EPOCHS = 1
GATE_SEEDS = 10

# Arm label -> cache directory name. The checkpoint's key adds its run id.
_CONTROL_KEY = "random_init"
_CEILING_KEY = "ceiling_public_videomaev2"


def finetune_ccc(
    encoder, model_cfg, *, manifest_root: Path, min_epochs: int, seed: int
) -> dict:
    """One end-to-end fine-tune of `encoder` on AFEW-VA, run for exactly
    `min_epochs` with resume off, scored on the final epoch over the
    identity-disjoint held-out split.

    `epochs` is the count `run_train_stage`'s loop ran -- never `min_epochs`
    echoed back, so a shortened run is visible to the gate.
    """
    manifest_root = Path(manifest_root)
    with tempfile.TemporaryDirectory() as work:
        work = Path(work)
        encoder_path = work / "encoder_init.pt"
        torch.save(
            {
                "model_state_dict": {
                    f"encoder.{k}": v for k, v in encoder.state_dict().items()
                }
            },
            encoder_path,
        )
        with initialize_config_dir(config_dir=str(CONF_DIR), version_base=None):
            cfg = compose(
                config_name="config",
                overrides=[
                    "experiment=videomae_va_finetune_afewva",
                    f"train.epochs={min_epochs}",
                    f"train.seed={seed}",
                    "++train.split=identity",
                    f"++train.split_fraction={SPLIT_FRACTION}",
                    "train.freeze_backbone=false",
                    "train.resume=false",
                    "train.save_latest=false",
                    f"train.learning_rate={GATE_LR}",
                    f"train.pretrain_encoder_path={encoder_path}",
                    f"train.checkpoint_root={work / 'ckpt'}",
                    f"train.results_root={work / 'results'}",
                    "dataset.present=true",
                    f"dataset.path={manifest_root}",
                    f"dataset.manifest_path={manifest_root / 'manifest.csv'}",
                ],
            )
        with open_dict(cfg):
            cfg.model = OmegaConf.create(OmegaConf.to_container(model_cfg))
            cfg.model.head_bounded = GATE_HEAD_BOUNDED
        result = train_mod.run_train_stage(
            cfg, rank=0, world_size=1, local_rank=0, device=_resolve_device()
        )

    final = result["final"]
    return {
        "ccc_valence": float(final["val_ccc_v"]),
        "ccc_arousal": float(final["val_ccc_a"]),
        "mean_ccc": float(final["mean_ccc"]),
        "epochs": int(result["epochs_run"]),
    }


def classify(
    ckpt: list[float], control: list[float]
) -> tuple[str, tuple[float, float], float]:
    """`(status, ci95, mde)` from a paired t-interval on `ckpt - control`.

    A t-interval rather than a percentile bootstrap: at n=3 there are only 27
    resamples, so a bootstrap degenerates toward min/max and under-covers --
    it would PASS on noise. `mde` is the interval's half-width, the smallest
    mean difference this seed variance could have resolved.
    """
    diffs = [a - b for a, b in zip(ckpt, control, strict=True)]
    n = len(diffs)
    mean = statistics.fmean(diffs)
    mde = _T975[n - 1] * statistics.stdev(diffs) / math.sqrt(n)
    ci95 = (mean - mde, mean + mde)
    if ci95[0] > 0:
        return "PASS", ci95, mde
    if ci95[1] < 0:
        return "FAIL", ci95, mde
    return "INCONCLUSIVE", ci95, mde


def encoder_fingerprint(encoder) -> str:
    """sha256 over the encoder's weights, so a cached arm result is reused only
    for the exact weights it was computed on -- a live run's master.pt moves."""
    digest = hashlib.sha256()
    for key, tensor in sorted(encoder.state_dict().items()):
        digest.update(key.encode())
        flat = tensor.detach().cpu().contiguous().reshape(-1)
        digest.update(flat.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _check_gate_params(cache_dir: Path, params: dict) -> None:
    """Refuse a cache written under other parameters: its control and ceiling
    numbers are not comparable with this invocation's checkpoint."""
    path = cache_dir / "gate_params.json"
    if path.exists():
        recorded = json.loads(path.read_text())
        for key, value in params.items():
            if recorded.get(key) != value:
                raise ValueError(
                    f"gate: {key}={value} differs from {recorded.get(key)} recorded "
                    f"in {path} -- a gate cache is only extended with comparable runs"
                )
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    _write_atomic(path, params)


def _write_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def _run_arm(
    key: str, encoder, *, cache_dir: Path, n_seeds: int, finetune, **kwargs
) -> list[dict]:
    """One arm's per-seed results, from the cache where the weights match and
    from `finetune` otherwise. Each result is written as soon as it returns,
    so a walltime kill loses at most the fine-tune in flight."""
    fingerprint = encoder_fingerprint(encoder)
    arm_dir = cache_dir / key
    arm_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in range(n_seeds):
        path = arm_dir / f"seed{seed}.json"
        cached = json.loads(path.read_text()) if path.exists() else None
        if cached is None or cached["fingerprint"] != fingerprint:
            result = finetune(encoder, seed=seed, **kwargs)
            if result["epochs"] != kwargs["min_epochs"]:
                raise ValueError(
                    f"gate: arm {key} seed {seed} ran {result['epochs']} epochs, "
                    f"not min_epochs={kwargs['min_epochs']} -- arms must be comparable"
                )
            cached = {**result, "fingerprint": fingerprint}
            _write_atomic(path, cached)
        results.append(cached)
    return results


def _sampled(values: list[float]) -> MetricValue:
    return MetricValue(
        mean=statistics.fmean(values),
        sd=statistics.stdev(values),
        n=len(values),
        seeds=tuple(range(len(values))),
        deterministic=False,
    )


def gate(
    bundle: DiagBundle,
    control_bundle: DiagBundle,
    ceiling_bundle: DiagBundle | None,
    *,
    model_cfg,
    manifest_root: Path,
    min_epochs: int,
    n_seeds: int,
    cache_dir: Path,
    finetune: Callable = finetune_ccc,
) -> DiagReport:
    """Fine-tune the checkpoint, `random_init` and (when given) the ceiling
    under one recipe and the same seeds; classify checkpoint vs `random_init`
    and place the checkpoint between the two ends.

    Seeds pair across arms (same head init and batch order; the split does not
    depend on the seed), which is what makes `checkpoint - control` a paired
    difference. The ceiling never affects `status`.
    """
    if n_seeds - 1 not in _T975:
        raise ValueError(
            f"gate: n_seeds must be in [3, 10], got {n_seeds} -- fewer cannot "
            "carry an interval, more exceed the tabulated t critical values"
        )
    cache_dir = Path(cache_dir)
    _check_gate_params(
        cache_dir,
        {
            "min_epochs": min_epochs,
            "gate_seeds": n_seeds,
            "split_fraction": SPLIT_FRACTION,
            "lr": GATE_LR,
            "head_bounded": GATE_HEAD_BOUNDED,
        },
    )
    arms = [
        ("checkpoint", f"ckpt_{bundle.run_id}", bundle),
        ("control", _CONTROL_KEY, control_bundle),
    ]
    if ceiling_bundle is not None:
        arms.append(("ceiling", _CEILING_KEY, ceiling_bundle))
    runs = {
        arm: _run_arm(
            key,
            arm_bundle.encoder,
            cache_dir=cache_dir,
            n_seeds=n_seeds,
            finetune=finetune,
            model_cfg=model_cfg,
            manifest_root=Path(manifest_root),
            min_epochs=min_epochs,
        )
        for arm, key, arm_bundle in arms
    }
    ccc = {arm: [r["mean_ccc"] for r in results] for arm, results in runs.items()}
    status, ci95, mde = classify(ccc["checkpoint"], ccc["control"])

    metrics = {f"{arm}_ccc": _sampled(values) for arm, values in ccc.items()}
    floor, ckpt = metrics["control_ccc"].mean, metrics["checkpoint_ccc"].mean
    metrics["diff_vs_control"] = MetricValue.deterministic_value(ckpt - floor)
    meta = {
        "LADDER_REVISION": LADDER_REVISION,
        "input_kind": "real",
        "status": status,
        "ci95": list(ci95),
        "mde": mde,
        "underpowered": status != "PASS" and mde > MDE_OF_INTEREST,
        "min_epochs": min_epochs,
        "n_seeds": n_seeds,
        "epochs_run": {
            arm: [r["epochs"] for r in results] for arm, results in runs.items()
        },
        "position": None,
    }
    if "ceiling_ccc" not in metrics:
        meta["position_reason"] = "no ceiling rung available"
    elif metrics["ceiling_ccc"].mean <= floor:
        meta["position_reason"] = (
            "ceiling did not fine-tune above the random_init floor"
        )
    else:
        meta["position"] = (ckpt - floor) / (metrics["ceiling_ccc"].mean - floor)
    return DiagReport(metrics=metrics, arrays={}, meta=meta)


def _gate_not_met(reason: str) -> dict[str, DiagReport]:
    """A report, not a skip: the orchestrator files skips outside `reports`,
    and the banner must be able to say why the gate was not met."""
    return {
        "finetune_afewva": DiagReport(
            metrics={},
            arrays={},
            meta={
                "LADDER_REVISION": LADDER_REVISION,
                "input_kind": "real",
                "status": "GATE_NOT_MET",
                "reason": reason,
            },
        )
    }


def check(bundle: DiagBundle, inputs: dict) -> dict[str, DiagReport]:
    """Orchestrator entry point. `inputs["gate"]` is supplied by the run axis
    only; every other axis reports GATE_NOT_MET rather than fine-tuning per
    rung or per round."""
    afewva_root = os.environ.get("AFEWVA_ROOT")
    if not afewva_root:
        return _gate_not_met("AFEWVA_ROOT unset")
    manifest_root = Path(afewva_root)
    if not (manifest_root / "manifest.csv").exists():
        return _gate_not_met(f"no manifest.csv at {manifest_root}")
    gate_inputs = inputs.get("gate")
    if gate_inputs is None:
        return _gate_not_met("the gate runs on the run axis only")
    if str(gate_inputs["model_cfg"].get("encoder", "")) == "levjepa_hf":
        return _gate_not_met("encoder levjepa_hf is not supported by the gate yet")
    return {
        "finetune_afewva": gate(
            bundle,
            inputs["control_bundle"],
            gate_inputs["ceiling_bundle"],
            model_cfg=gate_inputs["model_cfg"],
            manifest_root=manifest_root,
            min_epochs=gate_inputs["min_epochs"],
            n_seeds=gate_inputs["n_seeds"],
            cache_dir=gate_inputs["cache_dir"],
        )
    }
