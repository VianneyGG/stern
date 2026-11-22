"""AFEW-VA frozen-probe bake-off: one arm's probe CCC at each depth, against random init.

`stern.evaluation.diagnostics.probe.probe_by_depth` does the measurement. This
is the cluster entry point: one arm per invocation, one `DiagReport` JSON per
arm, so arms can run as separate jobs and be merged afterwards.

Probe CCC is used rather than `val_loss` or motion R^2. `val_loss` cannot see
the depth profile (arms with very different profiles land within ~0.005 of each
other), and motion R^2 at the final block ties random init. Probe CCC against a
paired random-init control is the reading that shows whether a checkpoint is
useful downstream, and at which depth to read it.

The model config comes from the run's recorded `experiment` yaml plus its
recorded `model.*` overrides, never from the run_id string. A run_id does not
reliably encode geometry. A mismatched config either fails on a shape mismatch
or, for a parameter-free knob, loads silently and reports the wrong architecture.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from stern.evaluation.diagnostics.common import (
    LADDER_REVISION,
    DiagBundle,
    DiagReport,
    MetricValue,
    load_bundle,
    random_init_bundle,
    replicate,
    run_hyperparams,
)
from stern.evaluation.diagnostics.probe import (
    _CHECKPOINT_SEED_BASE,
    _CONTROL_SEED_BASE,
    DEFAULT_PROBE_DEPTH_LAYERS,
    permute_labels,
    probe_by_depth,
    probe_ccc,
)

__all__ = ["main", "model_cfg_for_run", "permute_labels", "probe_by_depth"]

# The control arm's name. `random_init_bundle` leaves `run_id` empty, which is
# fine inside another arm's report. This CLI writes one report per arm, and an
# unnamed arm cannot be merged with the others.
RANDOM_INIT_RUN_ID = "random_init"

# Pinned so that jobs splitting one arm's layers across nodes (`--arm control`)
# all score against the same random encoder.
RANDOM_INIT_SEED = 0


def _coerce(raw: str):
    """Parse an override's right-hand side the way Hydra would.

    `model.depth=3` must arrive as an int. An `int()` cast in the encoder would
    hide a string, but comparisons like `len(encoder.blocks)` would not.
    """
    return OmegaConf.create(f"value: {raw}").value


def _overlay(cfg: DictConfig, updates: dict, source: str) -> None:
    """Apply `updates` to `cfg`, rejecting any key the base config lacks.

    Fails closed on purpose. A `model.*` key missing from the base config means
    the checkpoint was trained under a config this checkout no longer has. If the
    knob adds no parameters, loading would succeed and the report would describe
    an architecture that was never trained.
    """
    for key, value in updates.items():
        if key not in cfg:
            raise ValueError(
                f"{source}: model.{key} is not a key of the base model config -- "
                "the checkpoint was trained under a config this checkout does "
                "not have, and a parameter-free knob would load silently wrong"
            )
        cfg[key] = value


def model_cfg_for_run(
    base_cfg: DictConfig, entry: dict, experiment_dir: "Path | str"
) -> DictConfig:
    """`base_cfg` plus the geometry the run in `entry` was trained under.

    `entry` is a `runs/state/mesh_run_configs.json` record. Its `experiment`
    names the yaml whose `model:` block the launch composed, and its
    `extra_overrides` hold launch-time overrides, which win over the experiment
    block as in Hydra.
    """
    experiment = entry.get("experiment")
    if not experiment:
        raise ValueError(
            "run config entry names no experiment, so its model geometry "
            "cannot be recovered; refusing to guess it from the run_id"
        )
    experiment_path = Path(experiment_dir) / f"{experiment}.yaml"
    if not experiment_path.exists():
        raise FileNotFoundError(
            f"{experiment}: experiment yaml absent at {experiment_path} -- "
            "the run's geometry is unrecoverable without it"
        )

    cfg = base_cfg.copy()
    experiment_cfg = OmegaConf.load(experiment_path)
    model_block = experiment_cfg.get("model")
    if model_block:
        _overlay(cfg, dict(model_block), f"{experiment}.yaml model block")

    for override in entry.get("extra_overrides") or []:
        key, sep, raw = str(override).partition("=")
        if not sep or not key.startswith("model."):
            continue
        _overlay(
            cfg,
            {key.removeprefix("model."): _coerce(raw)},
            f"{entry.get('run_id', '?')} extra_overrides",
        )
    return cfg


def _arm_bundles(
    run_id: str,
    mesh_shared: "Path | str",
    model_cfg: DictConfig,
    *,
    round: int | None,
    load: bool = True,
) -> tuple["DiagBundle | None", DiagBundle]:
    """The arm under test and its control.

    For `random_init` the two are the same object. `probe_by_depth` then reports
    only probe-seed spread, which makes that run the noise floor. `load=False`
    skips the checkpoint (the `--arm control` job never reads it) and returns
    None in its place.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(RANDOM_INIT_SEED)
        control = random_init_bundle(model_cfg)
    if run_id == RANDOM_INIT_RUN_ID:
        bundle = dataclasses.replace(control, run_id=RANDOM_INIT_RUN_ID)
        return bundle, bundle
    if not load:
        return None, control
    return load_bundle(run_id, mesh_shared, model_cfg, round=round), control


def _single_arm(
    bundle: DiagBundle,
    manifest_path: Path,
    layers: list[int],
    *,
    seed_base: int,
    n_seeds: int,
) -> dict[int, MetricValue]:
    """One arm's probe CCC at each layer.

    Matches what `probe_by_depth` computes for that arm (same split,
    standardisation and seed base), minus the other arm and the forced final
    block. This lets a slow probe (~16 min per fit) be split into jobs that each
    fit a 2h walltime."""
    return {
        layer: replicate(
            lambda s, layer=layer: probe_ccc(
                bundle, manifest_path, layer=layer, seed=s
            ),
            n=n_seeds,
            seed=seed_base,
        )
        for layer in layers
    }


def _report(
    metrics: dict,
    *,
    arm: str,
    run_id: str,
    round: int | None,
    layers: list[int],
    n_seeds: int,
    manifest_path: Path,
    permute_labels_seed: int | None,
    model_cfg: DictConfig,
) -> DiagReport:
    return DiagReport(
        metrics=metrics,
        arrays={},
        meta={
            "LADDER_REVISION": LADDER_REVISION,
            "input_kind": "real",
            "run_id": run_id,
            "arm": arm,
            "round": round,
            "layers": layers,
            "n_seeds": n_seeds,
            "manifest": str(manifest_path),
            "permute_labels_seed": permute_labels_seed,
            "model_cfg": OmegaConf.to_container(model_cfg, resolve=True),
        },
    )


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--manifest", required=True, help="AFEW-VA probe-role manifest.csv"
    )
    parser.add_argument("--mesh-shared", default=str(Path.home() / "stern_mesh"))
    parser.add_argument(
        "--configs", default=str(Path("runs/state/mesh_run_configs.json"))
    )
    parser.add_argument(
        "--model-cfg", default=str(Path("configs/model/videomae_base.yaml"))
    )
    parser.add_argument(
        "--experiment-dir", default=str(Path("configs/experiment"))
    )
    parser.add_argument(
        "--layers",
        default=",".join(str(layer) for layer in DEFAULT_PROBE_DEPTH_LAYERS),
        help="comma-separated 1-based block outputs to probe",
    )
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--round", type=int, default=None)
    parser.add_argument(
        "--permute-labels-seed",
        type=int,
        default=None,
        help="negative control: break the token/label pairing before fitting, "
        "so a non-zero CCC means the protocol leaks",
    )
    parser.add_argument(
        "--arm",
        choices=("paired", "checkpoint", "control"),
        default="paired",
        help="paired: both arms plus the final block, as one report; "
        "checkpoint/control: that arm only, at exactly --layers",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.arm != "paired" and args.permute_labels_seed is not None:
        raise SystemExit(
            "--permute-labels-seed needs both arms under one permutation; "
            "run it with --arm paired"
        )

    layers = [int(part) for part in args.layers.split(",") if part.strip()]
    entry = run_hyperparams(args.run_id, args.configs)
    if not entry:
        raise SystemExit(
            f"{args.run_id}: no entry in {args.configs} -- its model geometry "
            "is unrecoverable, and probing it under the base config would "
            "report an architecture it may never have had"
        )

    model_cfg = model_cfg_for_run(
        OmegaConf.load(args.model_cfg), entry, args.experiment_dir
    )
    bundle, control_bundle = _arm_bundles(
        args.run_id,
        args.mesh_shared,
        model_cfg,
        round=args.round,
        load=args.arm != "control",
    )

    manifest_path = Path(args.manifest)
    if args.arm != "paired":
        is_control = args.arm == "control"
        values = _single_arm(
            control_bundle if is_control else bundle,
            manifest_path,
            layers,
            seed_base=_CONTROL_SEED_BASE if is_control else _CHECKPOINT_SEED_BASE,
            n_seeds=args.n_seeds,
        )
        suffix = "_control" if is_control else ""
        _report(
            {f"probe_ccc_L{layer}{suffix}": value for layer, value in values.items()},
            arm=args.arm,
            run_id=args.run_id,
            round=args.round,
            layers=layers,
            n_seeds=args.n_seeds,
            manifest_path=manifest_path,
            permute_labels_seed=None,
            model_cfg=model_cfg,
        ).to_json(args.out)
        print(f"{args.run_id} [{args.arm}]: probe CCC over {args.n_seeds} seeds")
        for layer, value in values.items():
            print(f"  L{layer:<3} {value.mean:+.4f} +- {value.sd:.4f}")
        print(f"wrote {args.out}")
        return 0

    per_layer, final_block_reference = probe_by_depth(
        bundle,
        control_bundle,
        manifest_path,
        layers,
        n_seeds=args.n_seeds,
        permute_labels_seed=args.permute_labels_seed,
    )

    metrics = {}
    for layer, (checkpoint_value, control_value) in per_layer.items():
        metrics[f"probe_ccc_L{layer}"] = checkpoint_value
        metrics[f"probe_ccc_L{layer}_control"] = control_value
    metrics["probe_ccc_final_block"] = final_block_reference
    report = _report(
        metrics,
        arm=args.arm,
        run_id=args.run_id,
        round=args.round,
        layers=layers,
        n_seeds=args.n_seeds,
        manifest_path=manifest_path,
        permute_labels_seed=args.permute_labels_seed,
        model_cfg=model_cfg,
    )
    report.to_json(args.out)

    print(f"{args.run_id}: probe CCC (mean +- sd over {args.n_seeds} probe seeds)")
    for layer in layers:
        checkpoint_value, control_value = per_layer[layer]
        print(
            f"  L{layer:<3} {checkpoint_value.mean:+.4f} +- {checkpoint_value.sd:.4f}"
            f"   random_init {control_value.mean:+.4f} +- {control_value.sd:.4f}"
            f"   delta {checkpoint_value.mean - control_value.mean:+.4f}"
        )
    print(
        f"  final block {final_block_reference.mean:+.4f} "
        f"+- {final_block_reference.sd:.4f}"
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
