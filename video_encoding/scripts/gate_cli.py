"""AFEW-VA gate CLI: one arm (an in-house snapshot, a second random init, or
the HF LeVJEPA ViT-L ceiling) fine-tuned over the gate's seeds and classified
against the seeded in-house random_init control.

Reuses diag_finetune_afewva's recipe (finetune_ccc, 1 epoch, lr 1e-5, unbounded
head, identity-disjoint split), its per-seed fingerprinted arm cache and its
paired t-interval. Its `gate()` is not used because it gives every arm one
model_cfg, and the HF ceiling needs its own.

  --ckpt PATH    a mesh master.pt / snapshot (encoder.* keys, strict load)
  --ckpt random  a second random init (seed 1) vs the control (seed 0): A/A
  --ckpt hf      the released LeVJEPA ViT-L ($LEVJEPA_HF_PATH)

Writes <out_dir>/<name>.json (PASS / FAIL / UNRESOLVED, ci95, mde) and
appends a line to <out_dir>/summary.tsv.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import statistics
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

from video_encoding.models.levjepa.encoder import LeVJEPAEncoder
from video_encoding.scripts import diag_finetune_afewva as g

STATUS = {"PASS": "PASS", "FAIL": "FAIL", "INCONCLUSIVE": "UNRESOLVED"}


def model_cfg(kind: str, geometry: str, overrides: list[str]):
    frames, size = (int(x) for x in geometry.split("x"))
    model = "levjepa_hf_large" if kind == "hf" else "levjepa_lean"
    ov = [f"model={model}"]
    if kind != "hf":  # the release's geometry is fixed
        ov += [f"model.num_frames={frames}", f"model.image_size={size}"]
    with initialize_config_dir(config_dir=str(g.CONF_DIR), version_base=None):
        return compose(
            "config", overrides=["experiment=levjepa_mesh_ssl", *ov, *overrides]
        ).model


def inhouse_encoder(cfg, *, ckpt: str | None, init_seed: int) -> torch.nn.Module:
    torch.manual_seed(init_seed)
    enc = LeVJEPAEncoder(cfg)
    if ckpt is not None:
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        state = state.get("model_state_dict", state)
        if any(k.startswith("encoder.") for k in state):
            state = {
                k[len("encoder.") :]: v
                for k, v in state.items()
                if k.startswith("encoder.")
            }
        # Strict: Stage 2's load only reports missing keys, which would
        # fine-tune a half-random encoder if the run used other model flags.
        enc.load_state_dict(state, strict=True)
    return enc


def hf_encoder(cfg) -> torch.nn.Module:
    from video_encoding.models.levjepa_hf import load_hf_levjepa

    return load_hf_levjepa(cfg.weights_path)


def gate_params(args) -> dict:
    """What a cache dir's results depend on. Recipe changes are recorded only
    when set, so default runs keep sharing one cache. `rng_seeded` is always
    set: caches from before the seed fix are refused."""
    params = {
        "min_epochs": args.epochs,
        "gate_seeds": args.seeds,
        "split_fraction": g.SPLIT_FRACTION,
        "lr": g.GATE_LR,
        "head_bounded": g.GATE_HEAD_BOUNDED,
        "rng_seeded": g.GATE_RNG_SEEDED,
    }
    if args.head_lr_mult != 1.0:
        params["head_lr_mult"] = args.head_lr_mult
    if args.head_token_norm:
        params["head_token_norm"] = True
    return params


def run(args) -> dict:
    root = Path(os.environ["AFEWVA_ROOT"])
    cache = Path(args.cache_dir) / args.geometry
    g._check_gate_params(cache, gate_params(args))
    common = dict(
        cache_dir=cache,
        n_seeds=args.seeds,
        finetune=functools.partial(
            g.finetune_ccc,
            head_lr_mult=args.head_lr_mult,
            persist_dir=None if args.persist_dir is None else Path(args.persist_dir),
        ),
        manifest_root=root,
        min_epochs=args.epochs,
        seed_subset=args.seed_subset,
    )

    lean_overrides = list(args.override)
    if args.head_token_norm:
        lean_overrides.append("+model.head_token_norm=true")
    lean = model_cfg("inhouse", args.geometry, lean_overrides)
    control = None if args.no_control else g._run_arm(
        "random_init",
        inhouse_encoder(lean, ckpt=None, init_seed=0),
        model_cfg=lean,
        **common,
    )
    if args.ckpt == "hf":
        cfg = model_cfg("hf", args.geometry, [])
        arm = g._run_arm("ceiling_levjepa_hf", hf_encoder(cfg), model_cfg=cfg, **common)
    elif args.ckpt == "random":
        arm = g._run_arm(
            "random_init_seed1",
            inhouse_encoder(lean, ckpt=None, init_seed=1),
            model_cfg=lean,
            **common,
        )
    else:
        arm = g._run_arm(
            f"ckpt_{args.name}",
            inhouse_encoder(lean, ckpt=args.ckpt, init_seed=0),
            model_cfg=lean,
            **common,
        )

    a = [r["mean_ccc"] for r in arm]
    # Arm (or a seed subset) cached only; a later full run classifies.
    if control is None or args.seed_subset is not None:
        return {"name": args.name, "status": "ARM_ONLY", "arm_ccc": a}
    c = [r["mean_ccc"] for r in control]
    status, ci95, mde = g.classify(a, c)
    return {
        "name": args.name,
        "ckpt": args.ckpt,
        "geometry": args.geometry,
        "overrides": args.override,
        "status": STATUS[status],
        "diff_mean": statistics.fmean(x - y for x, y in zip(a, c)),
        "ci95": list(ci95),
        "mde": mde,
        "underpowered": status != "PASS" and mde > g.MDE_OF_INTEREST,
        "arm_ccc": a,
        "control_ccc": c,
        "n_seeds": args.seeds,
        "epochs": args.epochs,
        "head_lr_mult": args.head_lr_mult,
        "head_token_norm": args.head_token_norm,
        "node": os.uname().nodename,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ckpt", required=True, help="PATH | random | hf")
    p.add_argument("--name", required=True)
    p.add_argument("--geometry", default="16x224", help="FRAMESxSIZE (in-house arms)")
    p.add_argument(
        "--override",
        action="append",
        default=[],
        help="model.KEY=VAL for the run's architecture flags",
    )
    p.add_argument("--no-control", action="store_true",
                   help="fine-tune and cache this arm only (parallel jobs)")
    p.add_argument("--seeds", type=int, default=g.GATE_SEEDS)
    p.add_argument("--epochs", type=int, default=g.GATE_MIN_EPOCHS)
    p.add_argument("--persist-dir", default=None,
                   help="resumable per-seed work dirs (node-local disk: a ViT-L "
                        "seed's checkpoint is several GB); keep a seed's jobs on one node")
    p.add_argument("--seed-subset", type=lambda s: [int(x) for x in s.split(",")],
                   default=None, help="e.g. 0,1,2: run and cache only these seeds")
    p.add_argument("--head-lr-mult", type=float, default=1.0,
                   help="head LR = lr * this (switches the fine-tune to AdamW, wd 0)")
    p.add_argument("--head-token-norm", action="store_true",
                   help="BatchNorm the tokens before the in-house VA head (arm and control)")
    p.add_argument("--out-dir", default=str(Path.home() / "stern_diag/gate"))
    p.add_argument("--cache-dir", default=str(Path.home() / "stern_diag/gate/cache"))
    args = p.parse_args()
    report = run(args)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if report["status"] == "ARM_ONLY":
        print(json.dumps(report))
        return
    g._write_atomic(out / f"{args.name}.json", report)
    with open(out / "summary.tsv", "a") as f:
        f.write(
            "\t".join(
                str(x)
                for x in (
                    args.name,
                    report["status"],
                    f"{report['diff_mean']:+.4f}",
                    f"[{report['ci95'][0]:+.4f},{report['ci95'][1]:+.4f}]",
                    f"{report['mde']:.4f}",
                    report["geometry"],
                    report["node"],
                )
            )
            + "\n"
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
