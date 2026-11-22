"""How much target motion does time-averaging destroy?

Motion-blindness hypothesis H1 assumes the training data has inter-frame change
worth keeping. This measures that without training anything. Time-averaging a
tubelet is linear, so the variance it removes has a closed form in the two
layouts' per-token sigmas:

    frac_var_lost = 1 - (sigma_time_avg / sigma_per_frame)^2

If this is negligible on real shards, the per_frame arm is a null intervention.
The factorial would then show no effect for a reason unrelated to the model.

The same measurement covers H3, which is not in the factorial. Windows are built
at stride 1, so with tubelet_size 2 a tubelet spans two adjacent frames. Reporting
the budget across strides shows whether a wider stride restores the signal. That
is a config change, not a retrain.

    ~/.local/bin/uv run --no-sync python -m stern.evaluation.diagnostics.motion_budget \\
        --shard-path <shard> --out runs/results/motion_budget.json
"""

from __future__ import annotations

import torch

from stern.evaluation.diagnostics.reconstruction import patch_stats

# Below REFUTED_BELOW, time-averaging destroys so little of the target that H1
# cannot explain motion blindness. Above SUPPORTED_ABOVE it destroys enough that
# the per_frame arm has something real to recover. Between, the factorial is
# worth running but the effect size should be expected to be small.
REFUTED_BELOW = 0.02
SUPPORTED_ABOVE = 0.10


def subsample_stride(
    pixel_values: torch.Tensor, stride: int, num_frames: int
) -> torch.Tensor:
    """Take `num_frames` frames `stride` apart, from the front of the clip."""
    needed = (num_frames - 1) * stride + 1
    available = pixel_values.shape[2]
    if available < needed:
        raise ValueError(
            f"stride {stride} x {num_frames} frames needs {needed} frames, "
            f"clip has {available}"
        )
    return pixel_values[:, :, :needed:stride]


def motion_budget(
    pixel_values: torch.Tensor, tubelet_size: int, patch_size: int
) -> dict:
    """Fraction of within-token target variance that time-averaging destroys.

    Pooled as a ratio of summed variances, not a mean of per-token ratios. This
    stays in [0, 1) and weights each token by its variance. Do not average
    per-token ratios: a black patch has sigma = 0 in both layouts, and its
    guarded ratio then yields a negative fraction. Letterboxing and dark scenes
    make such patches common.

    The pooled ratio is still invariant to clip brightness and contrast, so it
    tracks motion rather than exposure.
    """
    _, _, sigma_avg, _ = patch_stats(
        pixel_values, tubelet_size, patch_size, per_frame=False
    )
    _, _, sigma_pf, _ = patch_stats(
        pixel_values, tubelet_size, patch_size, per_frame=True
    )
    var_avg = float((sigma_avg**2).sum())
    var_pf = float((sigma_pf**2).sum())
    return {
        "sigma_ratio": (var_pf / var_avg) ** 0.5 if var_avg > 0 else 1.0,
        # Clamped at 0: sigma_ta <= sigma_pf always, so a negative value is
        # float noise from a clip with no variance at all to measure.
        "frac_var_lost": max(0.0, 1.0 - var_avg / var_pf) if var_pf > 0 else 0.0,
        "tubelet_size": tubelet_size,
        "n_frames": int(pixel_values.shape[2]),
    }


def h1_verdict(frac_var_lost: float) -> str:
    """refuted -> time-averaging destroys too little for H1 to explain anything.
    weak     -> a real but small effect; expect a small factorial effect size.
    holds    -> the premise is sound; the per_frame arm has signal to recover."""
    if frac_var_lost < REFUTED_BELOW:
        return "refuted"
    if frac_var_lost < SUPPORTED_ABOVE:
        return "weak"
    return "holds"


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    from omegaconf import OmegaConf

    from stern.models.videomae.encoder import preprocess_window
    from stern.evaluation.diagnostics.export_vit_internals import load_sample_windows

    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-path", type=Path, required=True)
    ap.add_argument("--strides", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--n-windows", type=int, default=32)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/model/videomae_base.yaml"),
    )
    args = ap.parse_args()

    cfg = OmegaConf.load(args.model_config)
    nf, isz = int(cfg.num_frames), int(cfg.image_size)
    ts, ps = int(cfg.tubelet_size), int(cfg.patch_size)

    # Long enough windows that the widest stride still yields nf frames.
    span = (nf - 1) * max(args.strides) + 1
    windows = load_sample_windows(args.shard_path, args.n_windows, span, isz, args.seed)
    clips = [preprocess_window(f, span, isz) for _k, f in windows]

    report = {}
    print(f"{'stride':>7} {'sigma_ratio':>12} {'frac_var_lost':>14} {'H1':>9}")
    for stride in args.strides:
        budgets = [
            motion_budget(subsample_stride(c, stride, nf), ts, ps) for c in clips
        ]
        lost = sum(b["frac_var_lost"] for b in budgets) / len(budgets)
        ratio = sum(b["sigma_ratio"] for b in budgets) / len(budgets)
        report[f"stride_{stride}"] = {
            "sigma_ratio": ratio,
            "frac_var_lost": lost,
            "h1_verdict": h1_verdict(lost),
        }
        print(f"{stride:>7} {ratio:12.4f} {lost:14.4f} {h1_verdict(lost):>9}")

    args.out.write_text(json.dumps(report, indent=1))
    print(f"wrote {args.out}")
    print(
        "\nstride 1 is what training uses today (windows.py builds consecutive\n"
        "frames). A budget that is 'refuted' at stride 1 and 'holds' at stride 4\n"
        "means the sampling rate, not the objective, is what erased the motion."
    )


if __name__ == "__main__":
    main()
