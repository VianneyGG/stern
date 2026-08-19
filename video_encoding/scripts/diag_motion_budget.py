"""How much motion does time-averaging actually destroy?

Both motion-blindness hypotheses assume the training data contains inter-frame
change worth preserving, and nothing in the investigation measured that. It is
measurable without training anything: time-averaging a tubelet is a linear
operator, so the target variance it removes follows in closed form from the two
layouts' per-token sigmas.

    frac_var_lost = 1 - (sigma_time_avg / sigma_per_frame)^2

That IS H1's premise as a number. If it comes back negligible on real shards,
`time_avg` erases almost nothing, `per_frame` is a null intervention, and the
factorial would return four indistinguishable numbers that read as "no
mechanism found" when the truth is "there was no motion in the target to lose".

The same measurement answers H3, which is not in the factorial at all:
video_encoding/data/windows.py builds stride-1 windows, so with tubelet_size 2
a tubelet spans two adjacent frames -- 80ms at 25fps. Reporting the budget
across sampling strides shows whether widening the stride restores the signal,
and that is a one-line config change rather than a retrain.

    ~/.local/bin/uv run --no-sync python -m video_encoding.scripts.diag_motion_budget \\
        --shard-path <shard> --out runs/results/motion_budget.json
"""

from __future__ import annotations

import torch

from video_encoding.scripts.diag_reconstruction import patch_stats

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
    return pixel_values[:, :, : needed : stride]


def motion_budget(
    pixel_values: torch.Tensor, tubelet_size: int, patch_size: int
) -> dict:
    """Fraction of within-token target variance that time-averaging destroys.

    Pooled over tokens -- the ratio of summed variances, not the mean of
    per-token ratios. Averaging can only remove variance, so pooling is bounded
    in [0, 1) by construction, and it weights each token by how much variance it
    actually has.

    The per-token mean is not merely noisier, it is wrong on real footage. A
    genuinely black patch has sigma = 0 in both layouts, the division guard then
    makes the ratio 1e-6 / 2e-6 = 0.5, and 1 - 1/0.5**2 = -3 for every such
    token. The first real-data run returned frac_var_lost = -1.12 that way, on
    a quantity that cannot be negative. Letterboxing and dark scenes make those
    patches common; synthetic clips have none.

    Still a ratio of variances, so still invariant to the clip's overall
    brightness and contrast -- it tracks motion, not exposure.
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

    from video_encoding.models.videomaev2.encoder import preprocess_window
    from video_encoding.scripts.export_vit_internals import load_sample_windows

    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-path", type=Path, required=True)
    ap.add_argument("--strides", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--n-windows", type=int, default=32)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--model-config",
        type=Path,
        default=Path("video_encoding/conf/model/videomae_base.yaml"),
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
