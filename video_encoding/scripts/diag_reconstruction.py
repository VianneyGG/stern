"""Reconstruction diagnostics, in z-space only.

Productionised from the scratch scripts on login-node. Two bugs were found in that
code before it was trusted: a degenerate positional R^2 (769 regressors against
160 samples, R^2 = 1.000 for every checkpoint) and pixel-space temporal
contamination. Both produced confident, wrong headline numbers, which is why the
metrics that decide whether to change mae.py live here with tests and the
exploratory ones do not.

Everything is measured in z-space. Inverting the z-score with the ground-truth
per-patch mu/sigma re-injects real structure the model never produced: measured
that way, a noise-emitting model scored a temporal-variance ratio of 0.538 and a
correlation of 0.691.
"""

from __future__ import annotations

import torch

from video_encoding.models.videomaev2.encoder import _IMAGENET_MEAN, _IMAGENET_STD


def temporal_split(
    grid: tuple[int, int, int], device: torch.device, keep_frac: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Visible = every position at early timesteps; masked = everything later.

    Unlike tube masking this leaves no masked column visible at any other
    timestep, so beating "copy the last visible frame" requires modelling
    motion. Returns (keep_indices, mask_indices, cut).
    """
    t_p, h_p, w_p = grid
    n_sp = h_p * w_p
    cut = int(round(t_p * keep_frac))
    idx = torch.arange(t_p * n_sp)
    return idx[idx < cut * n_sp].to(device), idx[idx >= cut * n_sp].to(device), cut


def patch_stats(
    pixel_values: torch.Tensor,
    tubelet_size: int,
    patch_size: int,
    *,
    per_frame: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int, int]]:
    """Un-z-scored patches plus the mu/sigma that would z-score them.

    Mirrors VideoMAEForPreTraining._patchify_targets for either layout, but is
    independent of any model's configured target_mode -- both layouts' statistics
    are needed to put the two arms on one scale.
    """
    B, C, T, H, W = pixel_values.shape
    mean = torch.tensor(
        _IMAGENET_MEAN, device=pixel_values.device, dtype=pixel_values.dtype
    ).view(1, C, 1, 1, 1)
    std = torch.tensor(
        _IMAGENET_STD, device=pixel_values.device, dtype=pixel_values.dtype
    ).view(1, C, 1, 1, 1)
    pv = pixel_values * std + mean

    t_p, h_p, w_p = T // tubelet_size, H // patch_size, W // patch_size
    x = pv.view(B, C, t_p, tubelet_size, H, W)
    if not per_frame:
        x = x.mean(dim=3, keepdim=True)
    f = x.shape[3]
    x = x.view(B, C, t_p, f, h_p, patch_size, w_p, patch_size)
    x = x.permute(0, 2, 4, 6, 3, 1, 5, 7)
    x = x.reshape(B, t_p * h_p * w_p, f * C * patch_size * patch_size)
    return x, x.mean(-1, keepdim=True), x.std(-1, keepdim=True), (t_p, h_p, w_p)


def collapse_to_time_avg_z(
    pred_z: torch.Tensor,
    *,
    target_mode: str,
    tubelet_size: int,
    sigma_per_frame: torch.Tensor,
    sigma_time_avg: torch.Tensor,
) -> torch.Tensor:
    """Map any arm's z-prediction onto the shared time-averaged z-scale.

    For a per_frame prediction, inverting its z-score, averaging over the tubelet
    frames and re-z-scoring against the time-averaged statistics reduces exactly
    to a frame-average rescaled by sigma_pf / sigma_ta -- the two layouts' mu are
    the same number (the same values under the same weights), so the offsets
    cancel. A zero prediction therefore stays a zero prediction, which is what
    puts both arms on a common zero-point of MSE = 1.0.
    """
    if target_mode == "time_avg":
        return pred_z
    B, N, D = pred_z.shape
    avg = pred_z.view(B, N, tubelet_size, D // tubelet_size).mean(dim=2)
    return avg * (sigma_per_frame / (sigma_time_avg + 1e-6))


@torch.no_grad()
def predict_z(
    model,
    pixel_values: torch.Tensor,
    keep_indices: torch.Tensor,
    mask_indices: torch.Tensor,
) -> torch.Tensor:
    """Full-grid z-space prediction. Mirrors the model's forward without slicing
    to masked positions, so a whole reconstruction can be rendered or scored."""
    enc = model.encoder
    x = enc.patch_embed(pixel_values) + enc.pos_embed
    visible = model.encoder_to_decoder(enc.forward_tokens(x, keep_indices))

    dec = visible.new_zeros(pixel_values.shape[0], x.shape[1], model.decoder_embed_dim)
    dec[:, keep_indices, :] = visible
    dec[:, mask_indices, :] = model.mask_token.expand(
        pixel_values.shape[0], mask_indices.shape[0], -1
    ).to(dec.dtype)
    dec = dec + model.decoder_pos_embed
    for block in model.decoder_blocks:
        dec = block(dec)
    return model.predict_head(model.decoder_norm(dec))


@torch.no_grad()
def z_space_prediction(
    model, pixel_values: torch.Tensor, device: torch.device, keep_frac: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(pred_z, target_z, mask_indices) on the shared time-averaged z-scale.

    Every arm's prediction is collapsed onto the one scale here, so each metric
    below is computed on identical quantities regardless of target_mode.
    """
    pixel_values = pixel_values.to(device)
    grid = model.encoder.token_grid
    keep, mask, _ = temporal_split(grid, device, keep_frac)

    pred = predict_z(model, pixel_values, keep, mask)
    target_avg, _, sigma_a, _ = patch_stats(
        pixel_values, model.tubelet_size, model.patch_size, per_frame=False
    )
    _, _, sigma_p, _ = patch_stats(
        pixel_values, model.tubelet_size, model.patch_size, per_frame=True
    )
    mu_a = target_avg.mean(-1, keepdim=True)
    target_z = (target_avg - mu_a) / (sigma_a + 1e-6)

    pred_z = collapse_to_time_avg_z(
        pred,
        target_mode=getattr(model, "target_mode", "time_avg"),
        tubelet_size=model.tubelet_size,
        sigma_per_frame=sigma_p,
        sigma_time_avg=sigma_a,
    )
    return pred_z, target_z, mask


@torch.no_grad()
def holdout_mse_z(
    model, pixel_values: torch.Tensor, device: torch.device, keep_frac: float = 0.5
) -> float:
    """The arm-invariant metric: temporal-hold-out MSE against the shared
    time-averaged z-target.

    MSE = (D-1)/D means "predicted each patch's own mean", in every arm -- not
    1.0, because _patchify_targets z-scores with Bessel-corrected torch.std.
    """
    pred_z, target_z, mask = z_space_prediction(model, pixel_values, device, keep_frac)
    return float(((pred_z[:, mask] - target_z[:, mask]) ** 2).mean())


def _temporal_deviations(
    pred_z: torch.Tensor,
    target_z: torch.Tensor,
    grid: tuple[int, int, int],
    mask_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Both tensors' deviations from their own across-time mean, restricted to
    masked spatial columns.

    Shared by the two temporal metrics so they can never disagree about which
    columns they are describing.
    """
    t_p, h_p, w_p = grid
    n_sp = h_p * w_p
    cols = torch.tensor(sorted({int(i) % n_sp for i in mask_indices.cpu()}))
    tgt = target_z.reshape(t_p, n_sp, -1)[:, cols, :]
    prd = pred_z.reshape(t_p, n_sp, -1)[:, cols, :]
    return prd - prd.mean(0, keepdim=True), tgt - tgt.mean(0, keepdim=True)


def temporal_variance_ratio(
    pred_z: torch.Tensor,
    target_z: torch.Tensor,
    grid: tuple[int, int, int],
    mask_indices: torch.Tensor,
) -> float:
    """Fraction of the target's across-time variance the prediction reproduces.

    A magnitude only: ~0 means the prediction is a temporally static average.
    It is NOT evidence of motion tracking on its own -- a model emitting
    temporal noise uncorrelated with the target also scores ~1.0, and a model
    that captures half the signal scores ~0.5, i.e. below the noise emitter.
    Always read it beside temporal_correlation; see that function.

    Computed in z-space -- see the module docstring for why pixel space is
    invalid here.
    """
    prd_dev, tgt_dev = _temporal_deviations(pred_z, target_z, grid, mask_indices)
    return float((prd_dev**2).mean() / max(float((tgt_dev**2).mean()), 1e-12))


def temporal_correlation(
    pred_z: torch.Tensor,
    target_z: torch.Tensor,
    grid: tuple[int, int, int],
    mask_indices: torch.Tensor,
) -> float:
    """Correlation of the prediction's across-time deviations with the target's.

    The direction half of the temporal diagnosis, and the half
    temporal_variance_ratio structurally cannot express. Read as a pair:

        ratio ~1, corr ~1  -> tracks motion
        ratio ~1, corr ~0  -> emits temporal noise (the trap)
        ratio ~0, corr ~0  -> emits a temporally static average
        ratio ~1, corr <0  -> right magnitude, inverted motion

    A temporally flat prediction has no deviation to correlate; that is reported
    as 0.0 rather than a 0/0 nan.
    """
    prd_dev, tgt_dev = _temporal_deviations(pred_z, target_z, grid, mask_indices)
    denom = float(
        (prd_dev**2).mean().sqrt() * (tgt_dev**2).mean().sqrt()
    )
    if denom < 1e-12:
        return 0.0
    return float((prd_dev * tgt_dev).mean()) / denom


def load_arm(checkpoint_path, model_config_path, mask_type, target_mode, device):
    """Build a model in the arm's own configuration and load its weights.

    The arms' predict_head widths differ, so each checkpoint must be loaded into
    a model built with that arm's target_mode; torch raises on the size mismatch
    otherwise, which is the intended behaviour.
    """
    from omegaconf import OmegaConf

    from video_encoding.mesh.outer_step import to_fp32_state_dict
    from video_encoding.models.videomaev2.mae import VideoMAEForPreTraining
    from video_encoding.training import checkpoint as ckpt_io

    cfg = OmegaConf.load(model_config_path)
    cfg.mask_type, cfg.target_mode = mask_type, target_mode
    model = VideoMAEForPreTraining(cfg)
    state = to_fp32_state_dict(
        ckpt_io.load_checkpoint_file(checkpoint_path)["model_state_dict"]
    )
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(f"{checkpoint_path}: key mismatch")
    return model.eval().to(device)


def main() -> None:
    """Score every factorial arm with the one metric that is comparable across
    them. Arm names are `<mask_type>-<target_mode>`."""
    import argparse
    import json
    from pathlib import Path

    from omegaconf import OmegaConf

    from video_encoding.models.videomaev2.encoder import preprocess_window
    from video_encoding.scripts.export_vit_internals import load_sample_windows

    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-path", type=Path, required=True)
    ap.add_argument("--factorial-root", type=Path, required=True)
    ap.add_argument(
        "--arms",
        nargs="+",
        default=[
            "tube-time_avg",
            "mixed-time_avg",
            "tube-per_frame",
            "mixed-per_frame",
        ],
    )
    ap.add_argument("--n-windows", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--model-config",
        type=Path,
        default=Path("video_encoding/conf/model/videomae_base.yaml"),
    )
    args = ap.parse_args()

    cfg = OmegaConf.load(args.model_config)
    nf, isz = int(cfg.num_frames), int(cfg.image_size)
    device = torch.device(args.device)
    windows = load_sample_windows(args.shard_path, args.n_windows, nf, isz, args.seed)
    pvs = [preprocess_window(f, nf, isz) for _k, f in windows]

    report = {}
    header = f"{'arm':>18} {'holdout_mse_z':>14} {'tvar_ratio':>11} {'tcorr':>7}"
    print(header)
    print("(tvar_ratio alone is not evidence: noise scores ~1.0 there too)")
    for arm in args.arms:
        mask_type, target_mode = arm.split("-", 1)
        model = load_arm(
            args.factorial_root / arm / "best.pt",
            args.model_config,
            mask_type,
            target_mode,
            device,
        )
        grid = model.encoder.token_grid
        mses, ratios, corrs = [], [], []
        for pv in pvs:
            pred_z, target_z, mask = z_space_prediction(model, pv, device)
            mses.append(float(((pred_z[:, mask] - target_z[:, mask]) ** 2).mean()))
            ratios.append(temporal_variance_ratio(pred_z, target_z, grid, mask))
            corrs.append(temporal_correlation(pred_z, target_z, grid, mask))
        report[arm] = {
            "holdout_mse_z": float(sum(mses) / len(mses)),
            "temporal_variance_ratio": float(sum(ratios) / len(ratios)),
            "temporal_correlation": float(sum(corrs) / len(corrs)),
            "mask_type": mask_type,
            "target_mode": target_mode,
        }
        r = report[arm]
        print(
            f"{arm:>18} {r['holdout_mse_z']:14.4f} "
            f"{r['temporal_variance_ratio']:11.4f} {r['temporal_correlation']:7.4f}"
        )

    args.out.write_text(json.dumps(report, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
