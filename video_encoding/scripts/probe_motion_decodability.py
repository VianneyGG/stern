"""Can motion be linearly read out of the frozen encoder's tokens?

"Motion is absent from the representation" and "motion is present but the
decoder ignores it" are different diseases with different fixes -- the first is
target-side, the second objective/decoder-side -- and no measurement in this
investigation so far distinguishes them. A linear readout from frozen tokens to
per-patch frame-difference magnitude does, scored against a random_init control.

    ~/.local/bin/uv run --no-sync python -m video_encoding.scripts.probe_motion_decodability \\
        --shard-path <shard> --mesh-shared ~/stern_mesh --out runs/results/motion_decodability.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from video_encoding.mesh.outer_step import to_fp32_state_dict
from video_encoding.models.videomaev2.encoder import preprocess_window
from video_encoding.models.videomaev2.mae import VideoMAEForPreTraining
from video_encoding.scripts.export_vit_internals import load_sample_windows
from video_encoding.training import checkpoint as ckpt_io


def frame_diff_targets(
    pixel_values: torch.Tensor, grid: tuple[int, int, int], patch_size: int
) -> torch.Tensor:
    """Mean absolute inter-frame difference per token, in token order.

    One scalar per (t, h, w) token: how much the pixels under that token moved.

    Single-clip only -- the return value has no batch axis to put a second clip
    in, and averaging across clips would silently produce a target that belongs
    to no clip.
    """
    t_p, h_p, w_p = grid
    B, C, T, H, W = pixel_values.shape
    if B != 1:
        raise ValueError(f"frame_diff_targets takes one clip, got batch of {B}")
    diff = (pixel_values[:, :, 1:] - pixel_values[:, :, :-1]).abs().mean(dim=1)
    # pad back to T so the reshape into tubelets is exact
    diff = torch.cat([diff, diff[:, -1:]], dim=1)
    ts = T // t_p
    diff = diff.view(B, t_p, ts, h_p, patch_size, w_p, patch_size)
    return diff.mean(dim=(0, 2, 4, 6)).reshape(-1)


def ridge_r2(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    alpha: float = 1.0,
    n_folds: int = 2,
    seed: int = 0,
) -> float:
    """Cross-validated R^2 of a ridge readout, solved in closed form.

    Cross-validated because the token count is comparable to the feature
    dimension: an in-sample fit would report R^2 ~ 1.0 for every checkpoint,
    which is exactly the degenerate result that invalidated an earlier
    positional probe in this investigation.

    The intercept is exempt from the penalty. The probe's targets are
    frame-difference *magnitudes* -- absolute values with a large positive mean
    -- and shrinking the intercept toward zero stops the ridge from reaching
    that mean, driving R^2 arbitrarily negative as alpha grows for reasons
    unrelated to the features (measured: -32.8 at alpha=100 on a random-init
    encoder, where the honest answer is ~0).
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(targets))
    folds = np.array_split(order, n_folds)
    preds = np.empty_like(targets, dtype=float)
    for i, test_idx in enumerate(folds):
        train_idx = np.concatenate([f for j, f in enumerate(folds) if j != i])
        x, y = features[train_idx], targets[train_idx]
        x = np.concatenate([x, np.ones((len(x), 1))], axis=1)
        penalty = alpha * np.eye(x.shape[1])
        penalty[-1, -1] = 0.0  # the intercept column is not shrunk
        w = np.linalg.solve(x.T @ x + penalty, x.T @ y)
        xt = np.concatenate([features[test_idx], np.ones((len(test_idx), 1))], axis=1)
        preds[test_idx] = xt @ w
    ss_res = float(((targets - preds) ** 2).sum())
    ss_tot = float(((targets - targets.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


@torch.no_grad()
def _tokens(model, pixel_values: torch.Tensor) -> torch.Tensor:
    enc = model.encoder
    x = enc.patch_embed(pixel_values) + enc.pos_embed
    return enc.forward_tokens(x, torch.arange(x.shape[1], device=x.device))[0]


def _load(ckpt_path, cfg, device):
    model = VideoMAEForPreTraining(cfg)
    if ckpt_path is not None:
        state = to_fp32_state_dict(
            ckpt_io.load_checkpoint_file(ckpt_path)["model_state_dict"]
        )
        result = model.load_state_dict(state, strict=False)
        if result.missing_keys or result.unexpected_keys:
            raise ValueError(f"{ckpt_path}: key mismatch")
    return model.eval().to(device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-path", type=Path, required=True)
    ap.add_argument("--mesh-shared", type=Path, required=True)
    ap.add_argument("--run-id", default="pool-cars-best-120r")
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

    ckpt_dir = args.mesh_shared / "checkpoints" / args.run_id
    entries = [("random_init", None)]
    entries += [(f"r{r}", ckpt_dir / f"master_round{r}.pt") for r in (20, 120)]
    entries += [("latest", ckpt_dir / "master.pt")]
    entries = [(k, p) for k, p in entries if p is None or p.exists()]

    report = {}
    for label, path in entries:
        model = _load(path, cfg, device)
        grid = model.encoder.token_grid
        feats, targs = [], []
        for pv in pvs:
            pv = pv.to(device)
            feats.append(_tokens(model, pv).float().cpu().numpy())
            targs.append(frame_diff_targets(pv, grid, model.patch_size).cpu().numpy())
        r2 = ridge_r2(np.concatenate(feats), np.concatenate(targs), seed=args.seed)
        report[label] = {"motion_r2": r2}
        print(f"{label:>12} motion_r2={r2:7.4f}")

    args.out.write_text(json.dumps(report, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
