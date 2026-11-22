"""Can motion be linearly read out of the frozen encoder's tokens?

"Motion is absent from the representation" and "motion is present but the
decoder ignores it" need different fixes: the first is target-side, the second
objective/decoder-side. A linear readout from frozen tokens to per-patch
frame-difference magnitude tells them apart, scored against a random_init control.

    ~/.local/bin/uv run --no-sync python -m stern.evaluation.probes.motion_decodability \\
        --shard-path <shard> --mesh-shared ~/stern_mesh --out runs/results/motion_decodability.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from stern.mesh.outer_step import to_fp32_state_dict
from stern.models.videomae.encoder import preprocess_window
from stern.models.videomae.mae import VideoMAEForPreTraining
from stern.evaluation.diagnostics.export_vit_internals import load_sample_windows
from stern.training import checkpoint as ckpt_io


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

    Cross-validated because the token count is close to the feature dimension.
    An in-sample fit would give R^2 near 1.0 for every checkpoint, including
    random init.

    The intercept is not penalised. The targets are frame-difference magnitudes
    with a large positive mean. Shrinking the intercept toward zero would stop
    the ridge from reaching that mean, and R^2 would go strongly negative as
    alpha grows for reasons unrelated to the features (-32.8 at alpha=100 on a
    random-init encoder, where the honest value is ~0).
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
    keep = torch.arange(x.shape[1], device=x.device).expand(x.shape[0], -1)
    return enc.forward_tokens(x, keep)[0]


def _load(ckpt_path, cfg, device):
    model = VideoMAEForPreTraining(cfg)
    if ckpt_path is not None:
        state = to_fp32_state_dict(
            ckpt_io.load_checkpoint_file(ckpt_path)["model_state_dict"]
        )
        result = model.load_state_dict(state, strict=False)
        # Checkpoints predating the ViT refactor lack encoder.registers.tokens.
        # At num_registers: 0 that parameter is empty and RegisterTokens passes
        # it through untouched, so its absence leaves the forward pass
        # identical; anything else missing means the weights really don't fit.
        params = dict(model.named_parameters())
        missing = [k for k in result.missing_keys if params[k].numel() > 0]
        if missing or result.unexpected_keys:
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
        default=Path("configs/model/videomae_base.yaml"),
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
