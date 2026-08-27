"""Render the two ViT-internals diagnostic PNGs from an
`export_vit_internals.py` `.npz` -- pure plotting, no torch, no checkpoints,
fast to iterate on locally.

    uv run python -m video_encoding.scripts.plot_vit_internals \\
        --npz internals.npz --out-dir runs/results/vit_internals/<timestamp>/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_contact_sheet(npz: dict, out_path: Path) -> None:
    run_ids = [str(r) for r in npz["run_ids"]]
    n_runs = len(run_ids)
    middle_frames = npz["middle_frames"]  # (W, H, Wimg, 3)
    received_maps = npz["received_maps"]  # (R, W, H', W')
    pca_maps = npz["pca_maps"]  # (R, W, H', W', 3)
    n_windows = middle_frames.shape[0]

    fig, axes = plt.subplots(
        n_runs,
        n_windows * 3,
        figsize=(3 * n_windows * 3, 3 * n_runs),
        squeeze=False,
    )
    for r, run_id in enumerate(run_ids):
        for w in range(n_windows):
            frame = middle_frames[w]
            h, wid = frame.shape[:2]
            extent = (0, wid, h, 0)

            ax_raw = axes[r][w * 3 + 0]
            ax_raw.imshow(frame)

            ax_attn = axes[r][w * 3 + 1]
            ax_attn.imshow(frame)
            ax_attn.imshow(
                received_maps[r, w],
                cmap="inferno",
                alpha=0.6,
                extent=extent,
                interpolation="bilinear",
            )

            ax_pca = axes[r][w * 3 + 2]
            ax_pca.imshow(pca_maps[r, w], extent=extent, interpolation="nearest")

            for ax in (ax_raw, ax_attn, ax_pca):
                ax.set_xticks([])
                ax.set_yticks([])

        axes[r][0].set_ylabel(run_id, fontsize=8)

    for w in range(n_windows):
        axes[0][w * 3 + 0].set_title(f"win{w} raw", fontsize=8)
        axes[0][w * 3 + 1].set_title(f"win{w} attn (last layer)", fontsize=8)
        axes[0][w * 3 + 2].set_title(f"win{w} PCA-RGB", fontsize=8)

    fig.suptitle(
        "PCA-RGB colors are fit independently per row -- not comparable across checkpoints",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_layer_diagnostics(npz: dict, out_path: Path) -> None:
    run_ids = [str(r) for r in npz["run_ids"]]
    n_runs = len(run_ids)
    cos_sim = npz["cos_sim"]  # (R, depth-1)
    norm_hist = npz["norm_hist"]  # (R, depth, n_bins)
    bin_edges = npz["bin_edges"]  # (n_bins+1,)
    depth = norm_hist.shape[1]
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])  # geometric mean, log bins

    fig, axes = plt.subplots(2, n_runs, figsize=(3 * n_runs, 6), squeeze=False)
    layer_pairs = np.arange(1, cos_sim.shape[1] + 1)
    layer_idx = np.arange(1, depth + 1)
    cos_lo, cos_hi = float(cos_sim.min()), float(cos_sim.max())
    vmax = float(norm_hist.max())

    im = None
    for i, run_id in enumerate(run_ids):
        ax_top = axes[0][i]
        ax_top.plot(layer_pairs, cos_sim[i], marker="o", ms=3)
        ax_top.set_ylim(cos_lo - 0.02, cos_hi + 0.02)
        ax_top.set_title(run_id, fontsize=8)
        ax_top.set_xlabel("layer pair", fontsize=7)
        if i == 0:
            ax_top.set_ylabel("consecutive cos sim", fontsize=7)

        ax_bot = axes[1][i]
        im = ax_bot.imshow(
            norm_hist[i].T,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            vmin=0,
            vmax=vmax,
            extent=(layer_idx[0] - 0.5, layer_idx[-1] + 0.5, 0, len(bin_centers)),
        )
        ax_bot.set_xlabel("layer", fontsize=7)
        if i == 0:
            ax_bot.set_ylabel("attn-branch output norm", fontsize=7)
            n_ticks = 5
            tick_pos = np.linspace(0, len(bin_centers) - 1, n_ticks)
            ax_bot.set_yticks(tick_pos)
            ax_bot.set_yticklabels(
                [f"{bin_centers[int(p)]:.1f}" for p in tick_pos], fontsize=6
            )
        else:
            ax_bot.set_yticks([])

    fig.colorbar(im, ax=axes[1].tolist(), label="proportion of tokens", shrink=0.8)
    fig.suptitle(
        "Per-layer diagnostics -- shared axes and color scale across checkpoints"
    )
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    npz = np.load(args.npz)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    contact_sheet_path = args.out_dir / "contact_sheet.png"
    layer_diagnostics_path = args.out_dir / "layer_diagnostics.png"
    plot_contact_sheet(npz, contact_sheet_path)
    plot_layer_diagnostics(npz, layer_diagnostics_path)
    print(f"wrote {contact_sheet_path}")
    print(f"wrote {layer_diagnostics_path}")


if __name__ == "__main__":
    main()
