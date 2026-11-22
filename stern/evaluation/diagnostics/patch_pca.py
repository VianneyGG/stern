"""Patch-token PCA -> RGB and query-patch cosine maps on one AFEW-VA clip.

Reproduces the qualitative figures of the LeVJEPA preprint (Kuhn et al.,
arXiv:2608.27395, Figs 3, 5a and 5b) for three encoders in one grid: an untrained
levjepa_lean control, a trained mesh checkpoint, and the released ViT-L.

Choices the paper leaves open: features are the final normed output, CLS dropped;
one PCA is fitted per model over all frames of the clip, so colours are
consistent across time; no foreground thresholding.

Usage (on the GPU host that holds the released weights and AFEW-VA):
    uv run --no-sync python -m stern.evaluation.diagnostics.patch_pca \
        --clip 001 --query 8,5,7 --out levjepa_patch_pca.png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]


def final_patch_grid(encoder, pixel_values: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
    """`pixel_values` (1, C, T, H, W) -> final-layer patch tokens (T, H/p, W/p, D).

    Tokens come frame-major then row-major after the leading CLS, for both the
    in-house encoder and the released one."""
    with torch.no_grad():
        if hasattr(encoder, "config"):  # the released HF model
            out = encoder(pixel_values=pixel_values).last_hidden_state
        else:
            out = encoder(pixel_values)
    tokens = out[0, 1:].float()
    side_h = pixel_values.shape[-2] // patch_size
    side_w = pixel_values.shape[-1] // patch_size
    return tokens.reshape(-1, side_h, side_w, tokens.shape[-1])


def pca_rgb(feats: torch.Tensor) -> np.ndarray:
    """(N, D) -> (N, 3) in [0, 1]: top three principal components, each min-max
    scaled."""
    x = feats.double().numpy()
    x = x - x.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    proj = u[:, :3] * s[:3]
    lo, hi = proj.min(axis=0), proj.max(axis=0)
    return (proj - lo) / np.where(hi > lo, hi - lo, 1.0)


def cosine_map(grid: torch.Tensor, query: tuple[int, int, int]) -> np.ndarray:
    """Cosine similarity of patch `query` (t, i, j) to every patch: (T, h, w).

    The clip's mean token is subtracted first, as PCA does: in these encoders
    89-99% of every token's squared norm is that shared vector, which pins raw
    cosine near 1 everywhere."""
    centered = grid.float() - grid.float().reshape(-1, grid.shape[-1]).mean(dim=0)
    unit = torch.nn.functional.normalize(centered, dim=-1)
    return (unit @ unit[query]).numpy()


def _load_clip(clip: str, num_frames: int, image_size: int) -> torch.Tensor:
    """The clip's first window, built exactly as the probe builds AFEW-VA input."""
    from stern.data.manifest import read_manifest
    from stern.evaluation.diagnostics.probe import _probe_dataset

    manifest = Path(os.environ["AFEWVA_ROOT"]) / "manifest.csv"
    records = [r for r in read_manifest(manifest, expected_role="probe") if r.clip_path == clip]
    if not records:
        raise SystemExit(f"clip {clip!r} not in {manifest}")
    ds = _probe_dataset(records, manifest.parent, num_frames=num_frames, image_size=image_size)
    return ds[0][0].unsqueeze(0)


def _encoders(run_id: str, mesh_shared: str, hf_path: str, seed: int):
    from stern.models.levjepa_hf import load_hf_levjepa
    from stern.evaluation.diagnostics.common import load_bundle, random_init_bundle

    lean_cfg = OmegaConf.load(REPO / "configs/model/levjepa_lean.yaml")
    torch.manual_seed(seed)
    yield f"random init (lean, seed {seed})", random_init_bundle(lean_cfg).encoder
    yield f"{run_id} (lean)", load_bundle(run_id, mesh_shared, lean_cfg).encoder.eval()
    yield "official LeVJEPA ViT-L", load_hf_levjepa(hf_path).eval()


def _denorm(pixel_values: torch.Tensor) -> np.ndarray:
    """(1, C, T, H, W) ImageNet-normalised -> (T, H, W, C) in [0, 1]."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
    return (pixel_values[0] * std + mean).clamp(0, 1).permute(1, 2, 3, 0).numpy()


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--clip", default="001", help="AFEW-VA clip id (manifest clip_path)")
    p.add_argument("--query", default="10,4,4", help="query patch t,i,j for the cosine map")
    p.add_argument("--frames", default="0,5,10,15", help="frames to show")
    p.add_argument("--run-id", default="long-lean-pool64")
    p.add_argument("--mesh-shared", default=str(Path.home() / "stern_mesh"))
    p.add_argument("--hf-path", default=os.environ.get("LEVJEPA_HF_PATH", ""))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--image-size", type=int, default=224,
                   help="input side in px; both encoders use RoPE, so e.g. 448 gives a 28x28 grid")
    p.add_argument("--out", default="levjepa_patch_pca.png")
    a = p.parse_args()
    query = tuple(int(v) for v in a.query.split(","))
    frames = [int(v) for v in a.frames.split(",")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pixel_values = _load_clip(a.clip, 16, a.image_size)
    images = _denorm(pixel_values)

    rows = []
    for name, encoder in _encoders(a.run_id, a.mesh_shared, a.hf_path, a.seed):
        grid = final_patch_grid(encoder.to(device), pixel_values.to(device)).cpu()
        T, h, w, D = grid.shape
        rgb = pca_rgb(grid.reshape(-1, D)).reshape(T, h, w, 3)
        rows.append((name, rgb, cosine_map(grid, query)))
        encoder.cpu()
        print(f"{name}: grid {tuple(grid.shape)}", flush=True)

    n_rows = 1 + 2 * len(rows)
    fig, axes = plt.subplots(n_rows, len(frames), figsize=(2.4 * len(frames), 2.4 * n_rows))
    patch = a.image_size // h
    for c, t in enumerate(frames):
        axes[0, c].imshow(images[t])
        axes[0, c].set_title(f"frame {t}", fontsize=9)
        if t == query[0]:
            axes[0, c].scatter([(query[2] + 0.5) * patch], [(query[1] + 0.5) * patch],
                               marker="x", c="red", s=60)
        for r, (name, rgb, sim) in enumerate(rows):
            axes[1 + 2 * r, c].imshow(rgb[t], interpolation="nearest")
            axes[2 + 2 * r, c].imshow(sim[t], cmap="inferno", vmin=-1, vmax=1,
                                      interpolation="nearest")
    for r, (name, _, _) in enumerate(rows):
        axes[1 + 2 * r, 0].set_ylabel(f"{name}\nPCA", fontsize=8)
        axes[2 + 2 * r, 0].set_ylabel(f"{name}\ncos to query", fontsize=8)
    axes[0, 0].set_ylabel(f"AFEW-VA {a.clip}", fontsize=8)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
