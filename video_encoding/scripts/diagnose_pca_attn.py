"""PCA and attention-map visualisation over the recipe2x2 checkpoints.

Follow-up to `diagnose_conditioning.py`'s spectrum-stats finding (activation
effective rank collapses from block 1 in the "paper" recipe). This script
makes that finding visible: per-block PCA of token activations (does the
token cloud collapse toward a line/point as depth increases?) and per-block
attention maps (do heads stay differentiated or go uniform/degenerate?).

Caveat (same as diagnose_conditioning.py): fixed-seed random input, not real
video data -- isolates architecture/weight-space collapse, not data-dependent
collapse.

Caveat (mirror drift risk, same class as export_vit_internals -- see
docs/superpowers/plans -- and the "export_vit_internals mirror drift" note):
`forward_with_attn` below hand-mirrors `ViTBlock.forward`
(video_encoding/models/ViT/block.py) so it can request `need_weights=True`
from attention, which `ViTBlock.forward` never does. If block.py's forward
changes, this mirror will silently drift. Not parity-tested; treat plots as
approximate if block.py changes after this script was written.

Usage:
    PYTHONPATH=. uv run --no-sync python video_encoding/scripts/diagnose_pca_attn.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from video_encoding.models.videomaev2.masking import RandomTubeMasking
from video_encoding.scripts._diag_common import load_bundle
from video_encoding.scripts.diagnose_conditioning import (
    BATCH_SIZE,
    CHECKPOINTS,
    SEED,
    _run_id_and_shared,
    load_encoder,
)

OUT_DIR = Path.home() / "stern" / "runs" / "results" / "diagnostics" / "pca_attn"
DEPTH_SAMPLES = (1, 6, 12)  # early / mid / late blocks to visualise in detail
VAR_THRESHOLD = 0.90


def forward_with_attn(encoder, x: torch.Tensor, keep_indices: torch.Tensor):
    """Mirrors ViTBlock.forward + VideoMAEEncoder.forward_tokens, but asks
    attention for weights (need_weights=True), which the real forward path
    never does. See module docstring's mirror-drift caveat."""
    idx = keep_indices.unsqueeze(-1).expand(-1, -1, x.shape[-1])
    x = torch.gather(x, 1, idx)
    x = encoder.registers.prepend(x)

    acts: list[torch.Tensor] = []
    attn_maps: list[torch.Tensor] = []
    for block in encoder.blocks:
        h = block.norm1(x)
        attn_out, attn_w = block.attn(h, need_weights=True)
        x = x + block.drop_path(block.layer_scale1(attn_out))
        x = x + block.drop_path(block.layer_scale2(block.mlp(block.norm2(x))))
        acts.append(encoder.norm(encoder.registers.strip(x)).detach())
        attn_maps.append(attn_w.detach())  # (B, heads, N, N)
    return acts, attn_maps


def pca_components_for_variance(
    mat: torch.Tensor, threshold: float
) -> tuple[int, torch.Tensor]:
    """mat: (N, D) activations. Returns (#components for `threshold` cumulative
    explained variance, top-2 principal-component projection (N, 2))."""
    centered = mat - mat.mean(dim=0, keepdim=True)
    q = min(centered.shape[0], centered.shape[1], 50)
    _, s, v = torch.pca_lowrank(centered, q=q)
    var = s.pow(2)
    cum = torch.cumsum(var, dim=0) / var.sum().clamp_min(1e-12)
    n_components = int((cum < threshold).sum().item()) + 1
    proj2d = centered @ v[:, :2]
    return n_components, proj2d


def load_model(name: str, path: Path, device: torch.device):
    _, model_cfg = load_encoder(device)
    run_id, mesh_shared = _run_id_and_shared(path)
    bundle = load_bundle(run_id, mesh_shared, model_cfg, full_model=False)
    encoder = bundle.encoder.to(device)
    return encoder, model_cfg


def analyze_checkpoint(name: str, path: Path, device: torch.device):
    print(f"\n=== {name} ({path}) ===")
    encoder, model_cfg = load_model(name, path, device)
    depth = int(model_cfg.depth)

    torch.manual_seed(SEED)
    grid = encoder.token_grid
    pixel_values = torch.randn(
        BATCH_SIZE,
        int(model_cfg.in_channels),
        int(model_cfg.num_frames),
        int(model_cfg.image_size),
        int(model_cfg.image_size),
        device=device,
    )
    masker = RandomTubeMasking(mask_ratio=float(model_cfg.mask_ratio))
    keep_indices, _ = masker(pixel_values, grid, device, BATCH_SIZE)

    with torch.no_grad():
        x = encoder.patch_embed(pixel_values) + encoder.pos_embed
        acts, attn_maps = forward_with_attn(encoder, x, keep_indices)

    n_components_per_block = []
    scatter_by_depth = {}
    for i in range(1, depth + 1):
        act_mat = acts[i - 1].reshape(-1, acts[i - 1].shape[-1])
        n_comp, proj2d = pca_components_for_variance(act_mat, VAR_THRESHOLD)
        n_components_per_block.append(n_comp)
        print(f"  block {i:2d}: PCs for {int(VAR_THRESHOLD * 100)}% var = {n_comp}")
        if i in DEPTH_SAMPLES:
            scatter_by_depth[i] = proj2d.cpu().numpy()

    attn_by_depth = {}
    for i in DEPTH_SAMPLES:
        # (B, heads, N, N) -> average over batch and heads -> (N, N)
        attn_by_depth[i] = attn_maps[i - 1].mean(dim=(0, 1)).cpu().numpy()

    return n_components_per_block, scatter_by_depth, attn_by_depth


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for name, path in CHECKPOINTS.items():
        if not path.exists():
            print(f"\n=== {name} === MISSING: {path}")
            continue
        try:
            results[name] = analyze_checkpoint(name, path, device)
        except Exception as exc:  # one bad checkpoint should not abort the rest
            print(f"  ERROR analyzing {name}: {exc}")

    if not results:
        print("no results, skipping plots")
        return

    # --- Figure 1: PCs-for-90%-variance vs depth, one line per checkpoint ---
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (n_components, _, _) in results.items():
        ax.plot(range(1, len(n_components) + 1), n_components, marker="o", label=name)
    ax.set_xlabel("encoder block")
    ax.set_ylabel(f"# PCs for {int(VAR_THRESHOLD * 100)}% variance")
    ax.set_title("Activation PCA dimensionality vs depth")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pca_dim_vs_depth.png", dpi=150)
    plt.close(fig)

    # --- Figure 2: PC1-PC2 scatter grid (rows=checkpoints, cols=DEPTH_SAMPLES) ---
    names = list(results.keys())
    fig, axes = plt.subplots(
        len(names), len(DEPTH_SAMPLES), figsize=(4 * len(DEPTH_SAMPLES), 4 * len(names))
    )
    for r, name in enumerate(names):
        _, scatter_by_depth, _ = results[name]
        for c, depth_i in enumerate(DEPTH_SAMPLES):
            axp = axes[r][c] if len(names) > 1 else axes[c]
            pts = scatter_by_depth.get(depth_i)
            if pts is not None:
                axp.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.5)
            axp.set_title(f"{name} @ blk{depth_i}", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pca_scatter_grid.png", dpi=150)
    plt.close(fig)

    # --- Figure 3: attention heatmap grid (rows=checkpoints, cols=DEPTH_SAMPLES) ---
    fig, axes = plt.subplots(
        len(names), len(DEPTH_SAMPLES), figsize=(4 * len(DEPTH_SAMPLES), 4 * len(names))
    )
    for r, name in enumerate(names):
        _, _, attn_by_depth = results[name]
        for c, depth_i in enumerate(DEPTH_SAMPLES):
            axp = axes[r][c] if len(names) > 1 else axes[c]
            m = attn_by_depth.get(depth_i)
            if m is not None:
                im = axp.imshow(m, cmap="viridis", aspect="auto")
                fig.colorbar(im, ax=axp, fraction=0.046)
            axp.set_title(f"{name} @ blk{depth_i}", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "attention_maps_grid.png", dpi=150)
    plt.close(fig)

    print(f"\nplots written to {OUT_DIR}")


if __name__ == "__main__":
    main()
