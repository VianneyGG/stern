"""Export ViT-internals diagnostics (attention maps, PCA-RGB tokens, per-layer
cosine similarity, per-layer attention-norm density) from cars-sweep DiLoCo
checkpoints, for `plot_vit_internals.py` to render.

Runs on a pool node (has both the node-local WebDataset shards and the
shared-NFS `stern_mesh/checkpoints`), e.g.:

    ssh <node> "cd ~/stern && ~/.local/bin/uv run --no-sync python -m \\
        stern.evaluation.diagnostics.export_vit_internals \\
        --shard-path /Data/$USER/voxceleb2_shards/shard-000000.tar \\
        --mesh-shared ~/stern_mesh \\
        --run-ids sw3-cars-lr060m090,sw3-cars-lr080m090,sw3-cars-accum128,sw3-cars-lrmult15,sw3-cars-wd010,pool-cars-full-50r \\
        --include-random-init --out internals.npz"

`master.pt`'s `model_state_dict` is a `VideoMAEForPreTraining` state dict (the
mesh trains encoder+decoder), so it must be loaded into that full module and
`.encoder` taken back out -- loading straight into a bare `VideoMAEEncoder`
matches zero keys and, under `strict=False`, silently yields a random encoder.
`load_encoder` here fails loudly on any key mismatch instead.
"""

from __future__ import annotations

import argparse
import random
import tarfile
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from stern.data.wds_frames import load_shard_index, read_shard_frames_indexed
from stern.data.windows import window_starts
from stern.models.videomae.encoder import (
    VideoMAEEncoder,
    preprocess_window,
)


def sample_clip_keys(
    shard_path: Path,
    n_windows: int,
    seed: int,
    shard_index: dict | None = None,
    num_frames: int | None = None,
) -> list[str]:
    """Pick `n_windows` clip keys present in `shard_path`, without pulling the
    shard's frame data -- only the tar's member names are scanned.

    Real clips vary in length: when `shard_index`/`num_frames` are given,
    candidates shorter than `num_frames` are excluded up front, instead of
    letting a too-short clip get sampled and fail later at window selection.
    """
    with tarfile.open(shard_path, "r") as tar:
        names = tar.getnames()
    clip_keys = sorted(n[: -len(".pickle")] for n in names if n.endswith(".pickle"))
    if shard_index is not None and num_frames is not None:
        clip_keys = [
            k
            for k in clip_keys
            if shard_index.get(k, {}).get("num_frames", 0) >= num_frames
        ]
    if len(clip_keys) < n_windows:
        raise ValueError(
            f"{shard_path}: only {len(clip_keys)} eligible clips available, need {n_windows}"
        )
    return random.Random(seed).sample(clip_keys, n_windows)


def pick_window_indices(
    clip_key: str, shard_index: dict, num_frames: int, rng: random.Random
) -> list[int]:
    n_frames = shard_index[clip_key]["num_frames"]
    starts = window_starts(n_frames, num_frames, overlap=0.0)
    if not starts:
        raise ValueError(
            f"{clip_key}: {n_frames} frames too short for a {num_frames}-frame window"
        )
    start = rng.choice(starts)
    return list(range(start, start + num_frames))


def load_sample_windows(
    shard_path: Path, n_windows: int, num_frames: int, image_size: int, seed: int
) -> list[tuple[str, list[np.ndarray]]]:
    """Returns `[(clip_key, frames), ...]`, `frames` a list of `num_frames`
    HxWx3 uint8 arrays -- reads only the sampled clips' shard members, never
    the whole (~850MB) shard."""
    shard_index = load_shard_index(shard_path.parent)
    clip_keys = sample_clip_keys(
        shard_path, n_windows, seed, shard_index=shard_index, num_frames=num_frames
    )
    rng = random.Random(seed)
    windows = []
    for clip_key in clip_keys:
        indices = pick_window_indices(clip_key, shard_index, num_frames, rng)
        frames = read_shard_frames_indexed(
            shard_path, clip_key, indices, size=image_size
        )
        windows.append((clip_key, frames))
    return windows


def model_cfg_for_run(base_cfg, run_id: str):
    """`base_cfg` assumes one target_mode/decoder shape for every run_id, but
    recipe2x2-style comparisons mix per_frame (predict_head width 1536) vs
    time_avg (768) checkpoints, and legacy (decoder 256/8) vs paper (384/6)
    recipes, in the same `--run-ids` list. Clone + override per-run from the
    run_id string so `load_encoder`'s strict state_dict load matches shape,
    same convention `diagnose_conditioning.py` uses."""
    cfg = base_cfg.copy()
    if "per_frame" in run_id:
        cfg.target_mode = "per_frame"
    # e2b-* arms all launched from videomae_mesh_paper_a_stride1_timeavg.yaml
    # (EXPERIMENTS.md) -- the paper decoder shape, but the id doesn't say
    # "paper" since it names the pool (fish/birds/cars/countries), not the recipe.
    if "paper" in run_id or run_id.startswith("e2b-") or run_id.startswith("e2b2-"):
        cfg.decoder_embed_dim = 384
        cfg.decoder_num_heads = 6
    # e2b2-* arms additionally add RegisterTokens (Darcet et al. 2023) on top
    # of the e2b-* paper recipe.
    if run_id.startswith("e2b2-"):
        cfg.num_registers = 4
    return cfg


def load_encoder(run_id: str, mesh_shared: Path, model_cfg) -> VideoMAEEncoder:
    """Loads `run_id`'s merged checkpoint into a full `VideoMAEForPreTraining`
    and returns its `.encoder`. `run_id == "random_init"` skips loading
    entirely, returning a freshly-initialized encoder as a control.

    Raises on any missing/unexpected key -- a silent `strict=False` pass here
    would mean plotting a random encoder's internals under a trained run's
    label.
    """
    # Local import: _diag_common imports load_sample_windows from this module,
    # so a module-level import here would be circular.
    from stern.evaluation.diagnostics.common import load_bundle, random_init_bundle

    model_cfg = model_cfg_for_run(model_cfg, run_id)
    if run_id == "random_init":
        return random_init_bundle(model_cfg, full_model=True).encoder
    return load_bundle(run_id, mesh_shared, model_cfg, full_model=True).encoder


def run_encoder_with_internals(
    encoder: VideoMAEEncoder, pixel_values: torch.Tensor
) -> dict:
    """Mirrors `VideoMAEEncoder.forward` + `Block.forward`, but calls each
    block's `attn` submodule directly with `need_weights=True` to recover
    attention weights `encoder.py` normally discards (`need_weights=False`).

    Reduces before retaining: the raw `(num_heads, N, N)` attention-weight
    matrix (118MB/layer at ViT-B's 1568 tokens) is collapsed to a `(N,)`
    received-attention vector immediately and never kept whole.
    """
    with torch.no_grad():
        x = encoder.patch_embed(pixel_values)
        x = x + encoder.pos_embed
        x = encoder.registers.prepend(x)
        n_reg = encoder.registers.num_registers
        received_attn: list[np.ndarray] = []
        attn_out_norms: list[np.ndarray] = []
        hidden: list[np.ndarray] = []
        for block in encoder.blocks:
            h = block.norm1(x)
            attn_out, attn_weights = block.attn(h, need_weights=True)
            received_attn.append(
                attn_weights[0].mean(dim=(0, 1))[n_reg:].numpy().copy()
            )
            attn_out_norms.append(attn_out[0].norm(dim=-1)[n_reg:].numpy().copy())
            del attn_weights
            x = x + block.drop_path(block.layer_scale1(attn_out))
            x = x + block.drop_path(block.layer_scale2(block.mlp(block.norm2(x))))
            hidden.append(x[0][n_reg:].numpy().copy())
        x = encoder.registers.strip(x)
        final_tokens = encoder.norm(x)[0].numpy().copy()
    return {
        "received_attn": received_attn,  # list[depth] of (N,)
        "attn_out_norms": attn_out_norms,  # list[depth] of (N,)
        "hidden": hidden,  # list[depth] of (N, D)
        "final_tokens": final_tokens,  # (N, D)
        "token_grid": encoder.token_grid,  # (T', H', W')
    }


def received_attention_map(
    received_attn_last_layer: np.ndarray, token_grid: tuple[int, int, int]
) -> np.ndarray:
    """`(N,)` received-attention (last layer) -> `(H', W')`, averaged over the
    `T'` temporal tubelets."""
    t, h, w = token_grid
    return received_attn_last_layer.reshape(t, h, w).mean(axis=0)


def pca_rgb(tokens: np.ndarray, token_grid: tuple[int, int, int]) -> np.ndarray:
    """`(N, D)` patch tokens -> `(H', W', 3)` uint8 via a 3-component PCA fit
    fresh on these tokens (axes aren't comparable across differently-trained
    models, so no cross-checkpoint basis is shared).

    Component signs are fixed (largest-magnitude loading forced positive) so
    SVD's arbitrary sign convention doesn't read as a real difference between
    otherwise-similar checkpoints.

    Temporal reduction happens after projection, not before: PCA is linear,
    so `mean_t(X_t) @ C.T == mean_t(X_t @ C.T)` -- projecting first and
    averaging after gives the identical result as the reverse order.
    """
    t, h, w = token_grid
    centered = tokens - tokens.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:3].copy()
    for i in range(3):
        j = int(np.argmax(np.abs(components[i])))
        if components[i, j] < 0:
            components[i] *= -1
    proj = centered @ components.T  # (N, 3)
    grid = proj.reshape(t, h, w, 3).mean(axis=0)  # (H', W', 3)
    lo = grid.min(axis=(0, 1), keepdims=True)
    hi = grid.max(axis=(0, 1), keepdims=True)
    rgb = (grid - lo) / np.clip(hi - lo, 1e-8, None)
    return (rgb * 255).astype(np.uint8)


def consecutive_cosine_similarity(hidden_layers: list[np.ndarray]) -> np.ndarray:
    """`depth` layers of `(N, D)` hidden states -> `(depth - 1,)` mean
    per-token cosine similarity between consecutive layers (per-token, not
    mean-pooled-then-compared, so token-level drift isn't washed out)."""
    sims = []
    for a, b in zip(hidden_layers[:-1], hidden_layers[1:]):
        num = (a * b).sum(axis=1)
        denom = np.clip(
            np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-8, None
        )
        sims.append(float(np.mean(num / denom)))
    return np.array(sims, dtype=np.float32)


def make_log_bin_edges(all_norms: np.ndarray, n_bins: int = 30) -> np.ndarray:
    """Log-spaced bin edges spanning `all_norms`' full range -- linear bins
    would collapse to one dark column given ViTs' known heavy-tailed
    high-norm outlier tokens."""
    lo = max(float(np.min(all_norms)), 1e-6)
    hi = float(np.max(all_norms))
    if hi <= lo:
        hi = lo * 10.0
    return np.logspace(np.log10(lo), np.log10(hi), n_bins + 1).astype(np.float32)


def attention_norm_density(
    attn_out_norm_layers: list[np.ndarray], bin_edges: np.ndarray
) -> np.ndarray:
    """`depth` layers of `(N,)` per-token attention-branch-output norms ->
    `(depth, n_bins)` proportion-of-tokens-per-bin (rows sum to 1), binned
    against `bin_edges` shared across every checkpoint so panels stay
    comparable."""
    n_bins = len(bin_edges) - 1
    out = np.zeros((len(attn_out_norm_layers), n_bins), dtype=np.float32)
    for i, norms in enumerate(attn_out_norm_layers):
        counts, _ = np.histogram(norms, bins=bin_edges)
        total = counts.sum()
        out[i] = counts / total if total > 0 else counts
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-path", type=Path, required=True)
    parser.add_argument("--mesh-shared", type=Path, required=True)
    parser.add_argument(
        "--run-ids", required=True, help="comma-separated run_ids to load"
    )
    parser.add_argument("--n-windows", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-random-init", action="store_true")
    parser.add_argument("--n-bins", type=int, default=30)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/model/videomae_base.yaml"),
    )
    args = parser.parse_args()

    model_cfg = OmegaConf.load(args.model_config)
    num_frames = int(model_cfg.num_frames)
    image_size = int(model_cfg.image_size)

    windows = load_sample_windows(
        args.shard_path, args.n_windows, num_frames, image_size, args.seed
    )

    run_ids = [r.strip() for r in args.run_ids.split(",") if r.strip()]
    if args.include_random_init:
        run_ids = [*run_ids, "random_init"]

    per_run: dict[str, dict] = {}
    all_norms: list[float] = []
    for run_id in run_ids:
        try:
            encoder = load_encoder(run_id, args.mesh_shared, model_cfg)
        except FileNotFoundError:
            print(f"WARNING: {run_id}: no checkpoint found, skipping")
            continue
        received_maps, pca_maps, cos_sims, norm_layers_per_window = [], [], [], []
        for _clip_key, frames in windows:
            pixel_values = preprocess_window(frames, num_frames, image_size)
            internals = run_encoder_with_internals(encoder, pixel_values)
            received_maps.append(
                received_attention_map(
                    internals["received_attn"][-1], internals["token_grid"]
                )
            )
            pca_maps.append(pca_rgb(internals["final_tokens"], internals["token_grid"]))
            cos_sims.append(consecutive_cosine_similarity(internals["hidden"]))
            norm_layers_per_window.append(internals["attn_out_norms"])
            all_norms.extend(
                float(v) for layer in internals["attn_out_norms"] for v in layer
            )
        per_run[run_id] = {
            "received_maps": np.stack(received_maps),  # (W, H', W')
            "pca_maps": np.stack(pca_maps),  # (W, H', W', 3)
            "cos_sim": np.mean(np.stack(cos_sims), axis=0),  # (depth-1,)
            "norm_layers_per_window": norm_layers_per_window,
        }

    if not per_run:
        raise ValueError("no checkpoint loaded successfully for any run_id")

    bin_edges = make_log_bin_edges(np.array(all_norms), n_bins=args.n_bins)
    run_ids_ok = list(per_run.keys())

    save_dict = {
        "run_ids": np.array(run_ids_ok),
        "clip_keys": np.array([ck for ck, _ in windows]),
        "middle_frames": np.stack([frames[len(frames) // 2] for _, frames in windows]),
        "bin_edges": bin_edges,
        "received_maps": np.stack([per_run[r]["received_maps"] for r in run_ids_ok]),
        "pca_maps": np.stack([per_run[r]["pca_maps"] for r in run_ids_ok]),
        "cos_sim": np.stack([per_run[r]["cos_sim"] for r in run_ids_ok]),
        "norm_hist": np.stack(
            [
                np.mean(
                    np.stack(
                        [
                            attention_norm_density(layers, bin_edges)
                            for layers in per_run[r]["norm_layers_per_window"]
                        ]
                    ),
                    axis=0,
                )
                for r in run_ids_ok
            ]
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **save_dict)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f}MB)")


if __name__ == "__main__":
    main()
