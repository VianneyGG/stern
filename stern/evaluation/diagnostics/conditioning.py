"""One-shot conditioning / rank diagnostic over the recipe2x2 checkpoints.

Motivation: the "new recipe" (paper-faithful, stride-1, time_avg) fixed
positional collapse, but a downstream probe found layers 8-12 go dead anyway
("depth collapse", invisible in val_loss). Before running the 4-pool E2b
ablation to test candidate fixes, this script asks a cheaper question first:
does a direct rank/conditioning read on the checkpoints already on disk show
*where* collapse sets in, without waiting on new training runs?

For each of the four `recipe2x2-{legacy,paper}-{per_frame,time_avg}` master
checkpoints, per encoder block (1..depth) this reports:
  - weight singular value spectrum for qkv/proj/fc1/fc2 (condition number,
    spectral-entropy effective rank)
  - activation effective rank: SVD of the block's token output (register-
    stripped, post block-norm) on one fixed-seed random input batch,
    reshaped to (B*N, D)
  - gradient singular value spectrum for qkv.weight from one MAE-loss
    backward pass at this checkpoint -- a single snapshot, not a trajectory
    across training (no intermediate-round checkpoints were pulled for
    recipe2x2, only the final master.pt per cell)

Caveat: input is fixed-seed random noise, not real video data -- no
manifest/shard path was wired for this script. This isolates
architecture/weight-space rank collapse; a real-data pass could show
additional data-dependent collapse not visible here.

Run on a single GPU node (orchids tier is sufficient -- read-only inference
plus one backward pass, no training). Usage:
    PYTHONPATH=. uv run --no-sync python stern/evaluation/diagnostics/conditioning.py
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from omegaconf import OmegaConf

from stern.models.videomae.encoder import VideoMAEEncoder
from stern.models.videomae.masking import RandomTubeMasking
from stern.evaluation.diagnostics.common import load_bundle

CHECKPOINT_ROOT = Path.home() / "stern_mesh" / "checkpoints"
CHECKPOINTS = {
    "legacy-per_frame": CHECKPOINT_ROOT
    / "recipe2x2-legacy-per_frame-20260823"
    / "master.pt",
    "legacy-time_avg": CHECKPOINT_ROOT
    / "recipe2x2-legacy-time_avg-20260823"
    / "master.pt",
    "paper-per_frame": CHECKPOINT_ROOT
    / "recipe2x2-paper-per_frame-20260823"
    / "master.pt",
    "paper-time_avg": CHECKPOINT_ROOT
    / "recipe2x2-paper-time_avg-20260823"
    / "master.pt",
}

MODEL_CFG_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "model" / "videomae_base.yaml"
)
SEED = 0
BATCH_SIZE = 2


def effective_rank(svals: torch.Tensor) -> float:
    """Spectral-entropy effective rank (Roy & Vetterli 2007): exp(H(p)) where
    p is the singular values normalised to a probability distribution.
    Soft rank measure -- 1.0 means all energy in one direction, min(shape)
    means a perfectly flat spectrum."""
    s = svals.clamp_min(1e-12)
    p = s / s.sum()
    entropy = -(p * p.log()).sum()
    return math.exp(entropy.item())


def spectrum_stats(mat: torch.Tensor) -> dict:
    s = torch.linalg.svdvals(mat.float())
    return {
        "cond": (s[0] / s[-1].clamp_min(1e-12)).item(),
        "eff_rank": effective_rank(s),
        "shape_min": min(mat.shape),
    }


def load_encoder(device: torch.device) -> tuple[VideoMAEEncoder, dict]:
    model_cfg = OmegaConf.load(MODEL_CFG_PATH)
    encoder = VideoMAEEncoder(model_cfg).to(device)
    return encoder, model_cfg


def _run_id_and_shared(path: Path) -> tuple[str, Path]:
    """`path` is `CHECKPOINT_ROOT/<run_id>/master.pt`, and `CHECKPOINT_ROOT`
    itself already ends in `checkpoints` -- `load_bundle` appends
    `checkpoints/<run_id>/master.pt` itself, so `mesh_shared` is three
    `.parent`s up, not two."""
    return path.parent.name, path.parent.parent.parent


def analyze_checkpoint(name: str, path: Path, device: torch.device) -> None:
    print(f"\n=== {name} ({path}) ===")
    run_id, mesh_shared = _run_id_and_shared(path)
    model_cfg = OmegaConf.load(MODEL_CFG_PATH)
    # predict_head shape depends on target_mode (time_avg: 768, per_frame: 1536);
    # checkpoints were trained under both, so this must match the checkpoint,
    # not the base config's default, or mae_model.load_state_dict shape-mismatches.
    model_cfg.target_mode = "per_frame" if "per_frame" in name else "time_avg"
    # "paper" recipe overrides decoder_embed_dim=384/decoder_num_heads=6 (base
    # config default is 256/8); "legacy" checkpoints were trained at the base
    # default. Same story as target_mode -- must match the checkpoint's shape.
    if "paper" in name:
        model_cfg.decoder_embed_dim = 384
        model_cfg.decoder_num_heads = 6

    bundle = load_bundle(run_id, mesh_shared, model_cfg, full_model=True)
    encoder = bundle.encoder.to(device)
    mae_model = bundle.model.to(device)

    depth = int(model_cfg.depth)
    torch.manual_seed(SEED)
    grid = encoder.token_grid  # (T', H', W')
    pixel_values = torch.randn(
        BATCH_SIZE,
        int(model_cfg.in_channels),
        int(model_cfg.num_frames),
        int(model_cfg.image_size),
        int(model_cfg.image_size),
        device=device,
    )

    masker = RandomTubeMasking(mask_ratio=float(model_cfg.mask_ratio))
    keep_indices, mask_indices = masker(pixel_values, grid, device, BATCH_SIZE)

    # --- weight spectra + activation spectra (no grad needed) ---
    encoder.eval()
    with torch.no_grad():
        x = encoder.patch_embed(pixel_values) + encoder.pos_embed
        _, intermediates = encoder.forward_tokens(
            x, keep_indices, return_layers=list(range(1, depth + 1))
        )

    print(
        f"  {'blk':>3} | {'qkv_cond':>10} {'qkv_erank':>10} | "
        f"{'proj_cond':>10} {'proj_erank':>10} | "
        f"{'act_cond':>10} {'act_erank':>10}"
    )
    for i, block in enumerate(encoder.blocks, start=1):
        qkv = spectrum_stats(block.attn.qkv.weight)
        proj = spectrum_stats(block.attn.proj.weight)
        act_mat = intermediates[i - 1].reshape(-1, intermediates[i - 1].shape[-1])
        act = spectrum_stats(act_mat)
        print(
            f"  {i:>3} | {qkv['cond']:>10.1f} {qkv['eff_rank']:>10.1f} | "
            f"{proj['cond']:>10.1f} {proj['eff_rank']:>10.1f} | "
            f"{act['cond']:>10.1f} {act['eff_rank']:>10.1f}"
        )

    # --- gradient spectrum: one MAE-loss backward pass at this checkpoint ---
    mae_model.train()
    mae_model.zero_grad()
    pred, target = mae_model(
        pixel_values, keep_indices=keep_indices, mask_indices=mask_indices
    )
    loss = torch.mean((pred - target) ** 2)
    loss.backward()

    print(f"  {'blk':>3} | {'grad_cond':>10} {'grad_erank':>10}")
    for i, block in enumerate(mae_model.encoder.blocks, start=1):
        grad = block.attn.qkv.weight.grad
        if grad is None:
            print(
                f"  {i:>3} | {'None':>10} {'None':>10}   (no grad -- outside keep_indices path)"
            )
            continue
        g = spectrum_stats(grad)
        print(f"  {i:>3} | {g['cond']:>10.1f} {g['eff_rank']:>10.1f}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    for name, path in CHECKPOINTS.items():
        if not path.exists():
            print(f"\n=== {name} === MISSING: {path}")
            continue
        try:
            analyze_checkpoint(name, path, device)
        except Exception as exc:  # one bad checkpoint should not abort the rest
            print(f"  ERROR analyzing {name}: {exc}")


if __name__ == "__main__":
    main()
