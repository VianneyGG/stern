"""Write one probe run's val-set predictions for the gate aggregator.

A Hydra entrypoint taking the same overrides as `training.train`, so the val
split is reproduced exactly: `_split_loaders` seeds `random_split` from
`cfg.train.seed`, and holding that constant across conditions is what keeps the
bootstrap paired.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from video_encoding.data.loaders import build_probe_loader
from video_encoding.data.probe_windows import VAWindowDataset
from video_encoding.models import VideoMAEVARegressor
from video_encoding.scripts.probe_temporal_conditions import prediction_path
from video_encoding.training import checkpoint as ckpt_io
from video_encoding.training.train import _split_loaders, _temporal_transform


@torch.no_grad()
def collect(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    preds, targets = [], []
    for pixel_values, label in loader:
        preds.append(model(pixel_values.to(device)).float().cpu().numpy())
        targets.append(label.numpy())
    return np.concatenate(preds), np.concatenate(targets)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(cfg.train.seed))

    labelled = build_probe_loader(cfg.dataset, cfg.preprocess.window)
    dataset = VAWindowDataset(
        labelled,
        num_frames=int(cfg.model.num_frames),
        image_size=int(cfg.model.image_size),
        overlap=float(cfg.preprocess.window.overlap),
        va_range=tuple(cfg.dataset.va_range),
        transform=_temporal_transform(cfg),
    )
    _, val_loader = _split_loaders(
        dataset,
        val_frac=0.1,
        batch_size=int(cfg.train.batch_size),
        seed=int(cfg.train.seed),
        rank=0,
        world_size=1,
    )
    if val_loader is None:
        raise ValueError("no validation split: dataset too small to score")

    model = VideoMAEVARegressor(cfg.model).to(device)
    best = Path(cfg.train.checkpoint_root) / cfg.experiment.name / "best.pt"
    state = ckpt_io.load_checkpoint_file(best)
    model.load_state_dict(state.get("model_state_dict", state))
    model.eval()

    preds, targets = collect(model, val_loader, device)
    out_dir = Path(cfg.train.probe_pred_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = prediction_path(
        out_dir,
        str(cfg.train.probe_checkpoint_label),
        str(cfg.train.get("temporal_condition", "normal")),
    )
    # VA is 2-dimensional; the gate is computed on valence and arousal pooled,
    # matching how concordance_correlation_coefficient is applied elsewhere.
    np.savez(out, pred=preds.reshape(-1), target=targets.reshape(-1))
    print(f"wrote {out} ({len(preds)} windows)", flush=True)


if __name__ == "__main__":
    main()
