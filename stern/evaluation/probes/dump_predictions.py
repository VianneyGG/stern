"""Write one probe run's validation predictions for the temporal-conditions gate.

A Hydra entrypoint that takes the same overrides as the training entrypoint, so
the val split is reproduced exactly. `_split_loaders` seeds its `random_split`
from `cfg.train.seed`. Keeping that seed fixed across conditions keeps the
bootstrap paired.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from stern.data.loaders import build_probe_loader
from stern.data.probe_windows import VAWindowDataset
from stern.models import VideoMAEVARegressor
from stern.evaluation.probes.temporal_conditions import prediction_path
from stern.training import checkpoint as ckpt_io
from stern.training.finetune import _split_loaders, _temporal_transform


@torch.no_grad()
def collect(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    preds, targets = [], []
    for pixel_values, label in loader:
        preds.append(model(pixel_values.to(device)).float().cpu().numpy())
        targets.append(label.numpy())
    return np.concatenate(preds), np.concatenate(targets)


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
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
    # Valence and arousal are pooled into one flat array for the gate's CCC.
    np.savez(out, pred=preds.reshape(-1), target=targets.reshape(-1))
    print(f"wrote {out} ({len(preds)} windows)", flush=True)


if __name__ == "__main__":
    main()
