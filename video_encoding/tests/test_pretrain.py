"""Unit tests for run_pretrain_stage's dataset injection point (the
molab-streaming-pretrain change): passing dataset= must bypass the internal
SSLWindowDataset/build_ssl_loader construction (and its local manifest
read) entirely, while dataset=None (every pre-existing caller) is
unchanged."""

from pathlib import Path
from unittest.mock import patch

import torch
from hydra import compose, initialize_config_dir
from torch.utils.data import TensorDataset

from video_encoding.training.pretrain import run_pretrain_stage

CONF_DIR = Path(__file__).resolve().parents[1] / "conf"


def _compose_cfg(tmp_path: Path):
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=videomae_ssl_pretrain_voxmini",
                "train.max_steps=1",
                "train.epochs=1",
                "train.num_workers=0",
                "train.batch_size=1",
                f"train.checkpoint_root={tmp_path}",
            ],
        )
    return cfg


def test_injected_dataset_bypasses_internal_construction(tmp_path):
    cfg = _compose_cfg(tmp_path)
    clip = torch.randn(
        3,
        int(cfg.model.num_frames),
        int(cfg.model.image_size),
        int(cfg.model.image_size),
    )

    class _Wrapped(TensorDataset):
        def __getitem__(self, idx):
            return super().__getitem__(idx)[0]

    dataset = _Wrapped(torch.stack([clip, clip]))

    with (
        patch("video_encoding.training.pretrain.SSLWindowDataset") as mock_ssl_dataset,
        patch("video_encoding.training.pretrain.build_ssl_loader") as mock_build_loader,
    ):
        run_pretrain_stage(
            cfg,
            rank=0,
            world_size=1,
            local_rank=0,
            device=torch.device("cpu"),
            dataset=dataset,
        )

    mock_ssl_dataset.assert_not_called()
    mock_build_loader.assert_not_called()
