"""Verify each experiment config self-selects its dataset/train overrides
via `# @package _global_` + `override /group: x`, instead of relying on
shell-level CLI flags (see scripts/run_experiment.sh)."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

pytestmark = pytest.mark.integration

CONF_DIR = Path(__file__).resolve().parents[1] / "conf"


def _compose(experiment: str):
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base=None):
        return compose(config_name="config", overrides=[f"experiment={experiment}"])


def test_ssl_pretrain_experiment_self_selects_dataset_and_train_group():
    cfg = _compose("videomae_ssl_pretrain_voxmini")
    assert cfg.dataset.name == "voxmini"
    assert cfg.experiment.stage == "pretrain"
    assert "base_lr" in cfg.train  # only conf/train/ssl.yaml has this field


def test_probe_experiment_self_selects_dataset_and_wires_pretrained_checkpoint():
    cfg = _compose("videomae_probe_veatic")
    assert cfg.dataset.name == "veatic"
    assert cfg.train.freeze_backbone is True
    assert cfg.train.pretrain_encoder_path.endswith(
        "videomae_ssl_pretrain_voxmini/best.pt"
    )


def test_finetune_experiments_default_to_no_freezing_and_select_their_dataset():
    cfg = _compose("videomae_va_finetune_afewva")
    assert cfg.dataset.name == "afewva"
    assert cfg.train.get("freeze_backbone", False) is False
    assert cfg.train.get("pretrain_encoder_path", None) is None

    cfg = _compose("videomae_va_finetune_veatic")
    assert cfg.dataset.name == "veatic"
