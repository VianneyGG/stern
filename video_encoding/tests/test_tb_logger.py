from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import patch

from omegaconf import OmegaConf

from video_encoding.training.tb_logger import result_json_filename, write_result_json


def test_result_json_filename_with_run_tag():
    assert result_json_filename("videomae_ssl_pretrain_voxmini", "p2-eb256") == (
        "videomae_ssl_pretrain_voxmini-p2-eb256.json"
    )


def test_result_json_filename_without_run_tag():
    assert result_json_filename("test_va_finetune", None) == "test_va_finetune.json"


def _cfg(run_tag=None):
    return OmegaConf.create({"experiment": {"name": "test_exp"}, "run_tag": run_tag})


def test_write_result_json_overwrite_reuses_stable_filename(tmp_path):
    out1 = write_result_json(
        {"loss": 1.0}, _cfg("tagA"), results_root=tmp_path, overwrite=True
    )
    out2 = write_result_json(
        {"loss": 0.5}, _cfg("tagA"), results_root=tmp_path, overwrite=True
    )
    assert out1 == out2
    assert len(list(tmp_path.glob("*.json"))) == 1
    assert json.loads(out2.read_text())["metrics"] == {"loss": 0.5}


def test_write_result_json_default_is_timestamped_and_not_overwritten(tmp_path):
    # Timestamp has second granularity -- pin two distinct instants so this
    # doesn't flake when both calls land in the same wall-clock second.
    with patch("video_encoding.training.tb_logger.datetime") as mock_dt:
        mock_dt.now.side_effect = [
            datetime(2026, 7, 15, 13, 0, 0),
            datetime(2026, 7, 15, 13, 0, 1),
        ]
        out1 = write_result_json({"loss": 1.0}, _cfg("tagB"), results_root=tmp_path)
        out2 = write_result_json({"loss": 0.5}, _cfg("tagB"), results_root=tmp_path)
    assert out1 != out2
    assert len(list(tmp_path.glob("*.json"))) == 2
