from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import patch

from omegaconf import OmegaConf

from video_encoding.training.tb_logger import (
    result_json_filename,
    write_mesh_round_scalars,
    write_result_json,
)


class _RecordingWriter:
    def __init__(self):
        self.calls: list[tuple[str, float, int]] = []

    def add_scalar(self, tag, value, step):
        self.calls.append((tag, value, step))


def _write_round_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_write_mesh_round_scalars_holds_back_the_newest_round(tmp_path):
    mesh_shared = tmp_path / "mesh_shared"
    _write_round_jsonl(
        mesh_shared / "metrics" / "run1" / "a.jsonl",
        [{"round": 1, "train_loss": 1.0, "val_loss": 2.0, "samp_s": 10.0}],
    )
    writer = _RecordingWriter()

    new_last_round = write_mesh_round_scalars(writer, mesh_shared, "run1", last_round=0)

    assert writer.calls == []
    assert new_last_round == 0


def test_write_mesh_round_scalars_flushes_once_a_newer_round_appears(tmp_path):
    mesh_shared = tmp_path / "mesh_shared"
    _write_round_jsonl(
        mesh_shared / "metrics" / "run1" / "a.jsonl",
        [
            {"round": 1, "train_loss": 1.0, "val_loss": 2.0, "samp_s": 10.0},
            {"round": 2, "train_loss": 5.0, "val_loss": 6.0, "samp_s": 11.0},
        ],
    )
    writer = _RecordingWriter()

    new_last_round = write_mesh_round_scalars(writer, mesh_shared, "run1", last_round=0)

    tags = {c[0]: c for c in writer.calls}
    assert tags["mesh_round/a/train_loss"] == ("mesh_round/a/train_loss", 1.0, 0)
    # round 2 held back -- no scalar at step 1 yet
    assert not any(c[2] == 1 for c in writer.calls)
    assert new_last_round == 1


def test_write_mesh_round_scalars_flushes_multiple_backlogged_rounds_in_order(tmp_path):
    mesh_shared = tmp_path / "mesh_shared"
    _write_round_jsonl(
        mesh_shared / "metrics" / "run1" / "a.jsonl",
        [
            {"round": 1, "train_loss": 1.0, "val_loss": 1.0, "samp_s": 1.0},
            {"round": 2, "train_loss": 2.0, "val_loss": 2.0, "samp_s": 2.0},
            {"round": 3, "train_loss": 3.0, "val_loss": 3.0, "samp_s": 3.0},
        ],
    )
    writer = _RecordingWriter()

    new_last_round = write_mesh_round_scalars(writer, mesh_shared, "run1", last_round=0)

    steps_written = sorted({c[2] for c in writer.calls})
    assert steps_written == [0, 1]  # rounds 1 and 2 -- round 3 held back
    assert new_last_round == 2


def test_write_mesh_round_scalars_skips_rounds_already_flushed(tmp_path):
    mesh_shared = tmp_path / "mesh_shared"
    _write_round_jsonl(
        mesh_shared / "metrics" / "run1" / "a.jsonl",
        [
            {"round": 1, "train_loss": 1.0, "val_loss": 1.0, "samp_s": 1.0},
            {"round": 2, "train_loss": 2.0, "val_loss": 2.0, "samp_s": 2.0},
            {"round": 3, "train_loss": 3.0, "val_loss": 3.0, "samp_s": 3.0},
        ],
    )
    writer = _RecordingWriter()

    new_last_round = write_mesh_round_scalars(writer, mesh_shared, "run1", last_round=2)

    # only round 3 is > last_round, and it's the newest -> held back
    assert writer.calls == []
    assert new_last_round == 2


def test_write_mesh_round_scalars_pool_mean_reflects_only_reporting_nodes(tmp_path):
    mesh_shared = tmp_path / "mesh_shared"
    _write_round_jsonl(
        mesh_shared / "metrics" / "run1" / "a.jsonl",
        [
            {"round": 1, "train_loss": 1.0, "val_loss": 1.0, "samp_s": 10.0},
            {"round": 2, "train_loss": 9.0, "val_loss": 9.0, "samp_s": 90.0},
        ],
    )
    _write_round_jsonl(
        mesh_shared / "metrics" / "run1" / "b.jsonl",
        [{"round": 2, "train_loss": 9.0, "val_loss": 9.0, "samp_s": 90.0}],
    )
    writer = _RecordingWriter()

    write_mesh_round_scalars(writer, mesh_shared, "run1", last_round=0)

    # round 1 (step 0) only has node a's record -- pool mean is a's value
    # alone, not blended with b (b has no round-1 record at all)
    tags = {c[0]: c[1] for c in writer.calls if c[2] == 0}
    assert tags["mesh_round/pool/train_loss_mean"] == 1.0
    assert tags["mesh_round/pool/samp_s_sum"] == 10.0


def test_write_mesh_round_scalars_tags_pool_with_first_reporting_nodes_job_id(
    tmp_path,
):
    mesh_shared = tmp_path / "mesh_shared"
    _write_round_jsonl(
        mesh_shared / "metrics" / "run1" / "b.jsonl",
        [
            {
                "round": 1,
                "train_loss": 1.0,
                "val_loss": 1.0,
                "samp_s": 1.0,
                "slurm_job_id": "222",
            },
            {"round": 2, "train_loss": 1.0, "val_loss": 1.0, "samp_s": 1.0},
        ],
    )
    _write_round_jsonl(
        mesh_shared / "metrics" / "run1" / "a.jsonl",
        [
            {
                "round": 1,
                "train_loss": 3.0,
                "val_loss": 3.0,
                "samp_s": 3.0,
                "slurm_job_id": "111",
            },
            {"round": 2, "train_loss": 3.0, "val_loss": 3.0, "samp_s": 3.0},
        ],
    )
    writer = _RecordingWriter()

    write_mesh_round_scalars(writer, mesh_shared, "run1", last_round=0)

    tags = {c[0]: c[1] for c in writer.calls if c[2] == 0}
    # nodes are visited sorted by name ("a" before "b") -- pool tag takes the
    # alphabetically-first reporting node's job id, not submission order
    assert tags["mesh_round/a/slurm_job_id"] == 111.0
    assert tags["mesh_round/b/slurm_job_id"] == 222.0
    assert tags["mesh_round/pool/slurm_job_id"] == 111.0


def test_write_mesh_round_scalars_omits_pool_job_id_when_none_reported(tmp_path):
    mesh_shared = tmp_path / "mesh_shared"
    _write_round_jsonl(
        mesh_shared / "metrics" / "run1" / "a.jsonl",
        [
            {"round": 1, "train_loss": 1.0, "val_loss": 1.0, "samp_s": 1.0},
            {"round": 2, "train_loss": 1.0, "val_loss": 1.0, "samp_s": 1.0},
        ],
    )
    writer = _RecordingWriter()

    write_mesh_round_scalars(writer, mesh_shared, "run1", last_round=0)

    tags = {c[0] for c in writer.calls if c[2] == 0}
    assert "mesh_round/pool/slurm_job_id" not in tags


def test_write_mesh_round_scalars_discovers_nodes_via_glob(tmp_path):
    mesh_shared = tmp_path / "mesh_shared"
    _write_round_jsonl(
        mesh_shared / "metrics" / "run1" / "node053.jsonl",
        [
            {"round": 1, "train_loss": 1.0, "val_loss": 1.0, "samp_s": 1.0},
            {"round": 2, "train_loss": 1.0, "val_loss": 1.0, "samp_s": 1.0},
        ],
    )
    _write_round_jsonl(
        mesh_shared / "metrics" / "run1" / "node055.jsonl",
        [
            {"round": 1, "train_loss": 3.0, "val_loss": 3.0, "samp_s": 3.0},
            {"round": 2, "train_loss": 3.0, "val_loss": 3.0, "samp_s": 3.0},
        ],
    )
    writer = _RecordingWriter()

    write_mesh_round_scalars(writer, mesh_shared, "run1", last_round=0)

    tags = {c[0] for c in writer.calls}
    assert "mesh_round/node053/train_loss" in tags
    assert "mesh_round/node055/train_loss" in tags


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
