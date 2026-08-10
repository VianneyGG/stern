import json

import pytest

from video_encoding.scripts.compare_sweep_cells import mean_val_loss_at_round


def _write(metrics_dir, run_id, node, records):
    path = metrics_dir / run_id / f"{node}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        for round_num, val_loss in records:
            fh.write(json.dumps({"round": round_num, "val_loss": val_loss}) + "\n")


def test_mean_val_loss_at_round_averages_across_nodes(tmp_path):
    _write(tmp_path, "run_a", "n00", [(30, 0.50)])
    _write(tmp_path, "run_a", "n01", [(30, 0.52)])
    assert mean_val_loss_at_round(tmp_path, "run_a", 30) == pytest.approx(0.51)


def test_mean_val_loss_at_round_ignores_other_rounds_and_runs(tmp_path):
    _write(tmp_path, "run_a", "n00", [(29, 0.90), (30, 0.50), (31, 0.10)])
    _write(tmp_path, "run_b", "n00", [(30, 999.0)])
    assert mean_val_loss_at_round(tmp_path, "run_a", 30) == pytest.approx(0.50)


def test_mean_val_loss_at_round_raises_when_round_never_reported(tmp_path):
    _write(tmp_path, "run_a", "n00", [(29, 0.90)])
    with pytest.raises(ValueError, match="run_a"):
        mean_val_loss_at_round(tmp_path, "run_a", 30)
