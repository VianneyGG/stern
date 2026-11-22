from stern.evaluation.throughput_audit.compute_config import (
    check_quantize_call_sites,
    read_compile_config,
    read_pacing_config,
)


def test_read_pacing_config_disabled():
    cfg_text = "mesh:\n  round_budget_s: null\n"
    assert read_pacing_config(cfg_text) == {
        "round_budget_s": None,
        "pacing_enabled": False,
    }


def test_read_pacing_config_enabled():
    cfg_text = "mesh:\n  round_budget_s: 240.0\n"
    assert read_pacing_config(cfg_text) == {
        "round_budget_s": 240.0,
        "pacing_enabled": True,
    }


def test_read_compile_config_off():
    cfg_text = "train:\n  compile: false\n"
    assert read_compile_config(cfg_text) == {"compile": False}


def test_check_quantize_call_sites_both_present(tmp_path):
    mesh_dir = tmp_path / "stern" / "mesh"
    training_dir = tmp_path / "stern" / "training"
    mesh_dir.mkdir(parents=True)
    training_dir.mkdir(parents=True)
    (mesh_dir / "run_outer_step.py").write_text(
        "from stern.mesh.quantize import quantize_with_error_feedback\n"
        "quantize_with_error_feedback(master_sd, residual_path)\n"
    )
    (training_dir / "mesh_round.py").write_text(
        "quantize_with_error_feedback(delta_sd, residual_path)\n"
    )

    result = check_quantize_call_sites(tmp_path)

    assert result == {"master_broadcast": True, "worker_push": True}


def test_check_quantize_call_sites_missing_worker_side(tmp_path):
    mesh_dir = tmp_path / "stern" / "mesh"
    training_dir = tmp_path / "stern" / "training"
    mesh_dir.mkdir(parents=True)
    training_dir.mkdir(parents=True)
    (mesh_dir / "run_outer_step.py").write_text(
        "quantize_with_error_feedback(master_sd, residual_path)\n"
    )
    (training_dir / "mesh_round.py").write_text("# no quantize call here\n")

    result = check_quantize_call_sites(tmp_path)

    assert result == {"master_broadcast": True, "worker_push": False}
