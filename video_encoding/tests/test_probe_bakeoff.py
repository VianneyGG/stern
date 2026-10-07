"""Tests for `scripts/probe_bakeoff.py`, the AFEW-VA frozen-probe CLI.

The science this CLI reports is verified by the cluster run it launches; what
is tested here is the two things that silently corrupt such a run:
a model config derived from the wrong source, and a label permutation that
isn't one.
"""

from __future__ import annotations

import json

import pytest
import torch
from omegaconf import OmegaConf

from video_encoding.scripts import probe_bakeoff
from video_encoding.scripts._diag_common import MetricValue


def _base_cfg():
    """A stand-in for `conf/model/videomae_base.yaml`: ViT-B geometry shrunk so
    the CLI tests can actually build an encoder from it, carrying every key
    those tests override. `model_cfg_for_run` rejects any `model.*` override
    absent from the base, so the base is what defines the allowlist."""
    return OmegaConf.create(
        {
            "num_frames": 4,
            "image_size": 32,
            "in_channels": 3,
            "tubelet_size": 2,
            "patch_size": 16,
            "embed_dim": 16,
            "depth": 2,
            "num_heads": 2,
            "mlp_ratio": 2.0,
            "dropout": 0.0,
            "drop_path_rate": 0.0,
            "target_mode": "time_avg",
            "decoder_embed_dim": 256,
            "decoder_num_heads": 8,
            "num_registers": 0,
        }
    )


def _write_experiment(tmp_path, name: str, body: str):
    experiment_dir = tmp_path / "experiment"
    experiment_dir.mkdir(exist_ok=True)
    (experiment_dir / f"{name}.yaml").write_text(body)
    return experiment_dir


def test_model_cfg_for_run_applies_the_experiment_model_block(tmp_path):
    experiment_dir = _write_experiment(
        tmp_path,
        "paper_b",
        "model:\n  target_mode: per_frame\n  decoder_embed_dim: 384\n",
    )
    cfg = probe_bakeoff.model_cfg_for_run(
        _base_cfg(), {"experiment": "paper_b"}, experiment_dir
    )
    assert cfg.target_mode == "per_frame"
    assert cfg.decoder_embed_dim == 384
    # untouched keys keep the base value
    assert cfg.decoder_num_heads == 8


def test_model_cfg_for_run_extra_override_beats_the_experiment_block(tmp_path):
    experiment_dir = _write_experiment(
        tmp_path, "paper_b", "model:\n  target_mode: per_frame\n"
    )
    cfg = probe_bakeoff.model_cfg_for_run(
        _base_cfg(),
        {"experiment": "paper_b", "extra_overrides": ["model.target_mode=time_avg"]},
        experiment_dir,
    )
    assert cfg.target_mode == "time_avg"


def test_model_cfg_for_run_applies_depth_override_as_an_int(tmp_path):
    """`depth=3` arriving as the string "3" would build a 3-character-deep
    nothing; the encoder's `int(cfg.depth)` hides the type, `len(blocks)`
    comparisons downstream do not."""
    experiment_dir = _write_experiment(tmp_path, "solo", "mesh:\n  rounds: 20\n")
    cfg = probe_bakeoff.model_cfg_for_run(
        _base_cfg(),
        {"experiment": "solo", "extra_overrides": ["model.depth=3"]},
        experiment_dir,
    )
    assert cfg.depth == 3
    assert isinstance(cfg.depth, int)


def test_model_cfg_for_run_rejects_a_model_key_absent_from_the_base(tmp_path):
    """The `layernorm_scaling` trap: a knob that adds no parameters loads
    against a mismatched config without raising, so the wrong architecture is
    reported under the run's name. Fail closed on the override instead."""
    experiment_dir = _write_experiment(tmp_path, "solo", "mesh:\n  rounds: 20\n")
    with pytest.raises(ValueError, match="layernorm_scaling"):
        probe_bakeoff.model_cfg_for_run(
            _base_cfg(),
            {"experiment": "solo", "extra_overrides": ["model.layernorm_scaling=true"]},
            experiment_dir,
        )


def test_model_cfg_for_run_rejects_an_experiment_model_key_absent_from_the_base(
    tmp_path,
):
    experiment_dir = _write_experiment(
        tmp_path, "stale", "model:\n  layernorm_scaling: true\n"
    )
    with pytest.raises(ValueError, match="layernorm_scaling"):
        probe_bakeoff.model_cfg_for_run(
            _base_cfg(), {"experiment": "stale"}, experiment_dir
        )


def test_model_cfg_for_run_ignores_non_model_overrides(tmp_path):
    experiment_dir = _write_experiment(tmp_path, "solo", "mesh:\n  rounds: 20\n")
    cfg = probe_bakeoff.model_cfg_for_run(
        _base_cfg(),
        {
            "experiment": "solo",
            "extra_overrides": ["train.lr_mult=2.0", "mesh.round_budget_s=null"],
        },
        experiment_dir,
    )
    assert "lr_mult" not in cfg
    assert cfg.target_mode == "time_avg"


def test_model_cfg_for_run_raises_on_a_missing_experiment_file(tmp_path):
    """A run whose experiment yaml is gone cannot have its geometry recovered
    from the run_id string, which is what the old `model_cfg_for_run` did --
    guessing there is how a per_frame checkpoint gets read as time_avg."""
    experiment_dir = _write_experiment(tmp_path, "present", "model: {}\n")
    with pytest.raises(FileNotFoundError, match="absent"):
        probe_bakeoff.model_cfg_for_run(
            _base_cfg(), {"experiment": "absent"}, experiment_dir
        )


def test_model_cfg_for_run_raises_when_the_entry_names_no_experiment(tmp_path):
    experiment_dir = _write_experiment(tmp_path, "present", "model: {}\n")
    with pytest.raises(ValueError, match="experiment"):
        probe_bakeoff.model_cfg_for_run(_base_cfg(), {}, experiment_dir)


# --- the label-permutation negative control ---------------------------------


def test_permuted_labels_are_a_permutation_of_the_originals():
    """Bound to the exact rows `torch.randperm` produces for this seed
    ([2, 5, 3, 0, 1, 4]), so a change of generator, of dimension permuted, or
    of the seed's derivation shows up here rather than in a CCC that merely
    looks low."""
    labels = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    permuted = probe_bakeoff.permute_labels(labels, seed=0)
    assert permuted.tolist() == [
        [4.0, 5.0],
        [10.0, 11.0],
        [6.0, 7.0],
        [0.0, 1.0],
        [2.0, 3.0],
        [8.0, 9.0],
    ]


def test_permuted_labels_keep_each_row_intact():
    """Valence and arousal must travel together: permuting within a row would
    destroy the label's own correlation structure as well as its pairing with
    the tokens, so a near-zero CCC would no longer isolate the pairing."""
    labels = torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])
    permuted = probe_bakeoff.permute_labels(labels, seed=1)
    for row in permuted:
        assert row[1] - row[0] == pytest.approx(1.0)


def test_permuted_labels_are_seed_determined():
    labels = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    assert torch.equal(
        probe_bakeoff.permute_labels(labels, seed=3),
        probe_bakeoff.permute_labels(labels, seed=3),
    )
    assert not torch.equal(
        probe_bakeoff.permute_labels(labels, seed=3),
        probe_bakeoff.permute_labels(labels, seed=4),
    )


def test_permuted_labels_leave_the_global_rng_untouched():
    """`probe_ccc` goes to some length to keep the caller's global RNG intact
    (`torch.random.fork_rng`); a permutation drawn from the global RNG would
    undo that and make every downstream probe fit depend on whether the
    negative control was requested."""
    torch.manual_seed(0)
    expected = torch.rand(1).item()
    torch.manual_seed(0)
    probe_bakeoff.permute_labels(torch.arange(8, dtype=torch.float32).reshape(4, 2), 0)
    assert torch.rand(1).item() == pytest.approx(expected)


# --- CLI wiring -------------------------------------------------------------


def _stub_probe_by_depth(recorder):
    def _stub(bundle, control_bundle, manifest_path, layers, **kwargs):
        recorder.update(
            bundle=bundle,
            control_bundle=control_bundle,
            manifest_path=manifest_path,
            layers=layers,
            kwargs=kwargs,
        )
        per_layer = {
            layer: (
                MetricValue(
                    mean=0.5, sd=0.01, n=3, seeds=(0, 1, 2), deterministic=False
                ),
                MetricValue(
                    mean=0.4, sd=0.01, n=3, seeds=(1, 2, 3), deterministic=False
                ),
            )
            for layer in layers
        }
        final = MetricValue(
            mean=0.3, sd=0.01, n=3, seeds=(0, 1, 2), deterministic=False
        )
        return per_layer, final

    return _stub


def _tiny_base_cfg_file(tmp_path):
    path = tmp_path / "videomae_base.yaml"
    path.write_text(OmegaConf.to_yaml(_base_cfg()))
    return path


def test_main_writes_a_report_with_both_arms_per_layer(tmp_path, monkeypatch):
    recorder: dict = {}
    monkeypatch.setattr(probe_bakeoff, "probe_by_depth", _stub_probe_by_depth(recorder))
    experiment_dir = _write_experiment(tmp_path, "solo", "model: {}\n")
    configs = tmp_path / "mesh_run_configs.json"
    configs.write_text(json.dumps([{"run_id": "random_init", "experiment": "solo"}]))
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("")
    out = tmp_path / "report.json"

    exit_code = probe_bakeoff.main(
        [
            "--run-id",
            "random_init",
            "--manifest",
            str(manifest),
            "--mesh-shared",
            str(tmp_path),
            "--configs",
            str(configs),
            "--model-cfg",
            str(_tiny_base_cfg_file(tmp_path)),
            "--experiment-dir",
            str(experiment_dir),
            "--layers",
            "1,2",
            "--out",
            str(out),
        ]
    )

    assert exit_code == 0
    payload = json.loads(out.read_text())
    assert payload["metrics"]["probe_ccc_L1"]["mean"] == pytest.approx(0.5)
    assert payload["metrics"]["probe_ccc_L1_control"]["mean"] == pytest.approx(0.4)
    assert payload["metrics"]["probe_ccc_L2"]["mean"] == pytest.approx(0.5)
    assert payload["metrics"]["probe_ccc_final_block"]["mean"] == pytest.approx(0.3)
    assert payload["meta"]["run_id"] == "random_init"
    assert payload["meta"]["layers"] == [1, 2]
    assert payload["meta"]["permute_labels_seed"] is None


def test_main_random_init_run_id_touches_no_checkpoint(tmp_path, monkeypatch):
    """`random_init` is the control arm and has no checkpoint directory; the
    CLI must build it from the config rather than looking for master.pt."""
    recorder: dict = {}
    monkeypatch.setattr(probe_bakeoff, "probe_by_depth", _stub_probe_by_depth(recorder))

    def _fail(*args, **kwargs):
        raise AssertionError("load_bundle must not be called for random_init")

    monkeypatch.setattr(probe_bakeoff, "load_bundle", _fail)
    experiment_dir = _write_experiment(tmp_path, "solo", "model: {}\n")
    configs = tmp_path / "mesh_run_configs.json"
    configs.write_text(json.dumps([{"run_id": "random_init", "experiment": "solo"}]))
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("")

    probe_bakeoff.main(
        [
            "--run-id",
            "random_init",
            "--manifest",
            str(manifest),
            "--mesh-shared",
            str(tmp_path),
            "--configs",
            str(configs),
            "--model-cfg",
            str(_tiny_base_cfg_file(tmp_path)),
            "--experiment-dir",
            str(experiment_dir),
            "--layers",
            "1",
            "--out",
            str(tmp_path / "report.json"),
        ]
    )
    assert recorder["bundle"].run_id == "random_init"


def test_main_passes_the_permute_seed_through(tmp_path, monkeypatch):
    recorder: dict = {}
    monkeypatch.setattr(probe_bakeoff, "probe_by_depth", _stub_probe_by_depth(recorder))
    experiment_dir = _write_experiment(tmp_path, "solo", "model: {}\n")
    configs = tmp_path / "mesh_run_configs.json"
    configs.write_text(json.dumps([{"run_id": "random_init", "experiment": "solo"}]))
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("")
    out = tmp_path / "report.json"

    probe_bakeoff.main(
        [
            "--run-id",
            "random_init",
            "--manifest",
            str(manifest),
            "--mesh-shared",
            str(tmp_path),
            "--configs",
            str(configs),
            "--model-cfg",
            str(_tiny_base_cfg_file(tmp_path)),
            "--experiment-dir",
            str(experiment_dir),
            "--layers",
            "1",
            "--permute-labels-seed",
            "7",
            "--out",
            str(out),
        ]
    )
    assert recorder["kwargs"]["permute_labels_seed"] == 7
    assert json.loads(out.read_text())["meta"]["permute_labels_seed"] == 7


def test_main_records_the_resolved_model_cfg_in_the_report(tmp_path, monkeypatch):
    """The report has to say which geometry was loaded: two arms probed under
    different depths are not comparable, and the run_id does not carry it."""
    monkeypatch.setattr(probe_bakeoff, "probe_by_depth", _stub_probe_by_depth({}))
    experiment_dir = _write_experiment(
        tmp_path, "solo", "model:\n  target_mode: per_frame\n"
    )
    configs = tmp_path / "mesh_run_configs.json"
    configs.write_text(json.dumps([{"run_id": "random_init", "experiment": "solo"}]))
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("")
    out = tmp_path / "report.json"

    probe_bakeoff.main(
        [
            "--run-id",
            "random_init",
            "--manifest",
            str(manifest),
            "--mesh-shared",
            str(tmp_path),
            "--configs",
            str(configs),
            "--model-cfg",
            str(_tiny_base_cfg_file(tmp_path)),
            "--experiment-dir",
            str(experiment_dir),
            "--layers",
            "1",
            "--out",
            str(out),
        ]
    )
    meta = json.loads(out.read_text())["meta"]
    assert meta["model_cfg"]["target_mode"] == "per_frame"
    assert meta["model_cfg"]["depth"] == _base_cfg().depth


# --- single-arm mode ----------------------------------------------------------


def _single_arm_argv(tmp_path, arm: str, *extra: str) -> list[str]:
    experiment_dir = _write_experiment(tmp_path, "solo", "model: {}\n")
    configs = tmp_path / "mesh_run_configs.json"
    configs.write_text(json.dumps([{"run_id": "lev-run", "experiment": "solo"}]))
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("")
    return [
        "--run-id",
        "lev-run",
        "--manifest",
        str(manifest),
        "--mesh-shared",
        str(tmp_path),
        "--configs",
        str(configs),
        "--model-cfg",
        str(_tiny_base_cfg_file(tmp_path)),
        "--experiment-dir",
        str(experiment_dir),
        "--layers",
        "1,2",
        "--arm",
        arm,
        "--out",
        str(tmp_path / "report.json"),
        *extra,
    ]


def _record_probe_ccc(monkeypatch):
    calls: list[dict] = []

    def _fake(bundle, manifest_path, *, layer, seed, **kwargs):
        calls.append({"bundle": bundle, "layer": layer, "seed": seed})
        return 0.1 * layer + (seed % 7) * 1e-3

    monkeypatch.setattr(probe_bakeoff, "probe_ccc", _fake)
    monkeypatch.setattr(
        probe_bakeoff,
        "probe_by_depth",
        lambda *a, **k: pytest.fail("single-arm mode must not run the paired sweep"),
    )
    return calls


def _replicate_seeds(base: int, n: int = 3) -> list[int]:
    import random

    rng = random.Random(base)
    return [rng.randrange(2**32) for _ in range(n)]


def test_control_arm_never_loads_the_checkpoint_and_writes_only_control_keys(
    tmp_path, monkeypatch
):
    calls = _record_probe_ccc(monkeypatch)
    monkeypatch.setattr(
        probe_bakeoff,
        "load_bundle",
        lambda *a, **k: pytest.fail("the control arm must not read master.pt"),
    )

    assert probe_bakeoff.main(_single_arm_argv(tmp_path, "control")) == 0

    payload = json.loads((tmp_path / "report.json").read_text())
    assert set(payload["metrics"]) == {"probe_ccc_L1_control", "probe_ccc_L2_control"}
    assert payload["meta"]["arm"] == "control"
    assert payload["meta"]["run_id"] == "lev-run"
    # The paired sweep's control seed base (1), so a split run reproduces it.
    assert [c["seed"] for c in calls if c["layer"] == 1] == _replicate_seeds(1)
    assert sorted({c["layer"] for c in calls}) == [1, 2]


def test_checkpoint_arm_writes_only_checkpoint_keys_on_the_loaded_bundle(
    tmp_path, monkeypatch
):
    calls = _record_probe_ccc(monkeypatch)
    loaded = probe_bakeoff.random_init_bundle(_base_cfg())
    monkeypatch.setattr(probe_bakeoff, "load_bundle", lambda *a, **k: loaded)

    probe_bakeoff.main(_single_arm_argv(tmp_path, "checkpoint"))

    payload = json.loads((tmp_path / "report.json").read_text())
    assert set(payload["metrics"]) == {"probe_ccc_L1", "probe_ccc_L2"}
    assert payload["metrics"]["probe_ccc_L2"]["n"] == 3
    assert payload["meta"]["arm"] == "checkpoint"
    assert all(c["bundle"] is loaded for c in calls)
    assert [c["seed"] for c in calls if c["layer"] == 2] == _replicate_seeds(0)


def test_control_encoder_is_identical_across_invocations():
    """Split jobs each build their own control; they must all be the same
    random encoder, or layer L and layer L' are scored against different
    baselines."""
    _, first = probe_bakeoff._arm_bundles(
        "x", "/nonexistent", _base_cfg(), round=None, load=False
    )
    torch.randn(5)  # disturb the global RNG between the two builds
    _, second = probe_bakeoff._arm_bundles(
        "x", "/nonexistent", _base_cfg(), round=None, load=False
    )

    a, b = first.encoder.state_dict(), second.encoder.state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_single_arm_refuses_a_label_permutation(tmp_path, monkeypatch):
    _record_probe_ccc(monkeypatch)
    with pytest.raises(SystemExit, match="permute"):
        probe_bakeoff.main(
            _single_arm_argv(tmp_path, "control", "--permute-labels-seed", "3")
        )
