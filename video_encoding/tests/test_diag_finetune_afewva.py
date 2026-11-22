"""AFEW-VA fine-tune acceptance gate (ssl-diag-08)."""

from __future__ import annotations

import inspect
import json

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from video_encoding.models.videomaev2.encoder import VideoMAEEncoder
from video_encoding.scripts import diag_finetune_afewva as gate_mod
from video_encoding.training import train as train_mod


def _tiny_model_cfg(**overrides):
    cfg = {
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
    }
    cfg.update(overrides)
    return OmegaConf.create(cfg)


def _encoder(seed: int = 0) -> VideoMAEEncoder:
    torch.manual_seed(seed)
    return VideoMAEEncoder(_tiny_model_cfg())


# --- finetune_ccc ------------------------------------------------------------


def _spy_train_stage(monkeypatch, returned: dict) -> list:
    """Replace run_train_stage; record (cfg, loaded encoder file) per call."""
    seen: list = []

    def _spy(cfg, *, rank, world_size, local_rank, device):
        loaded = torch.load(cfg.train.pretrain_encoder_path, weights_only=False)
        seen.append((cfg, loaded, (rank, world_size, local_rank)))
        return returned

    monkeypatch.setattr(train_mod, "run_train_stage", _spy)
    return seen


_RETURNED = {
    "stage": "train",
    "val_ccc_v": 0.9,
    "val_ccc_a": 0.7,
    "mean_ccc": 0.8,
    "final": {"val_ccc_v": 0.3, "val_ccc_a": 0.1, "mean_ccc": 0.2},
    "epochs_run": 2,
}


def test_finetune_ccc_composes_the_gate_recipe(tmp_path, monkeypatch):
    seen = _spy_train_stage(monkeypatch, _RETURNED)
    encoder = _encoder()
    model_cfg = _tiny_model_cfg()

    gate_mod.finetune_ccc(
        encoder, model_cfg, manifest_root=tmp_path, min_epochs=2, seed=5
    )

    ((cfg, loaded, ranks),) = seen
    assert cfg.train.freeze_backbone is False
    assert cfg.train.resume is False
    assert cfg.train.save_latest is False
    assert cfg.train.split == "identity"
    assert cfg.train.split_fraction == gate_mod.SPLIT_FRACTION
    assert cfg.train.epochs == 2
    assert cfg.train.seed == 5
    assert cfg.experiment.stage == "va_finetune"
    assert cfg.train.learning_rate == gate_mod.GATE_LR
    assert cfg.model.head_bounded is False
    expected_model = {**OmegaConf.to_container(model_cfg), "head_bounded": False}
    assert OmegaConf.to_container(cfg.model) == expected_model
    assert "head_bounded" not in model_cfg
    assert cfg.dataset.manifest_path == str(tmp_path / "manifest.csv")
    assert cfg.dataset.path == str(tmp_path)
    assert ranks == (0, 1, 0)
    state = encoder.state_dict()
    assert set(loaded["model_state_dict"]) == {f"encoder.{k}" for k in state}
    for k, v in state.items():
        assert torch.equal(loaded["model_state_dict"][f"encoder.{k}"], v)


class _BackboneAndHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(4, 4)
        self.head = torch.nn.Linear(4, 2)


def _lrs_from_composed_train_cfg(cfg) -> dict[str, float]:
    """The LRs the real optimizer builder assigns under the composed recipe."""
    model = _BackboneAndHead()
    opt, _ = train_mod._build_optimizer(model, cfg.train)
    lr_of = {id(p): g["lr"] for g in opt.param_groups for p in g["params"]}
    return {
        "backbone": lr_of[id(model.backbone.weight)],
        "head": lr_of[id(model.head.weight)],
    }


def test_finetune_ccc_head_lr_mult_reaches_the_optimizer(tmp_path, monkeypatch):
    seen = _spy_train_stage(monkeypatch, _RETURNED)
    gate_mod.finetune_ccc(
        _encoder(), _tiny_model_cfg(), manifest_root=tmp_path, min_epochs=2, seed=0,
        head_lr_mult=100.0,
    )
    ((cfg, _, _),) = seen
    lrs = _lrs_from_composed_train_cfg(cfg)
    assert lrs["backbone"] == pytest.approx(gate_mod.GATE_LR)
    assert lrs["head"] == pytest.approx(100.0 * gate_mod.GATE_LR)
    assert float(cfg.train.weight_decay) == 0.0


def test_finetune_ccc_default_keeps_one_lr_for_every_param(tmp_path, monkeypatch):
    seen = _spy_train_stage(monkeypatch, _RETURNED)
    gate_mod.finetune_ccc(
        _encoder(), _tiny_model_cfg(), manifest_root=tmp_path, min_epochs=2, seed=0
    )
    ((cfg, _, _),) = seen
    lrs = _lrs_from_composed_train_cfg(cfg)
    assert lrs["backbone"] == lrs["head"] == pytest.approx(gate_mod.GATE_LR)


def test_run_arm_seed_subsets_fill_one_cache_that_a_full_call_reads_back(tmp_path):
    calls: list[int] = []

    def fake(encoder, *, seed, min_epochs, **_):
        calls.append(seed)
        return {"mean_ccc": float(seed), "epochs": min_epochs}

    kw = dict(cache_dir=tmp_path, n_seeds=5, finetune=fake, min_epochs=1)
    enc = _encoder()
    part = gate_mod._run_arm("arm", enc, seed_subset=[1, 3], **kw)
    assert calls == [1, 3]
    assert [r["mean_ccc"] for r in part] == [1.0, 3.0]

    full = gate_mod._run_arm("arm", enc, **kw)
    assert calls == [1, 3, 0, 2, 4]  # cached seeds are not recomputed
    assert [r["mean_ccc"] for r in full] == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_run_arm_rejects_a_seed_outside_the_gate(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        gate_mod._run_arm(
            "arm", _encoder(), cache_dir=tmp_path, n_seeds=3, finetune=None,
            min_epochs=1, seed_subset=[3],
        )


def test_finetune_ccc_reports_the_loops_epoch_count_not_its_argument(
    tmp_path, monkeypatch
):
    _spy_train_stage(monkeypatch, {**_RETURNED, "epochs_run": 1})

    result = gate_mod.finetune_ccc(
        _encoder(), _tiny_model_cfg(), manifest_root=tmp_path, min_epochs=2, seed=0
    )

    assert result["epochs"] == 1


def test_finetune_ccc_scores_the_final_epoch_not_the_best(tmp_path, monkeypatch):
    _spy_train_stage(monkeypatch, _RETURNED)

    result = gate_mod.finetune_ccc(
        _encoder(), _tiny_model_cfg(), manifest_root=tmp_path, min_epochs=2, seed=0
    )

    assert result == {
        "ccc_valence": 0.3,
        "ccc_arousal": 0.1,
        "mean_ccc": 0.2,
        "epochs": 2,
    }


def test_finetune_ccc_writes_nothing_outside_its_temp_dir(tmp_path, monkeypatch):
    """checkpoint_root/results_root must not point at the shared defaults, or
    concurrent arms would resume from one another's latest.pt."""
    seen = _spy_train_stage(monkeypatch, _RETURNED)

    gate_mod.finetune_ccc(
        _encoder(), _tiny_model_cfg(), manifest_root=tmp_path, min_epochs=2, seed=0
    )

    ((cfg, _, _),) = seen
    work = str(cfg.train.pretrain_encoder_path).rsplit("/", 1)[0]
    assert str(cfg.train.checkpoint_root).startswith(work)
    assert str(cfg.train.results_root).startswith(work)


@pytest.mark.integration
def test_finetune_ccc_end_to_end_on_a_tiny_encoder(tmp_path):
    """Real Hydra composition, real run_train_stage, real VideoMAEEncoder --
    pins that the composed recipe actually trains and that the count comes
    from the loop."""
    from video_encoding.scripts.build_afewva_manifest import build
    from video_encoding.tests.test_train_stage import _write_clip

    rng = np.random.default_rng(0)
    for i in range(10):
        _write_clip(tmp_path / f"{i:03d}", 24, rng, actor=f"actor{i // 2}")
    build(tmp_path, tmp_path)

    # VideoMAEVARegressor's head has 12 attention heads, so embed_dim % 12 == 0.
    model_cfg = _tiny_model_cfg(embed_dim=24)
    torch.manual_seed(0)
    result = gate_mod.finetune_ccc(
        VideoMAEEncoder(model_cfg),
        model_cfg,
        manifest_root=tmp_path,
        min_epochs=2,
        seed=0,
    )

    assert result["epochs"] == 2
    assert np.isfinite(result["ccc_valence"]) and np.isfinite(result["ccc_arousal"])


def test_finetune_ccc_seeds_the_rng_from_its_seed_not_the_callers_state(
    tmp_path, monkeypatch
):
    """train.seed is only applied by run_training, which the gate bypasses:
    the training RNG must depend on `seed` alone, or parallel seed-subset jobs
    replay each other's runs."""
    draws: list = []

    def _spy(cfg, **_):
        draws.append(torch.rand(4))
        return _RETURNED

    monkeypatch.setattr(train_mod, "run_train_stage", _spy)
    for burn, seed in [(0, 3), (99, 3), (0, 4)]:
        encoder = _encoder()  # reseeds the global RNG, as gate_cli's does
        torch.rand(burn)  # the state an earlier run in the job leaves behind
        gate_mod.finetune_ccc(
            encoder, _tiny_model_cfg(), manifest_root=tmp_path, min_epochs=1, seed=seed
        )

    assert torch.equal(draws[0], draws[1])
    assert not torch.equal(draws[0], draws[2])


class _Killed(Exception):
    pass


def test_finetune_ccc_persist_dir_resumes_after_a_kill_and_keeps_the_result(
    tmp_path, monkeypatch
):
    """A job killed during epoch 3 of 3 must resume at epoch 3 in the next job
    (not restart), score the full run, and leave no multi-GB checkpoints."""
    from video_encoding.scripts.build_afewva_manifest import build
    from video_encoding.tests.test_train_stage import _write_clip

    data = tmp_path / "data"
    rng = np.random.default_rng(0)
    for i in range(10):
        _write_clip(data / f"{i:03d}", 24, rng, actor=f"actor{i // 2}")
    build(data, data)
    model_cfg = _tiny_model_cfg(embed_dim=24)
    torch.manual_seed(0)
    encoder = VideoMAEEncoder(model_cfg)
    persist = tmp_path / "persist"
    kw = dict(manifest_root=data, min_epochs=3, seed=0, persist_dir=persist)

    real_train_epoch = train_mod._train_epoch
    epochs_started: list[int] = []

    def kill_on_third(*a, **k):
        epochs_started.append(1)
        if len(epochs_started) == 3:
            raise _Killed
        return real_train_epoch(*a, **k)

    monkeypatch.setattr(train_mod, "_train_epoch", kill_on_third)
    with pytest.raises(_Killed):
        gate_mod.finetune_ccc(encoder, model_cfg, **kw)

    epochs_started.clear()
    monkeypatch.setattr(train_mod, "_train_epoch", lambda *a, **k: (
        epochs_started.append(1), real_train_epoch(*a, **k))[1])
    result = gate_mod.finetune_ccc(encoder, model_cfg, **kw)

    assert len(epochs_started) == 1  # resumed at epoch 3, did not restart
    assert result["epochs"] == 3
    assert np.isfinite(result["mean_ccc"])
    (seed_dir,) = persist.glob("*/seed0")
    assert json.loads((seed_dir / "final.json").read_text()) == result
    assert not (seed_dir / "ckpt" / "latest.pt").exists()
    assert not (seed_dir / "ckpt" / "best.pt").exists()

    # A finished seed is read back, never retrained.
    monkeypatch.setattr(train_mod, "_train_epoch", lambda *a, **k: pytest.fail("retrained"))
    assert gate_mod.finetune_ccc(encoder, model_cfg, **kw) == result


# --- classify ------------------------------------------------------------------


def test_classify_borderline_pass_with_nonzero_spread():
    status, ci95, _ = gate_mod.classify([0.30, 0.42, 0.48], [0.10, 0.12, 0.08])
    assert status == "PASS"
    assert ci95[0] > 0


def test_classify_fail():
    status, ci95, _ = gate_mod.classify([0.10, 0.08, 0.12], [0.50, 0.52, 0.48])
    assert status == "FAIL"
    assert ci95[1] < 0


def test_classify_spread_exceeding_the_difference_is_inconclusive():
    """The mean favours the checkpoint -- a one-seed gate would have passed."""
    status, ci95, _ = gate_mod.classify([0.9, 0.05, 0.4], [0.3, 0.35, 0.32])
    assert status == "INCONCLUSIVE"
    assert ci95[0] <= 0 <= ci95[1]


def test_classify_uses_t_not_z():
    _, ci95, mde = gate_mod.classify([0.5, 0.6, 0.7], [0.0, 0.0, 0.0])
    assert mde == pytest.approx(4.303 * 0.1 / np.sqrt(3), abs=1e-3)
    assert ci95 == pytest.approx((0.6 - mde, 0.6 + mde))


def test_classify_mde_grows_with_seed_variance():
    control = [0.10, 0.11, 0.09]
    _, _, tight = gate_mod.classify([0.50, 0.51, 0.49], control)
    _, _, loose = gate_mod.classify([0.30, 0.70, 0.50], control)
    assert loose > tight


def test_classify_uses_the_critical_value_for_its_own_n():
    """n=5 must read df=4 (2.776), not the n=3 value."""
    diffs = [0.1, 0.2, 0.3, 0.4, 0.5]
    _, _, mde = gate_mod.classify(diffs, [0.0] * 5)
    assert mde == pytest.approx(2.776 * np.std(diffs, ddof=1) / np.sqrt(5), abs=1e-3)


# --- gate --------------------------------------------------------------------


def _bundle(run_id: str, seed: int):
    from video_encoding.scripts._diag_common import DiagBundle

    return DiagBundle(encoder=_encoder(seed), model=None, run_id=run_id, round=None)


class _FakeFinetune:
    """Scripted per-(arm, seed) CCCs; the arm is recognised by encoder object."""

    def __init__(self, arms: dict, scripted: dict, *, epochs=None, fail_on=None):
        self.arm_of = {id(bundle.encoder): name for name, bundle in arms.items()}
        self.scripted = scripted
        self.epochs = epochs or {}
        self.fail_on = fail_on
        self.calls: list[tuple[str, int]] = []
        self.kwargs: list[dict] = []

    def __call__(
        self, encoder, model_cfg, *, manifest_root, min_epochs, seed, head_lr_mult=1.0,
        persist_dir=None,
    ):
        arm = self.arm_of[id(encoder)]
        self.calls.append((arm, seed))
        self.kwargs.append(
            {
                "model_cfg": model_cfg,
                "manifest_root": manifest_root,
                "min_epochs": min_epochs,
            }
        )
        if self.fail_on is not None and len(self.calls) == self.fail_on:
            raise RuntimeError("walltime")
        ccc = self.scripted[arm][seed]
        return {
            "ccc_valence": ccc,
            "ccc_arousal": ccc,
            "mean_ccc": ccc,
            "epochs": self.epochs.get((arm, seed), min_epochs),
        }


_SCRIPTED = {
    "checkpoint": [0.30, 0.32, 0.28],
    "random_init": [0.10, 0.11, 0.09],
    "ceiling": [0.50, 0.52, 0.48],
}


def _arms():
    return {
        "checkpoint": _bundle("r1", 1),
        "random_init": _bundle("random_init", 2),
        "ceiling": _bundle("ceiling", 3),
    }


def _gate(arms, fake, cache_dir, *, ceiling=True, min_epochs=2, n_seeds=3):
    return gate_mod.gate(
        arms["checkpoint"],
        arms["random_init"],
        arms["ceiling"] if ceiling else None,
        model_cfg=_tiny_model_cfg(),
        manifest_root=cache_dir.parent,
        min_epochs=min_epochs,
        n_seeds=n_seeds,
        cache_dir=cache_dir,
        finetune=fake,
    )


@pytest.mark.parametrize("n_seeds", [2, 11])
def test_gate_refuses_seed_counts_it_cannot_put_an_interval_on(tmp_path, n_seeds):
    arms = _arms()
    fake = _FakeFinetune(arms, _SCRIPTED)
    with pytest.raises(ValueError, match="n_seeds"):
        _gate(arms, fake, tmp_path / "cache", n_seeds=n_seeds)
    assert fake.calls == []


@pytest.mark.parametrize("ceiling, n_calls", [(True, 9), (False, 6)])
def test_gate_runs_each_arm_seed_once(tmp_path, ceiling, n_calls):
    arms = _arms()
    fake = _FakeFinetune(arms, _SCRIPTED)
    _gate(arms, fake, tmp_path / "cache", ceiling=ceiling)
    assert len(fake.calls) == n_calls
    assert len(set(fake.calls)) == n_calls
    for arm in {a for a, _ in fake.calls}:
        assert {s for a, s in fake.calls if a == arm} == {0, 1, 2}


def test_gate_refuses_an_arm_that_ran_short(tmp_path):
    arms = _arms()
    fake = _FakeFinetune(arms, _SCRIPTED, epochs={("random_init", 1): 1})
    with pytest.raises(ValueError, match="random_init"):
        _gate(arms, fake, tmp_path / "cache")


def test_gate_report_carries_both_arms_the_diff_and_the_interval(tmp_path):
    arms = _arms()
    report = _gate(arms, _FakeFinetune(arms, _SCRIPTED), tmp_path / "cache")

    ckpt, control = report.metrics["checkpoint_ccc"], report.metrics["control_ccc"]
    assert ckpt.n == control.n == 3 and not ckpt.deterministic
    assert ckpt.mean == pytest.approx(0.30)
    assert control.mean == pytest.approx(0.10)
    assert report.metrics["diff_vs_control"].mean == pytest.approx(0.20)
    expected = gate_mod.classify(_SCRIPTED["checkpoint"], _SCRIPTED["random_init"])
    assert report.meta["status"] == expected[0] == "PASS"
    assert report.meta["ci95"] == pytest.approx(list(expected[1]))
    assert report.meta["mde"] == pytest.approx(expected[2])
    assert report.meta["underpowered"] is False
    assert report.meta["min_epochs"] == 2 and report.meta["n_seeds"] == 3
    assert report.meta["input_kind"] == "real"


def test_gate_forwards_one_recipe_to_every_fine_tune(tmp_path):
    arms = _arms()
    fake = _FakeFinetune(arms, _SCRIPTED)
    report = _gate(arms, fake, tmp_path / "cache", min_epochs=4)

    assert len(fake.kwargs) == 9
    for kwargs in fake.kwargs:
        assert kwargs["model_cfg"] == _tiny_model_cfg()
        assert kwargs["manifest_root"] == tmp_path
        assert kwargs["min_epochs"] == 4
    assert report.meta["epochs_run"] == {
        arm: [4, 4, 4] for arm in ("checkpoint", "control", "ceiling")
    }


def test_the_fake_fine_tune_matches_the_real_signature():
    """Every gate test injects `_FakeFinetune`; if `finetune_ccc`'s signature
    drifted, `gate`'s default path would break while the suite stayed green."""
    assert (
        inspect.signature(_FakeFinetune.__call__).parameters.keys() - {"self"}
        == inspect.signature(gate_mod.finetune_ccc).parameters.keys()
    )
    assert inspect.signature(gate_mod.gate).parameters["finetune"].default is (
        gate_mod.finetune_ccc
    )


def test_a_wide_pass_is_not_flagged_underpowered(tmp_path):
    arms = _arms()
    wide = {**_SCRIPTED, "checkpoint": [0.9, 0.95, 1.0], "random_init": [0.1, 0.5, 0.2]}
    report = _gate(arms, _FakeFinetune(arms, wide), tmp_path / "cache")
    assert report.meta["status"] == "PASS"
    assert report.meta["mde"] > gate_mod.MDE_OF_INTEREST
    assert report.meta["underpowered"] is False


def test_gate_flags_a_non_pass_the_seeds_could_not_resolve(tmp_path):
    arms = _arms()
    noisy = {**_SCRIPTED, "checkpoint": [0.9, 0.05, 0.4]}
    report = _gate(arms, _FakeFinetune(arms, noisy), tmp_path / "cache")
    assert report.meta["status"] == "INCONCLUSIVE"
    assert report.meta["mde"] > gate_mod.MDE_OF_INTEREST
    assert report.meta["underpowered"] is True


def test_gate_positions_the_checkpoint_between_floor_and_ceiling(tmp_path):
    arms = _arms()
    report = _gate(arms, _FakeFinetune(arms, _SCRIPTED), tmp_path / "cache")
    assert report.meta["position"] == pytest.approx(0.5)
    assert report.metrics["ceiling_ccc"].mean == pytest.approx(0.50)


def test_gate_without_a_ceiling_still_classifies(tmp_path):
    arms = _arms()
    report = _gate(
        arms, _FakeFinetune(arms, _SCRIPTED), tmp_path / "cache", ceiling=False
    )
    assert report.meta["position"] is None
    assert "ceiling" in report.meta["position_reason"]
    assert "ceiling_ccc" not in report.metrics
    assert report.meta["status"] == "PASS"


def test_gate_has_no_position_when_the_ceiling_is_not_above_the_floor(tmp_path):
    arms = _arms()
    inverted = {**_SCRIPTED, "ceiling": [0.05, 0.06, 0.04]}
    report = _gate(arms, _FakeFinetune(arms, inverted), tmp_path / "cache")
    assert report.meta["position"] is None
    assert report.meta["position_reason"]


# --- cache ---------------------------------------------------------------------


def test_warm_cache_only_fine_tunes_the_new_checkpoint(tmp_path):
    cache = tmp_path / "cache"
    arms = _arms()
    _gate(arms, _FakeFinetune(arms, _SCRIPTED), cache)

    arms_b = {**arms, "checkpoint": _bundle("r2", 4)}
    fake_b = _FakeFinetune(arms_b, _SCRIPTED)
    _gate(arms_b, fake_b, cache)

    assert sorted(fake_b.calls) == [
        ("checkpoint", 0),
        ("checkpoint", 1),
        ("checkpoint", 2),
    ]


def test_changed_weights_invalidate_the_cached_arm(tmp_path):
    cache = tmp_path / "cache"
    arms = _arms()
    _gate(arms, _FakeFinetune(arms, _SCRIPTED), cache)

    with torch.no_grad():
        next(arms["checkpoint"].encoder.parameters()).add_(1.0)
    fake = _FakeFinetune(arms, _SCRIPTED)
    _gate(arms, fake, cache)

    assert sorted(fake.calls) == [
        ("checkpoint", 0),
        ("checkpoint", 1),
        ("checkpoint", 2),
    ]


@pytest.mark.parametrize(
    "changed, match",
    [({"min_epochs": 5}, "min_epochs"), ({"n_seeds": 4}, "gate_seeds")],
)
def test_cache_written_under_other_parameters_is_refused(tmp_path, changed, match):
    cache = tmp_path / "cache"
    arms = _arms()
    _gate(arms, _FakeFinetune(arms, _SCRIPTED), cache, min_epochs=10)

    fake = _FakeFinetune(arms, _SCRIPTED)
    with pytest.raises(ValueError, match=match):
        _gate(arms, fake, cache, **{"min_epochs": 10, **changed})
    assert fake.calls == []


@pytest.mark.parametrize("key, value", [("lr", 1e-4), ("head_bounded", True)])
def test_cache_written_under_another_recipe_is_refused(tmp_path, key, value):
    # ssl-diag-08 job 362 collapsed at lr 1e-4 with a bounded head; numbers
    # cached under that recipe must never be mixed with the fixed one's.
    cache = tmp_path / "cache"
    arms = _arms()
    _gate(arms, _FakeFinetune(arms, _SCRIPTED), cache)
    params = cache / "gate_params.json"
    params.write_text(json.dumps({**json.loads(params.read_text()), key: value}))

    fake = _FakeFinetune(arms, _SCRIPTED)
    with pytest.raises(ValueError, match=key):
        _gate(arms, fake, cache)
    assert fake.calls == []


def test_cache_written_under_another_split_is_refused(tmp_path):
    cache = tmp_path / "cache"
    arms = _arms()
    _gate(arms, _FakeFinetune(arms, _SCRIPTED), cache)
    params = cache / "gate_params.json"
    params.write_text(
        json.dumps({**json.loads(params.read_text()), "split_fraction": 0.3})
    )

    fake = _FakeFinetune(arms, _SCRIPTED)
    with pytest.raises(ValueError, match="split_fraction"):
        _gate(arms, fake, cache)
    assert fake.calls == []


def test_a_rerun_with_unchanged_weights_fine_tunes_nothing(tmp_path):
    cache = tmp_path / "cache"
    arms = _arms()
    _gate(arms, _FakeFinetune(arms, _SCRIPTED), cache)

    fake = _FakeFinetune(arms, _SCRIPTED)
    _gate(arms, fake, cache)

    assert fake.calls == []


def test_interrupted_gate_resumes_where_it_stopped(tmp_path):
    cache = tmp_path / "cache"
    arms = _arms()
    first = _FakeFinetune(arms, _SCRIPTED, fail_on=5)
    with pytest.raises(RuntimeError):
        _gate(arms, first, cache)
    done = set(first.calls[:4])

    second = _FakeFinetune(arms, _SCRIPTED)
    report = _gate(arms, second, cache)

    assert len(second.calls) == 9 - 4
    assert done.isdisjoint(second.calls)
    assert report.meta["status"] == "PASS"


# --- check -----------------------------------------------------------------


def _gate_inputs(tmp_path, **overrides):
    arms = _arms()
    gate_inputs = {
        "model_cfg": _tiny_model_cfg(),
        "min_epochs": 2,
        "n_seeds": 3,
        "cache_dir": tmp_path / "cache",
        "ceiling_bundle": arms["ceiling"],
        **overrides,
    }
    return arms, {"control_bundle": arms["random_init"], "gate": gate_inputs}


def _afewva(tmp_path, monkeypatch, *, manifest=True):
    root = tmp_path / "afewva"
    root.mkdir()
    if manifest:
        (root / "manifest.csv").write_text("")
    monkeypatch.setenv("AFEWVA_ROOT", str(root))
    return root


def _assert_gate_not_met(result, reason_part):
    (report,) = result.values()
    assert list(result) == ["finetune_afewva"]
    assert report.meta["status"] == "GATE_NOT_MET"
    assert reason_part in report.meta["reason"]
    assert report.metrics == {}
    assert "skipped" not in report.meta


def test_check_without_afewva_root_is_gate_not_met(tmp_path, monkeypatch):
    monkeypatch.delenv("AFEWVA_ROOT", raising=False)
    arms, inputs = _gate_inputs(tmp_path)
    _assert_gate_not_met(gate_mod.check(arms["checkpoint"], inputs), "AFEWVA_ROOT")


def test_check_without_a_manifest_is_gate_not_met(tmp_path, monkeypatch):
    _afewva(tmp_path, monkeypatch, manifest=False)
    arms, inputs = _gate_inputs(tmp_path)
    _assert_gate_not_met(gate_mod.check(arms["checkpoint"], inputs), "manifest.csv")


def test_check_off_the_run_axis_is_gate_not_met(tmp_path, monkeypatch):
    _afewva(tmp_path, monkeypatch)
    arms, inputs = _gate_inputs(tmp_path)
    del inputs["gate"]
    _assert_gate_not_met(gate_mod.check(arms["checkpoint"], inputs), "run axis")


def test_check_on_a_levjepa_checkpoint_is_gate_not_met(tmp_path, monkeypatch):
    _afewva(tmp_path, monkeypatch)
    arms, inputs = _gate_inputs(
        tmp_path, model_cfg=_tiny_model_cfg(encoder="levjepa_hf")
    )
    _assert_gate_not_met(gate_mod.check(arms["checkpoint"], inputs), "levjepa_hf")


def test_check_forwards_its_inputs_to_gate(tmp_path, monkeypatch):
    root = _afewva(tmp_path, monkeypatch)
    arms, inputs = _gate_inputs(tmp_path)
    seen = {}

    def _spy(bundle, control_bundle, ceiling_bundle, **kwargs):
        seen.update(bundle=bundle, control=control_bundle, ceiling=ceiling_bundle)
        seen.update(kwargs)
        return "report"

    monkeypatch.setattr(gate_mod, "gate", _spy)
    result = gate_mod.check(arms["checkpoint"], inputs)

    assert result == {"finetune_afewva": "report"}
    assert seen["bundle"] is arms["checkpoint"]
    assert seen["control"] is arms["random_init"]
    assert seen["ceiling"] is arms["ceiling"]
    assert seen["manifest_root"] == root
    assert seen["min_epochs"] == 2 and seen["n_seeds"] == 3
    assert seen["cache_dir"] == tmp_path / "cache"
    assert seen["model_cfg"] is inputs["gate"]["model_cfg"]
