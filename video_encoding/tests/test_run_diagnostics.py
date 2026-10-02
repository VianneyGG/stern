"""Behavioural tests for run_diagnostics.py -- the orchestrator that turns the
landed `check(bundle, inputs)` functions into one diagnostics.json.

Every test that loads a checkpoint uses a real (tiny) VideoMAE state dict and a
real WebDataset shard fixture, so the checks under the orchestrator run for
real; only the registry itself is patched, and only where a test needs a check
with a controlled failure mode.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pytest

from video_encoding.scripts import run_diagnostics
from video_encoding.scripts.render_report import render
from video_encoding.scripts._diag_common import (
    LADDER_REVISION,
    LADDER_RUN_IDS,
    RungUnavailable,
    DiagReport,
    MetricValue,
    Rung,
    random_init_bundle,
)
from video_encoding.tests.test_diag_common import (
    _tiny_model_cfg,
    make_fake_shard,
    write_checkpoint,
    write_snapshot,
)


def _cfg():
    return _tiny_model_cfg(embed_dim=32, num_heads=4)


def _shard(tmp_path):
    return make_fake_shard(tmp_path, {"c0": 8, "c1": 8, "c2": 8})


def _report(**meta) -> DiagReport:
    return DiagReport(
        metrics={"m": MetricValue.deterministic_value(1.0)},
        arrays={},
        meta={"LADDER_REVISION": LADDER_REVISION, "input_kind": "real", **meta},
    )


# --- registry, normalise, build_inputs -------------------------------------


def test_normalise_dict_result_keeps_its_keys():
    r = _report()
    pairs = run_diagnostics.normalise(
        "diag_depth", {"residual_branch_norm": r}, run_diagnostics.EMITS["diag_depth"]
    )
    assert pairs == [("residual_branch_norm", r)]


def test_normalise_list_result_names_every_pair_by_registry_key():
    r1, r2 = _report(), _report(input_kind="random")
    pairs = run_diagnostics.normalise(
        "labelfree", [r1, r2], run_diagnostics.EMITS["labelfree"]
    )
    assert pairs == [("labelfree", r1), ("labelfree", r2)]


def test_normalise_rejects_an_undeclared_emitted_name():
    with pytest.raises(ValueError, match="sneaky_metric"):
        run_diagnostics.normalise(
            "diag_depth",
            {"sneaky_metric": _report()},
            run_diagnostics.EMITS["diag_depth"],
        )


def test_validate_registry_rejects_an_uncontrolled_emitted_name():
    with pytest.raises(ValueError, match="uncontrolled_thing"):
        run_diagnostics.validate_registry(
            {"k": lambda b, i: []}, {"k": frozenset({"uncontrolled_thing"})}
        )


def test_validate_registry_rejects_a_check_without_emits():
    with pytest.raises(ValueError, match="orphan"):
        run_diagnostics.validate_registry({"orphan": lambda b, i: []}, {})


def test_real_registry_passes_validation():
    assert (
        run_diagnostics.validate_registry(run_diagnostics.CHECKS, run_diagnostics.EMITS)
        is None
    )


def test_build_inputs_returns_exactly_the_seven_keys(tmp_path):
    cfg = _cfg()
    bundle = random_init_bundle(cfg, seed=0)
    inputs = run_diagnostics.build_inputs(
        bundle,
        shard_path=_shard(tmp_path),
        model_cfg=cfg,
        seed=0,
        n_windows=2,
        ref_bundle=None,
    )
    assert set(inputs) == {
        "real",
        "random",
        "shard_path",
        "n_windows",
        "seed",
        "control_bundle",
        "ref_bundle",
    }
    assert inputs["real"].shape == inputs["random"].shape
    assert inputs["real"].shape[0] == 2
    assert inputs["control_bundle"] is not bundle
    assert inputs["ref_bundle"] is None


# --- run axis --------------------------------------------------------------


def _mesh_with_run(root, cfg, run_id="r1"):
    mesh_shared = root / "mesh"
    write_checkpoint(mesh_shared, run_id, cfg)
    return mesh_shared


def _run(
    tmp_path, *, out_name="out", checks=("diag_depth",), run_ids=("r1",), **kwargs
):
    cfg = _cfg()
    mesh_shared = kwargs.pop("mesh_shared", None) or _mesh_with_run(tmp_path, cfg)
    shard_path = kwargs.pop("shard_path", None) or _shard(tmp_path)
    out_dir = tmp_path / out_name
    params = dict(
        run_ids=list(run_ids),
        checks=list(checks),
        mesh_shared=mesh_shared,
        shard_path=shard_path,
        model_cfg=cfg,
        out_dir=out_dir,
        ladder_path=None,
        skip_ladder_check=True,
        n_windows=2,
    )
    params.update(kwargs)
    run_diagnostics.run(**params)
    return json.loads((params["out_dir"] / "diagnostics.json").read_text())


@pytest.fixture(scope="module")
def two_check_run(tmp_path_factory):
    """diag_depth + attention_entropy_by_block on one fixture run, computed once
    and shared by the read-only assertions below."""
    return _run(
        tmp_path_factory.mktemp("two_check"),
        checks=("diag_depth", "attention_entropy_by_block"),
    )


def test_paired_check_writes_both_input_kinds(two_check_run):
    kinds = sorted(
        e["input_kind"]
        for e in two_check_run["reports"]
        if e["check"] == "attention_entropy_by_block"
    )
    assert kinds == ["random", "real"]


def test_sibling_metrics_of_diag_depth_are_written(two_check_run):
    ratio = [e for e in two_check_run["reports"] if e["check"] == "branch_norm_ratio"]
    assert len(ratio) == 1
    assert "block_0_attn_ratio" in ratio[0]["metrics"]


def test_envelope_carries_the_schema_fields(two_check_run):
    envelope = two_check_run["reports"][0]
    assert set(envelope) == {
        "check",
        "run_id",
        "round",
        "rung",
        "input_kind",
        "metrics",
        "meta",
    }
    assert (envelope["run_id"], envelope["round"], envelope["rung"]) == (
        "r1",
        None,
        None,
    )
    assert two_check_run["schema"] == 1


def test_missing_ref_bundle_is_a_skip_not_a_report(two_check_run):
    assert {
        "check": "residual_branch_norm",
        "run_id": "r1",
        "round": None,
        "rung": None,
        "reason": "no ref_bundle supplied",
    } in two_check_run["skipped"]
    assert "residual_branch_norm" not in {e["check"] for e in two_check_run["reports"]}


def test_replication_rule_holds_over_the_whole_file(two_check_run):
    metrics = [
        (e["check"], m) for e in two_check_run["reports"] for m in e["metrics"].values()
    ]
    assert len(metrics) > 10
    for check, m in metrics:
        assert m["deterministic"] or m["n"] >= 3, (check, m)
        if m["deterministic"] and check not in run_diagnostics.WEIGHTS_ONLY:
            assert m["replicated"] is False, (check, m)
    # attention entropy is deterministic but reads activations: never replicated.
    entropy = [m for c, m in metrics if c == "attention_entropy_by_block"]
    assert {m["replicated"] for m in entropy} == {False}
    # decode_from_depth is sampled over seeds, so it IS replicated.
    decode = [m for c, m in metrics if c == "decode_from_depth"]
    assert {(m["replicated"], m["n"]) for m in decode} == {(True, 3)}


def test_raising_check_is_recorded_and_others_still_write(tmp_path, monkeypatch):
    def _boom(bundle, inputs):
        raise RuntimeError("intentional")

    monkeypatch.setitem(run_diagnostics.CHECKS, "_boom", _boom)
    monkeypatch.setitem(run_diagnostics.EMITS, "_boom", frozenset({"_boom"}))
    out = _run(tmp_path, checks=("_boom", "attention_entropy_by_block"))
    boom = [e for e in out["errors"] if e["check"] == "_boom"]
    assert len(boom) == 1
    assert "intentional" in boom[0]["reason"]
    assert boom[0]["run_id"] == "r1"
    # The AFEW-VA gate always runs (GATES), so it is not "another check".
    assert {e["check"] for e in out["reports"]} - run_diagnostics.GATES == {
        "attention_entropy_by_block"
    }


def test_unknown_check_name_raises_listing_valid_names(tmp_path):
    with pytest.raises(ValueError, match="diag_depth"):
        run_diagnostics.run(
            run_ids=["r1"],
            checks=["not_a_check"],
            mesh_shared=tmp_path,
            shard_path=tmp_path,
            model_cfg=_cfg(),
            out_dir=tmp_path / "out",
            ladder_path=None,
            skip_ladder_check=True,
        )


def test_n_replicates_below_three_raises(tmp_path):
    with pytest.raises(ValueError, match="n_replicates"):
        run_diagnostics.run(
            run_ids=["r1"],
            checks=["diag_depth"],
            mesh_shared=tmp_path,
            shard_path=tmp_path,
            model_cfg=_cfg(),
            out_dir=tmp_path / "out",
            ladder_path=None,
            skip_ladder_check=True,
            n_replicates=2,
        )


def test_missing_checkpoint_is_an_error_naming_its_path(tmp_path):
    out = _run(tmp_path, run_ids=("nope",), checks=("attention_entropy_by_block",))
    assert [e["run_id"] for e in out["errors"]] == ["nope"]
    assert "checkpoints/nope/master.pt" in out["errors"][0]["reason"]
    assert out["reports"] == []


def test_gate_runs_even_when_not_requested(tmp_path, monkeypatch):
    calls = []

    def _gate(bundle, inputs):
        calls.append(bundle.run_id)
        return []

    monkeypatch.setitem(run_diagnostics.CHECKS, "_gate", _gate)
    monkeypatch.setitem(run_diagnostics.EMITS, "_gate", frozenset({"_gate"}))
    monkeypatch.setattr(run_diagnostics, "GATES", frozenset({"_gate"}))
    _run(tmp_path, checks=("attention_entropy_by_block",))
    assert calls == ["r1"]


def test_paired_check_missing_random_is_a_wiring_error(tmp_path, monkeypatch):
    def _half(bundle, inputs):
        return [_report()]

    monkeypatch.setitem(run_diagnostics.CHECKS, "_half", _half)
    monkeypatch.setitem(run_diagnostics.EMITS, "_half", frozenset({"_half"}))
    monkeypatch.setattr(run_diagnostics, "PAIRED", run_diagnostics.PAIRED | {"_half"})
    out = _run(tmp_path, checks=("_half",))
    assert [e["check"] for e in out["errors"]] == ["_half"]
    assert "random" in out["errors"][0]["reason"]
    assert [e for e in out["reports"] if e["check"] not in run_diagnostics.GATES] == []


def test_identical_invocations_write_identical_bytes(tmp_path):
    cfg = _cfg()
    mesh_shared = _mesh_with_run(tmp_path, cfg)
    shard_path = _shard(tmp_path)
    for name in ("a", "b"):
        _run(
            tmp_path,
            out_name=name,
            checks=("diag_depth", "attention_entropy_by_block"),
            mesh_shared=mesh_shared,
            shard_path=shard_path,
        )
    a = (tmp_path / "a" / "diagnostics.json").read_bytes()
    assert a == (tmp_path / "b" / "diagnostics.json").read_bytes()


def test_no_ladder_on_record_raises_before_any_load(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(run_diagnostics, "load_bundle", lambda *a, **k: calls.append(a))
    with pytest.raises(ValueError, match="ladder"):
        run_diagnostics.run(
            run_ids=["r1"],
            checks=["diag_depth"],
            mesh_shared=tmp_path,
            shard_path=tmp_path,
            model_cfg=_cfg(),
            out_dir=tmp_path / "out",
            ladder_path=None,
        )
    assert calls == []


def test_stale_ladder_revision_raises_naming_it(tmp_path):
    ladder_path = tmp_path / "ladder.json"
    ladder_path.write_text(
        json.dumps({"ladder_revision": "1999-01-01.0", "ladder_tiers": {}})
    )
    with pytest.raises(ValueError, match="1999-01-01.0"):
        run_diagnostics.run(
            run_ids=["r1"],
            checks=["diag_depth"],
            mesh_shared=tmp_path,
            shard_path=tmp_path,
            model_cfg=_cfg(),
            out_dir=tmp_path / "out",
            ladder_path=ladder_path,
        )


# --- ladder axis -----------------------------------------------------------


@pytest.fixture(scope="module")
def ladder_dir(tmp_path_factory):
    """run_ladder with ROUND and OVERFIT checkpoints and no public or
    supervised checkpoint -- three rungs present, two unavailable."""
    root = tmp_path_factory.mktemp("ladder")
    cfg = _cfg()
    mesh_shared = root / "mesh"
    write_checkpoint(mesh_shared, "round-fixture", cfg)
    write_checkpoint(mesh_shared, "overfit-fixture", cfg)
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("PUBLIC_VIDEOMAEV2_LOCAL_PATH", raising=False)
        run_diagnostics.run_ladder(
            checks=["attention_entropy_by_block"],
            mesh_shared=mesh_shared,
            shard_path=_shard(root),
            model_cfg=cfg,
            out_dir=root / "out",
            ladder_run_ids={
                Rung.ROUND: "round-fixture",
                Rung.OVERFIT: "overfit-fixture",
            },
            n_windows=2,
        )
    return root / "out", mesh_shared


def test_ladder_records_unavailable_rungs_with_reasons(ladder_dir):
    out_dir, _ = ladder_dir
    ladder = json.loads((out_dir / "ladder.json").read_text())
    status = {r["rung"]: r for r in ladder["rung_status"]}
    assert list(status) == [
        "random_init",
        "round",
        "overfit",
        "public_videomaev2",
        "supervised_va",
    ]
    assert status["random_init"]["status"] == "built"
    assert status["public_videomaev2"]["status"] == "unavailable"
    assert (
        "public_videomaev2 checkpoint not available"
        in status["public_videomaev2"]["reason"]
    )
    assert ladder["ladder_revision"] == LADDER_REVISION


def test_ladder_with_three_rungs_tiers_everything_c(ladder_dir):
    out_dir, _ = ladder_dir
    tiers = json.loads((out_dir / "ladder.json").read_text())["ladder_tiers"]
    assert len(tiers) == 2 * 2  # two blocks x {real, random}
    assert {t["tier"] for t in tiers.values()} == {"C"}
    assert {t["present_rungs"] for t in tiers.values()} == {3}
    entry = tiers["attention_entropy_by_block.real.block0"]
    assert list(entry["rungs"]) == ["random_init", "round", "overfit"]


def test_ladder_envelopes_are_labelled_by_rung(ladder_dir):
    out_dir, _ = ladder_dir
    reports = json.loads((out_dir / "ladder.json").read_text())["reports"]
    assert sorted({e["rung"] for e in reports}) == ["overfit", "random_init", "round"]
    assert {e["run_id"] for e in reports if e["rung"] == "round"} == {"round-fixture"}


def test_ladder_writes_no_diagnostics_json(ladder_dir):
    out_dir, _ = ladder_dir
    assert sorted(p.name for p in out_dir.iterdir()) == ["ladder.json"]


def test_run_into_the_ladder_dir_leaves_ladder_json_untouched(ladder_dir, tmp_path):
    out_dir, mesh_shared = ladder_dir
    ladder_path = out_dir / "ladder.json"
    before = ladder_path.read_bytes()
    out = _run(
        tmp_path,
        run_ids=("round-fixture",),
        checks=("attention_entropy_by_block",),
        mesh_shared=mesh_shared,
        out_name="unused",
        out_dir=out_dir,
        ladder_path=ladder_path,
        skip_ladder_check=False,
    )
    assert ladder_path.read_bytes() == before
    assert out["ladder_tiers"] == json.loads(before)["ladder_tiers"]


def test_ladder_run_ids_parse_to_rungs():
    parsed = run_diagnostics.parse_ladder_run_ids("round=a, overfit=b")
    assert parsed == {Rung.ROUND: "a", Rung.OVERFIT: "b"}


def test_ladder_run_ids_resolve_the_recorded_solo_route():
    """Rung 2 is a merge=false run with no master.pt; a bare run id would load
    a file that does not exist, so its recorded route is substituted."""
    parsed = run_diagnostics.parse_ladder_run_ids("overfit=ssl-diag-overfit50")
    assert parsed == {Rung.OVERFIT: LADDER_RUN_IDS[Rung.OVERFIT]}
    assert parsed[Rung.OVERFIT]["node"] == "node030"


def test_run_ladder_seeds_the_random_init_rung(tmp_path, monkeypatch):
    seen = {}

    def fake_load(rung, model_cfg, **kw):
        seen[rung] = kw.get("seed")
        raise RungUnavailable("stub")

    monkeypatch.setattr(run_diagnostics, "load_ladder_rung", fake_load)
    run_diagnostics.run_ladder(
        checks=[],
        mesh_shared=tmp_path,
        shard_path=tmp_path / "shard-000000.tar",
        model_cfg=_tiny_model_cfg(),
        out_dir=tmp_path / "out",
        ladder_run_ids={},
        seed=7,
    )
    assert seen[Rung.RANDOM_INIT] == 7


def test_ladder_run_ids_reject_an_unknown_rung():
    with pytest.raises(ValueError, match="bogus"):
        run_diagnostics.parse_ladder_run_ids("bogus=a")


# --- round axis ------------------------------------------------------------


def _snapshots(root, cfg, rounds, *, gaps=None, manifest_rounds=None):
    snapshot_dir = root / "snapshots"
    for r in rounds:
        write_snapshot(snapshot_dir, "r1", cfg, round_num=r)
    lines = [
        json.dumps({"round": r, "gaps": (gaps or {}).get(r, [])})
        for r in (manifest_rounds if manifest_rounds is not None else rounds)
    ]
    (snapshot_dir / "r1" / "manifest.jsonl").write_text("\n".join(lines) + "\n")
    return snapshot_dir


def _run_rounds(
    tmp_path, rounds, snapshot_dir, *, checks=("diag_depth",), out_name="out"
):
    run_diagnostics.run_rounds(
        run_id="r1",
        rounds=rounds,
        snapshot_dir=snapshot_dir,
        checks=list(checks),
        shard_path=_shard(tmp_path),
        model_cfg=_cfg(),
        out_dir=tmp_path / out_name,
        ladder_path=None,
        skip_ladder_check=True,
        n_windows=2,
    )
    return json.loads((tmp_path / out_name / "diagnostics.json").read_text())


def test_rounds_without_snapshot_dir_raises(tmp_path):
    with pytest.raises(ValueError, match="snapshot"):
        _run_rounds(tmp_path, [5], None)


def test_rounds_missing_from_the_manifest_raise_naming_run_and_round(
    tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        run_diagnostics, "load_snapshot_bundle", lambda *a, **k: calls.append(a)
    )
    snapshot_dir = _snapshots(tmp_path, _cfg(), [5])
    with pytest.raises(ValueError, match=r"r1.*round 10: not captured"):
        _run_rounds(tmp_path, [5, 10], snapshot_dir)
    assert calls == []


def test_a_gapped_round_raises_with_its_gap_text(tmp_path):
    snapshot_dir = _snapshots(
        tmp_path, _cfg(), [5], gaps={5: ["no pending checkpoints found for round 5"]}
    )
    with pytest.raises(ValueError, match="no pending checkpoints found for round 5"):
        _run_rounds(tmp_path, [5], snapshot_dir)


@pytest.fixture(scope="module")
def rounds_4_5(tmp_path_factory):
    root = tmp_path_factory.mktemp("rounds")
    return _run_rounds(root, [4, 5], _snapshots(root, _cfg(), [4, 5]))


def test_round_n_minus_1_supplies_the_reference(rounds_4_5):
    residual = [
        e for e in rounds_4_5["reports"] if e["check"] == "residual_branch_norm"
    ]
    assert [(e["run_id"], e["round"]) for e in residual] == [("r1", 5)]
    assert "block_0_attn" in residual[0]["metrics"]
    skip = [s for s in rounds_4_5["skipped"] if s["check"] == "residual_branch_norm"]
    assert [s["round"] for s in skip] == [4]
    assert "round 3 reference snapshot not captured" in skip[0]["reason"]


def test_run_and_round_axes_share_one_schema(rounds_4_5, two_check_run, tmp_path):
    assert set(rounds_4_5) == set(two_check_run)
    assert set(rounds_4_5["reports"][0]) == set(two_check_run["reports"][0])
    for diagnostics in (rounds_4_5, two_check_run):
        html = render(diagnostics, tmp_path)
        assert "<h1>Verdict: GATE_NOT_MET</h1>" in html


def test_cli_rounds_with_two_run_ids_raises_before_any_load(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        run_diagnostics, "load_snapshot_bundle", lambda *a, **k: calls.append(a)
    )
    monkeypatch.setattr(run_diagnostics, "load_bundle", lambda *a, **k: calls.append(a))
    with pytest.raises(ValueError, match="exactly one"):
        run_diagnostics.main(
            [
                "--rounds",
                "4,5",
                "--run-ids",
                "a,b",
                "--snapshot-dir",
                str(tmp_path),
                "--shard-path",
                str(tmp_path),
                "--out-dir",
                str(tmp_path / "out"),
                "--skip-ladder-check",
            ]
        )
    assert calls == []


# --- verdicts and plots ----------------------------------------------------


def _sampled(mean, sd=1.0):
    return {
        "mean": mean,
        "sd": sd,
        "n": 3,
        "seeds": [1, 2, 3],
        "deterministic": False,
        "replicated": True,
    }


def _envelope(metric, check="residual_branch_norm", arrays_meta=None):
    return {
        "check": check,
        "run_id": "r1",
        "round": None,
        "rung": None,
        "input_kind": "real",
        "metrics": {"m": metric},
        "meta": dict(arrays_meta or {}),
    }


def _tiers(tier="A", check="residual_branch_norm"):
    return {
        f"{check}.real.m": {
            "tau": 1.0,
            "tier": tier,
            "present_rungs": 2,
            "rungs": {
                "random_init": {
                    "mean": 10.0,
                    "sd": 1.0,
                    "n": 3,
                    "deterministic": False,
                },
                "overfit": {"mean": 30.0, "sd": 1.0, "n": 3, "deterministic": False},
            },
        }
    }


def _stamped(metric, tier="A"):
    envelope = _envelope(metric)
    run_diagnostics.stamp_verdicts([envelope], _tiers(tier))
    return envelope["metrics"]["m"]


def test_a_bracketed_separated_value():
    m = _stamped(_sampled(20.0))
    assert m["bracket"] == "20 ± 1 · between random_init=10 and overfit=30"
    assert m["verdict"] == "SEPARATED"


def test_a_value_inside_the_nearest_rungs_band_is_inconclusive():
    assert _stamped(_sampled(10.5))["verdict"] == "INCONCLUSIVE"


def test_a_value_beyond_the_ladder_is_out_of_range():
    m = _stamped(_sampled(45.0))
    assert m["verdict"] == "OUT_OF_RANGE"
    assert "outside" in m["bracket"]


def test_an_unreplicated_metric_gets_no_verdict():
    unreplicated = {
        "mean": 20.0,
        "sd": None,
        "n": None,
        "seeds": [],
        "deterministic": True,
        "replicated": False,
    }
    m = _stamped(unreplicated)
    assert (m["verdict"], m["red"]) == (None, False)


def test_a_weights_only_metric_is_compared_exactly_against_deterministic_rungs():
    exact = {"mean": 10.0, "sd": None, "n": None, "seeds": [], "deterministic": True}
    tiers = _tiers()
    for rung in tiers["residual_branch_norm.real.m"]["rungs"].values():
        rung.update(sd=None, n=None, deterministic=True)

    on_a_rung = _envelope({**exact, "replicated": True})
    off_the_rungs = _envelope({**exact, "mean": 10.5, "replicated": True})
    run_diagnostics.stamp_verdicts([on_a_rung, off_the_rungs], tiers)

    on, off = on_a_rung["metrics"]["m"], off_the_rungs["metrics"]["m"]
    assert on["bracket"] == "10 · between random_init=10 and overfit=30"
    assert on["verdict"] == "INCONCLUSIVE"
    assert off["verdict"] == "SEPARATED"


def test_an_uncalibrated_metric_gets_no_verdict():
    envelope = _envelope(_sampled(20.0))
    run_diagnostics.stamp_verdicts([envelope], {})
    m = envelope["metrics"]["m"]
    assert (m["bracket"], m["verdict"], m["red"]) == (None, None, False)


def test_red_only_on_the_bad_side_and_only_for_tier_a_or_b():
    # 22 is nearest overfit=30 and separated below it: the bad side for a
    # residual_branch_norm, whose BAD_DIRECTION is "low".
    assert _stamped(_sampled(22.0), tier="A")["red"] is True
    assert _stamped(_sampled(22.0), tier="B")["red"] is True
    assert _stamped(_sampled(22.0), tier="C")["red"] is False
    # 18 is nearest random_init=10 and separated above it: the good side.
    assert _stamped(_sampled(18.0), tier="A")["red"] is False


def test_write_plots_draws_a_heatmap_deterministically(tmp_path):
    envelope = _envelope(_sampled(1.0), check="cka_matrix")
    skipped = run_diagnostics.write_plots(
        [envelope], [{"cka_full": np.eye(3)}], tmp_path / "a"
    )
    assert skipped == []
    assert envelope["meta"]["panel"] == "cka"
    (name,) = envelope["meta"]["plot"]
    first = (tmp_path / "a" / name).read_bytes()
    assert first[:8] == b"\x89PNG\r\n\x1a\n"
    again = _envelope(_sampled(1.0), check="cka_matrix")
    run_diagnostics.write_plots([again], [{"cka_full": np.eye(3)}], tmp_path / "b")
    assert (tmp_path / "b" / name).read_bytes() == first


def test_write_plots_skips_an_unsupported_shape(tmp_path):
    envelope = _envelope(_sampled(1.0), check="diag_qualitative")
    skipped = run_diagnostics.write_plots(
        [envelope], [{"mask_ratio_grid": np.zeros((2, 2, 2, 2))}], tmp_path
    )
    assert [s["check"] for s in skipped] == ["diag_qualitative"]
    assert "shape (2, 2, 2, 2)" in skipped[0]["reason"]
    assert list(tmp_path.glob("*.png")) == []
    assert "plot" not in envelope["meta"]


def test_run_stamps_verdicts_and_writes_plots(two_check_run):
    cka = [e for e in two_check_run["reports"] if e["check"] == "cka_matrix"]
    assert len(cka[0]["meta"]["plot"]) == 3  # full / trimmed / diff
    assert cka[0]["meta"]["panel"] == "cka"
    for e in two_check_run["reports"]:
        for m in e["metrics"].values():
            assert (m["bracket"], m["verdict"], m["red"]) == (None, None, False)


# --- end to end ------------------------------------------------------------


def test_ladder_then_run_then_render_through_both_clis(ladder_dir, tmp_path):
    from omegaconf import OmegaConf

    from video_encoding.scripts import render_report

    ladder_out, mesh_shared = ladder_dir
    model_config = tmp_path / "model.yaml"
    OmegaConf.save(_cfg(), model_config)
    out_dir = tmp_path / "out"
    run_diagnostics.main(
        [
            "--run-ids",
            "round-fixture",
            "--checks",
            "diag_depth,attention_entropy_by_block",
            "--mesh-shared",
            str(mesh_shared),
            "--shard-path",
            str(_shard(tmp_path)),
            "--out-dir",
            str(out_dir),
            "--ladder-path",
            str(ladder_out / "ladder.json"),
            "--model-config",
            str(model_config),
            "--n-windows",
            "2",
        ]
    )
    render_report.main(["--diagnostics", str(out_dir / "diagnostics.json")])

    html = (out_dir / "report.html").read_text()
    assert "<h1>Verdict: GATE_NOT_MET</h1>" in html
    for scheme in ("http://", "https://", "file://"):
        assert scheme not in html
    srcs = re.findall(r'<img[^>]*\ssrc="([^"]*)"', html)
    assert len(srcs) == 3  # the three cka_matrix heatmaps
    assert all(src.startswith("data:image/png;base64,") for src in srcs)
    # the run's attention entropy was tiered against the ladder on record
    assert "attention_entropy_by_block.real.block0" in html
    assert '<tr class="tier-C"><td>attention_entropy_by_block.real.block0' in html


# --- diag_diloco registration and the round axis's DiLoCo inputs -----------

DILOCO_NAMES = {
    "agreement_matrix",
    "effective_replicas",
    "per_block_divergence",
    "merge_magnitude",
    "momentum_alignment",
    "merge_regression",
    "counterfactual_sweep",
}


def test_diag_diloco_is_registered_with_every_name_controlled():
    from video_encoding.scripts import diag_diloco
    from video_encoding.scripts.render_report import CONTROLLED, PANEL_OF

    assert run_diagnostics.CHECKS["diag_diloco"] is diag_diloco.check
    assert run_diagnostics.EMITS["diag_diloco"] == DILOCO_NAMES
    assert DILOCO_NAMES <= CONTROLLED
    assert {PANEL_OF[name] for name in DILOCO_NAMES} == {"diloco"}
    for registry in (
        run_diagnostics.PAIRED,
        run_diagnostics.WEIGHTS_ONLY,
        run_diagnostics.GATES,
    ):
        assert "diag_diloco" not in registry


def test_a_merge_regression_cell_renders_unstruck(tmp_path):
    diagnostics = {
        "schema": run_diagnostics.SCHEMA,
        "ladder_revision": LADDER_REVISION,
        "ladder_tiers": {},
        "hyperparams": {},
        "reports": [_envelope(_sampled(0.2, 0.05), check="merge_regression")],
        "errors": [],
        "skipped": [],
    }
    html = render(diagnostics, tmp_path)
    row = re.search(r"<tr[^>]*><td>merge_regression\.real\.m</td>.*?</tr>", html)
    assert row is not None
    assert "<s>" not in row.group(0)


def test_run_axis_skips_diag_diloco_cleanly(tmp_path):
    diagnostics = _run(tmp_path, checks=("diag_diloco",))
    assert diagnostics["errors"] == []
    skipped = [s for s in diagnostics["skipped"] if s["check"] in DILOCO_NAMES]
    assert {s["check"] for s in skipped} == DILOCO_NAMES
    assert {s["reason"] for s in skipped} == {"no diloco snapshot supplied"}


def _configs(tmp_path, **entry):
    path = tmp_path / "mesh_run_configs.json"
    path.write_text(json.dumps([{"run_id": "r1", **entry}]))
    return path


def test_diloco_inputs_keys_and_snapshot_path(tmp_path):
    cfg = _cfg()
    inputs = run_diagnostics.diloco_inputs(
        "r1",
        5,
        snapshot_dir=tmp_path,
        configs_path=_configs(tmp_path, outer_lr=0.7, outer_momentum=0.9, weighting="token"),
        eval_shard=None,
        assignments_path=None,
        model_cfg=cfg,
        n_replicates=3,
    )
    assert inputs["diloco_snapshot_dir"] == tmp_path / "r1"
    assert inputs["diloco_round"] == 5
    assert (inputs["diloco_outer_lr"], inputs["diloco_outer_momentum"]) == (0.7, 0.9)
    assert inputs["diloco_weighting"] == "token"
    assert inputs["model_cfg"] is cfg
    assert inputs["diloco_n_replicates"] == 3
    assert "diloco_eval_shard" not in inputs
    assert "diloco_assignments" not in inputs


def test_rounds_refuse_diag_diloco_without_outer_hyperparameters(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        run_diagnostics, "load_snapshot_bundle", lambda *a, **k: calls.append(a)
    )
    snapshot_dir = _snapshots(tmp_path, _cfg(), [5])
    with pytest.raises(ValueError, match=r"r1.*outer_lr"):
        run_diagnostics.run_rounds(
            run_id="r1",
            rounds=[5],
            snapshot_dir=snapshot_dir,
            checks=["diag_diloco"],
            shard_path=_shard(tmp_path),
            model_cfg=_cfg(),
            out_dir=tmp_path / "out",
            ladder_path=None,
            skip_ladder_check=True,
            configs_path=_configs(tmp_path),
            n_windows=2,
        )
    assert calls == []


def test_round_axis_writes_diloco_reports_end_to_end(tmp_path):
    import torch

    from video_encoding.models.videomaev2.mae import VideoMAEForPreTraining
    from video_encoding.tests.test_diag_diloco import _model_payload, _write_snapshot

    cfg = _cfg()
    torch.manual_seed(0)
    master = VideoMAEForPreTraining(cfg).state_dict()
    nodes = {
        n: _model_payload(
            {
                k: v + 0.01 * s * torch.ones_like(v) if v.is_floating_point() else v
                for k, v in master.items()
            }
        )
        for s, n in enumerate(("a", "b"), start=1)
    }
    run_dir = tmp_path / "snapshots" / "r1"
    run_dir.mkdir(parents=True)
    _write_snapshot(run_dir, "r1", 4, nodes, master=master)

    run_diagnostics.run_rounds(
        run_id="r1",
        rounds=[4],
        snapshot_dir=tmp_path / "snapshots",
        checks=["diag_diloco"],
        shard_path=_shard(tmp_path),
        model_cfg=cfg,
        out_dir=tmp_path / "out",
        ladder_path=None,
        skip_ladder_check=True,
        configs_path=_configs(tmp_path, outer_lr=0.7, outer_momentum=0.9, weighting="token"),
        n_windows=2,
    )
    diagnostics = json.loads((tmp_path / "out" / "diagnostics.json").read_text())
    assert diagnostics["errors"] == []
    agreement = [e for e in diagnostics["reports"] if e["check"] == "agreement_matrix"]
    assert [(e["run_id"], e["round"]) for e in agreement] == [("r1", 4)]
    assert agreement[0]["meta"]["panel"] == "diloco"
    skipped = {s["check"]: s["reason"] for s in diagnostics["skipped"]}
    assert skipped["merge_regression"] == "no diloco eval shard supplied"
    assert skipped["counterfactual_sweep"] == "no diloco eval shard supplied"


def test_diloco_inputs_refuse_one_eval_flag_without_the_other(tmp_path):
    for shard, assignments in ((tmp_path / "val" / "s.tar", None), (None, tmp_path / "a.json")):
        with pytest.raises(ValueError, match="together"):
            run_diagnostics.diloco_inputs(
                "r1",
                5,
                snapshot_dir=tmp_path,
                configs_path=_configs(
                    tmp_path, outer_lr=0.7, outer_momentum=0.9, weighting="data"
                ),
                eval_shard=shard,
                assignments_path=assignments,
                model_cfg=_cfg(),
                n_replicates=3,
            )


def test_diloco_inputs_require_the_run_weighting(tmp_path):
    with pytest.raises(ValueError, match="weighting"):
        run_diagnostics.diloco_inputs(
            "r1",
            5,
            snapshot_dir=tmp_path,
            configs_path=_configs(tmp_path, outer_lr=0.7, outer_momentum=0.9),
            eval_shard=None,
            assignments_path=None,
            model_cfg=_cfg(),
            n_replicates=3,
        )


def test_default_checks_skip_diag_diloco_without_hyperparameters(tmp_path):
    snapshot_dir = _snapshots(tmp_path, _cfg(), [5])
    run_diagnostics.run_rounds(
        run_id="r1",
        rounds=[5],
        snapshot_dir=snapshot_dir,
        checks=["diag_diloco"],
        shard_path=_shard(tmp_path),
        model_cfg=_cfg(),
        out_dir=tmp_path / "out",
        ladder_path=None,
        skip_ladder_check=True,
        n_windows=2,
        diloco_optional=True,
    )
    diagnostics = json.loads((tmp_path / "out" / "diagnostics.json").read_text())
    assert diagnostics["errors"] == []
    reasons = {s["check"]: s["reason"] for s in diagnostics["skipped"]}
    assert set(reasons) == DILOCO_NAMES
    assert {r for r in reasons.values()} == {
        "no outer hyperparameters for this run (--mesh-run-configs)"
    }


def test_cli_marks_diag_diloco_optional_only_when_checks_is_defaulted(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(
        run_diagnostics, "run_rounds", lambda **kw: seen.append(kw["diloco_optional"])
    )
    base = ["--run-ids", "r1", "--rounds", "5", "--snapshot-dir", str(tmp_path),
            "--shard-path", str(tmp_path / "s.tar"), "--out-dir", str(tmp_path / "o")]
    run_diagnostics.main(base)
    run_diagnostics.main(base + ["--checks", "diag_diloco"])
    assert seen == [True, False]


def test_cli_eval_flags_reach_the_forward_pass_end_to_end(tmp_path):
    """--diloco-eval-shard/--diloco-assignments through main() -> run_rounds ->
    diloco_inputs -> check -> real tiny-model forward passes."""
    import torch
    from omegaconf import OmegaConf

    from video_encoding.models.videomaev2.mae import VideoMAEForPreTraining
    from video_encoding.tests.test_diag_common import make_fake_shard
    from video_encoding.tests.test_diag_diloco import _model_payload, _write_snapshot

    cfg = _cfg()
    cfg_path = tmp_path / "model.yaml"
    OmegaConf.save(cfg, cfg_path)
    torch.manual_seed(0)
    master = VideoMAEForPreTraining(cfg).state_dict()
    nodes = {
        n: _model_payload(
            {k: v + 0.01 * s * torch.ones_like(v) for k, v in master.items()},
            n_samples=100 * s,
        )
        for s, n in enumerate(("a", "b"), start=1)
    }
    run_dir = tmp_path / "snapshots" / "r1"
    run_dir.mkdir(parents=True)
    _write_snapshot(run_dir, "r1", 4, nodes, master=master)
    (tmp_path / "val").mkdir()
    eval_shard = make_fake_shard(tmp_path / "val", {"c0": 8, "c1": 8})
    assignments = tmp_path / "shard_assignments_r1.json"
    assignments.write_text(json.dumps({"a": ["t0.tar"], "b": ["t1.tar"]}))

    run_diagnostics.main(
        [
            "--run-ids", "r1", "--rounds", "4",
            "--snapshot-dir", str(tmp_path / "snapshots"),
            "--shard-path", str(_shard(tmp_path)),
            "--out-dir", str(tmp_path / "out"),
            "--skip-ladder-check",
            "--checks", "diag_diloco",
            "--n-windows", "2",
            "--model-config", str(cfg_path),
            "--mesh-run-configs",
            str(_configs(tmp_path, outer_lr=0.7, outer_momentum=0.9, weighting="data")),
            "--diloco-eval-shard", str(eval_shard),
            "--diloco-assignments", str(assignments),
        ]
    )
    diagnostics = json.loads((tmp_path / "out" / "diagnostics.json").read_text())
    assert diagnostics["errors"] == []
    by_check = {e["check"]: e for e in diagnostics["reports"]}
    assert by_check["merge_regression"]["metrics"]["vs_alpha_mean"]["n"] == 3
    assert by_check["counterfactual_sweep"]["meta"]["configured_cell"]["weighting"] == "data"
    assert by_check["merge_regression"]["meta"]["eval_shard"] == str(eval_shard)


# --- AFEW-VA gate wiring (ssl-diag-08) ---------------------------------------


def test_the_afewva_gate_is_the_registered_gate():
    from video_encoding.scripts import diag_finetune_afewva
    from video_encoding.scripts.render_report import CONTROLLED

    assert run_diagnostics.GATES == frozenset({"finetune_afewva"})
    assert run_diagnostics.CHECKS["finetune_afewva"] is diag_finetune_afewva.check
    assert run_diagnostics.EMITS["finetune_afewva"] == frozenset({"finetune_afewva"})
    assert "finetune_afewva" in CONTROLLED


def test_intrinsic_checks_alone_never_pass_the_gate(tmp_path, monkeypatch):
    from video_encoding.scripts.render_report import verdict

    monkeypatch.delenv("AFEWVA_ROOT", raising=False)
    diagnostics = _run(tmp_path, checks=("diag_depth",))

    gates = [e for e in diagnostics["reports"] if e["check"] == "finetune_afewva"]
    assert len(gates) == 1
    assert gates[0]["meta"]["status"] == "GATE_NOT_MET"
    assert not [e for e in diagnostics["errors"] if e["check"] == "finetune_afewva"]
    assert verdict(diagnostics["reports"]) == "GATE_NOT_MET"


def _spy_gate_inputs(monkeypatch) -> list[dict]:
    seen: list[dict] = []

    def _spy(bundle, inputs):
        seen.append(inputs)
        return {}

    monkeypatch.setitem(run_diagnostics.CHECKS, "finetune_afewva", _spy)
    return seen


def test_run_threads_the_gate_parameters(tmp_path, monkeypatch):
    seen = _spy_gate_inputs(monkeypatch)
    _run(tmp_path, checks=("diag_depth",), min_epochs=4, gate_seeds=5)

    (inputs,) = seen
    gate = inputs["gate"]
    assert gate["min_epochs"] == 4 and gate["n_seeds"] == 5
    assert gate["cache_dir"] == tmp_path / "out" / "gate_cache"
    assert gate["model_cfg"] == _cfg()
    assert inputs["control_bundle"] is not None


def test_run_without_a_ceiling_checkpoint_is_not_an_error(tmp_path, monkeypatch):
    seen = _spy_gate_inputs(monkeypatch)
    diagnostics = _run(tmp_path, checks=("diag_depth",))

    assert seen[0]["gate"]["ceiling_bundle"] is None
    assert diagnostics["errors"] == []


def test_an_unloadable_ceiling_is_absent_not_an_error(tmp_path, monkeypatch):
    seen = _spy_gate_inputs(monkeypatch)
    diagnostics = _run(
        tmp_path,
        checks=("diag_depth",),
        public_videomaev2_ckpt=tmp_path / "missing.pth",
    )

    assert seen[0]["gate"]["ceiling_bundle"] is None
    assert diagnostics["errors"] == []


def test_the_round_axis_never_fine_tunes(tmp_path, monkeypatch):
    afewva = tmp_path / "afewva"
    afewva.mkdir()
    (afewva / "manifest.csv").write_text("")
    monkeypatch.setenv("AFEWVA_ROOT", str(afewva))
    snapshot_dir = _snapshots(tmp_path, _cfg(), [4, 5])

    diagnostics = _run_rounds(tmp_path, [4, 5], snapshot_dir)

    gates = [e for e in diagnostics["reports"] if e["check"] == "finetune_afewva"]
    assert len(gates) == 2
    for e in gates:
        assert e["meta"]["status"] == "GATE_NOT_MET"
        assert "run axis" in e["meta"]["reason"]


def test_an_incomparable_gate_cache_is_an_error_and_not_a_pass(tmp_path, monkeypatch):
    from video_encoding.scripts.render_report import verdict

    afewva = tmp_path / "afewva"
    afewva.mkdir()
    (afewva / "manifest.csv").write_text("")
    monkeypatch.setenv("AFEWVA_ROOT", str(afewva))
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "gate_params.json").write_text(
        json.dumps({"min_epochs": 10, "gate_seeds": 3, "split_fraction": 0.2})
    )

    diagnostics = _run(
        tmp_path, checks=("diag_depth",), min_epochs=5, gate_cache_dir=cache
    )

    (error,) = [e for e in diagnostics["errors"] if e["check"] == "finetune_afewva"]
    assert "min_epochs" in error["reason"]
    assert verdict(diagnostics["reports"]) == "GATE_NOT_MET"


def test_cli_threads_the_gate_flags(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(run_diagnostics, "run", lambda **kw: seen.update(kw))
    run_diagnostics.main(
        [
            "--run-ids", "r1",
            "--shard-path", str(tmp_path),
            "--out-dir", str(tmp_path / "out"),
            "--min-epochs", "7",
            "--gate-seeds", "4",
            "--gate-cache-dir", str(tmp_path / "cache"),
            "--public-videomaev2-ckpt", str(tmp_path / "v2.pth"),
        ]
    )  # fmt: skip
    assert seen["min_epochs"] == 7 and seen["gate_seeds"] == 4
    assert seen["gate_cache_dir"] == tmp_path / "cache"
    assert seen["public_videomaev2_ckpt"] == tmp_path / "v2.pth"
