import argparse
import json

import pytest

from video_encoding.scripts import diag_finetune_afewva as g
from video_encoding.scripts import gate_cli

# What the pre-flag gate_cli recorded in every existing cache's gate_params.json.
_HISTORICAL = {
    "min_epochs": g.GATE_MIN_EPOCHS,
    "gate_seeds": g.GATE_SEEDS,
    "split_fraction": g.SPLIT_FRACTION,
    "lr": g.GATE_LR,
    "head_bounded": g.GATE_HEAD_BOUNDED,
}


def _args(**kw) -> argparse.Namespace:
    base = dict(epochs=g.GATE_MIN_EPOCHS, seeds=g.GATE_SEEDS, head_lr_mult=1.0,
                head_token_norm=False)
    return argparse.Namespace(**{**base, **kw})


_SEEDED = {**_HISTORICAL, "rng_seeded": True}


def test_default_args_add_only_the_seed_fix_to_the_historical_params():
    assert gate_cli.gate_params(_args()) == _SEEDED


def test_a_cache_from_before_the_seed_fix_is_refused(tmp_path):
    # Its "seeds" repeated across parallel jobs; never extend it.
    (tmp_path / "gate_params.json").write_text(json.dumps(_HISTORICAL))
    with pytest.raises(ValueError, match="rng_seeded"):
        g._check_gate_params(tmp_path, gate_cli.gate_params(_args()))


def test_a_seeded_default_cache_accepts_default_args(tmp_path):
    (tmp_path / "gate_params.json").write_text(json.dumps(_SEEDED))
    g._check_gate_params(tmp_path, gate_cli.gate_params(_args()))  # no raise


@pytest.mark.parametrize(
    "change", [{"epochs": 10}, {"head_lr_mult": 100.0}, {"head_token_norm": True}]
)
def test_a_seeded_default_cache_refuses_a_changed_recipe(tmp_path, change):
    (tmp_path / "gate_params.json").write_text(json.dumps(_SEEDED))
    with pytest.raises(ValueError, match="differs"):
        g._check_gate_params(tmp_path, gate_cli.gate_params(_args(**change)))


def test_token_norm_reaches_the_in_house_model_cfg():
    cfg = gate_cli.model_cfg("inhouse", "16x224", ["+model.head_token_norm=true"])
    assert cfg.head_token_norm is True
    assert cfg.name == "levjepa_lean"
