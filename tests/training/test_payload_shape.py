"""Locks in the exact checkpoint payload shape for both training stages.

training/train.py and training/pretrain.py both moved their `_payload()`
builders onto the shared `checkpoint.base_payload()` helper. This test is
the safety net for that refactor: an in-flight SSL run on the cluster may
have checkpoints on disk, and `checkpoint.load()`'s `expected_model_name`/
`expected_optimizer_name` checks depend on these exact string literals and
key sets not changing.
"""

from __future__ import annotations

import torch
from torch import nn

from stern.training.pretrain import _payload as ssl_payload
from stern.training.finetune import _payload as probe_payload


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(2, 2)


def test_probe_payload_shape():
    model = _TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    payload = probe_payload(model=model, optimizer=optimizer, epoch=3, best_ccc=0.5)

    assert set(payload.keys()) == {
        "model_state_dict",
        "optimizer_state_dict",
        "optimizer_name",
        "epoch",
        "best_val_metric",
        "model_name",
    }
    assert payload["optimizer_name"] == "adam"
    assert payload["model_name"] == "videomae_va_regressor"
    assert payload["epoch"] == 3
    assert payload["best_val_metric"] == 0.5


def test_ssl_payload_shape():
    model = _TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    payload = ssl_payload(
        model=model, optimizer=optimizer, scheduler=scheduler, epoch=7, best_loss=0.1
    )

    assert set(payload.keys()) == {
        "model_state_dict",
        "optimizer_state_dict",
        "optimizer_name",
        "scheduler_state_dict",
        "epoch",
        "best_val_metric",
        "model_name",
    }
    assert payload["optimizer_name"] == "adamw"
    assert payload["model_name"] == "videomae_for_pretraining"
    assert payload["scheduler_state_dict"] is not None
    assert payload["epoch"] == 7
    assert payload["best_val_metric"] == 0.1


def test_ssl_payload_scheduler_state_dict_none_when_no_scheduler():
    model = _TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    payload = ssl_payload(
        model=model, optimizer=optimizer, scheduler=None, epoch=0, best_loss=1.0
    )

    assert payload["scheduler_state_dict"] is None
