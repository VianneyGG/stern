from __future__ import annotations

import torch

from video_encoding.models.ViT.registers import RegisterTokens


def test_zero_registers_is_a_no_op():
    reg = RegisterTokens(embed_dim=8, num_registers=0)
    x = torch.randn(2, 5, 8)
    assert torch.equal(reg.prepend(x), x)
    assert reg.prepend(x).sub(x).abs().max().item() == 0.0
    assert torch.equal(reg.strip(x), x)
    assert reg.strip(x).sub(x).abs().max().item() == 0.0


def test_prepend_adds_register_count_and_strip_undoes_it():
    reg = RegisterTokens(embed_dim=8, num_registers=3)
    x = torch.randn(2, 5, 8)

    prepended = reg.prepend(x)
    assert prepended.shape == (2, 8, 8)  # 5 patch tokens + 3 registers

    stripped = reg.strip(prepended)
    assert stripped.shape == (2, 5, 8)
    assert torch.equal(stripped, x)  # round-trip is exact -- registers are prefixed
    assert stripped.sub(x).abs().max().item() == 0.0


def test_prepend_places_registers_first():
    reg = RegisterTokens(embed_dim=4, num_registers=2)
    x = torch.randn(1, 3, 4)
    prepended = reg.prepend(x)
    assert torch.equal(prepended[:, :2, :], reg.tokens)
    assert prepended[:, :2, :].sub(reg.tokens).abs().max().item() == 0.0
    assert torch.equal(prepended[:, 2:, :], x)
    assert prepended[:, 2:, :].sub(x).abs().max().item() == 0.0
