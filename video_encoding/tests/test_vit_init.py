from __future__ import annotations

import torch

from video_encoding.models.ViT.init import (
    cutting_skip_attention_init,
    cutting_skip_logits,
)


def test_cutting_skip_logits_are_diagonally_dominant():
    torch.manual_seed(0)
    d = 16
    alpha = 0.1
    beta = 5.0
    q = torch.randn(d, d) / d**0.5
    k = torch.randn(d, d) / d**0.5

    logits = cutting_skip_logits(q, k, alpha=alpha, beta=beta)

    diag = logits.diag()
    off_diag = logits - torch.diag_embed(diag)
    assert float(diag.mean()) > 0.0
    assert float((off_diag.abs().mean())) < float(diag.abs().mean())


def test_cutting_skip_attention_init_sets_qk_and_vo_targets():
    torch.manual_seed(0)
    attn = cutting_skip_attention_init(
        embed_dim=16,
        num_heads=4,
        alpha=0.1,
        beta=5.0,
        c=0.5,
    )

    qkv = attn.qkv.weight
    q, k, v = qkv.chunk(3, dim=0)
    proj = attn.proj.weight

    qk_logits = cutting_skip_logits(q, k, alpha=0.1, beta=5.0)
    diag = qk_logits.diag()
    off_diag = qk_logits - torch.diag_embed(diag)

    assert q.shape == (16, 16)
    assert k.shape == (16, 16)
    assert v.shape == (16, 16)
    assert proj.shape == (16, 16)
    assert float(diag.abs().mean()) > float(off_diag.abs().mean())
