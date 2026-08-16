from __future__ import annotations

import pytest
import torch
from torch import nn

from video_encoding.models.ViT.attention import SelfAttention


def test_self_attention_matches_multihead_attention_given_same_weights():
    """Given identical weights, SelfAttention(qk_norm=False) must compute
    bit-close-identical output to nn.MultiheadAttention -- the guarantee
    that lets ViTBlock's default config reproduce today's exact numerics
    despite the hand-written attention rewrite."""
    torch.manual_seed(0)
    embed_dim, num_heads, seq_len, batch = 16, 4, 5, 2
    reference = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    ours = SelfAttention(embed_dim, num_heads, qk_norm=False)

    with torch.no_grad():
        ours.qkv.weight.copy_(reference.in_proj_weight)
        ours.qkv.bias.copy_(reference.in_proj_bias)
        ours.proj.weight.copy_(reference.out_proj.weight)
        ours.proj.bias.copy_(reference.out_proj.bias)

    x = torch.randn(batch, seq_len, embed_dim)
    reference.eval()
    ours.eval()
    with torch.no_grad():
        ref_out, _ = reference(x, x, x, need_weights=False)
        our_out, _ = ours(x, need_weights=False)

    assert torch.allclose(ref_out, our_out, atol=1e-5)


def test_self_attention_need_weights_returns_valid_softmax_rows():
    attn = SelfAttention(embed_dim=16, num_heads=4)
    x = torch.randn(2, 5, 16)
    out, weights = attn(x, need_weights=True)
    assert out.shape == (2, 5, 16)
    assert weights.shape == (2, 4, 5, 5)  # (B, heads, N, N)
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_qk_norm_changes_output_versus_no_qk_norm():
    torch.manual_seed(0)
    x = torch.randn(2, 5, 16)

    torch.manual_seed(1)
    plain = SelfAttention(embed_dim=16, num_heads=4, qk_norm=False)
    torch.manual_seed(1)
    normed = SelfAttention(embed_dim=16, num_heads=4, qk_norm=True)

    out_plain, _ = plain(x, need_weights=False)
    out_normed, _ = normed(x, need_weights=False)
    assert not torch.allclose(out_plain, out_normed)


def test_rejects_embed_dim_not_divisible_by_num_heads():
    with pytest.raises(ValueError, match="not divisible"):
        SelfAttention(embed_dim=15, num_heads=4)
