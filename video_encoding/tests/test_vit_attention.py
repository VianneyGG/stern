from __future__ import annotations

import math

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


def test_qk_norm_logit_scale_is_a_learnable_parameter_init_to_sqrt_head_dim():
    """QK-norm bounds q.k to [-1, 1]; without a compensating logit scale the
    softmax over N tokens is flat regardless of content. The learnable
    per-head scale is what restores usable logit range, so its presence and
    its sqrt(head_dim) init are part of the contract, not an implementation
    detail."""
    attn = SelfAttention(embed_dim=256, num_heads=4, qk_norm=True)
    assert isinstance(attn.logit_scale, nn.Parameter)
    assert attn.logit_scale.requires_grad is True
    assert attn.logit_scale.shape == (4, 1, 1)  # per-head
    assert attn.logit_scale.detach().unique().tolist() == [8.0]  # sqrt(256 / 4)


def test_qk_norm_attention_is_not_collapsed_to_uniform():
    """Regression guard for the bug this replaced: multiplying L2-normalised
    q.k by head_dim**-0.5 left logits in [-0.125, 0.125], which softmax over
    many tokens flattens to uniform -- i.e. the ablation axis measured
    "attention disabled" rather than "QK-normed ViT". A mere output-differs
    assertion passes in that broken state; this pins the distribution.

    Measured at this geometry: broken 1.04x uniform, fixed 10.3x."""
    torch.manual_seed(0)
    embed_dim, num_heads, n_tokens = 256, 4, 256
    attn = SelfAttention(embed_dim, num_heads, qk_norm=True).eval()
    x = torch.nn.functional.layer_norm(
        torch.randn(1, n_tokens, embed_dim), (embed_dim,)
    )

    with torch.no_grad():
        _, weights = attn(x, need_weights=True)

    uniform = 1.0 / n_tokens
    peak_vs_uniform = (weights.max(dim=-1).values.mean() / uniform).item()
    assert peak_vs_uniform > 3.0, (
        f"attention collapsed toward uniform: peak is only "
        f"{peak_vs_uniform:.2f}x the uniform weight {uniform}"
    )

    entropy = -(weights * (weights + 1e-12).log()).sum(dim=-1).mean().item()
    max_entropy = math.log(n_tokens)
    assert entropy < 0.95 * max_entropy, (
        f"attention entropy {entropy:.4f} is within 5% of the uniform "
        f"maximum {max_entropy:.4f}"
    )


def test_qk_norm_off_has_no_logit_scale_parameter():
    """The default path must stay byte-identical to nn.MultiheadAttention --
    an unused extra Parameter would also change every checkpoint's key set."""
    attn = SelfAttention(embed_dim=16, num_heads=4, qk_norm=False)
    assert "logit_scale" not in dict(attn.named_parameters())


def test_rejects_embed_dim_not_divisible_by_num_heads():
    with pytest.raises(ValueError, match="not divisible"):
        SelfAttention(embed_dim=15, num_heads=4)
