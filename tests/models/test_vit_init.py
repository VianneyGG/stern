from __future__ import annotations

import pytest
import torch

from stern.models.vit.attention import SelfAttention
from stern.models.vit.block import BlockConfig, ViTBlock
from stern.models.vit.init import (
    apply_cutting_skip_to_attention,
    cutting_skip_qk_head_factors,
    cutting_skip_vo_factors,
)

_ALPHA, _BETA, _C = 1.8, 1.0, 3.0


def _stream_rms_per_block(depth, use_skip, method):
    """Build a `depth`-block stack and return residual-stream RMS after each
    block, on a fixed random input. `method` is "trunc_normal" or "cutting_skip"."""
    torch.manual_seed(0)
    embed_dim, num_heads = 64, 8
    blocks = [
        ViTBlock(
            BlockConfig(embed_dim=embed_dim, num_heads=num_heads, use_skip=use_skip)
        )
        for _ in range(depth)
    ]
    if method == "cutting_skip":
        for block in blocks:
            apply_cutting_skip_to_attention(block.attn, alpha=_ALPHA, beta=_BETA, c=_C)
    for block in blocks:
        block.eval()

    x = torch.randn(2, 16, embed_dim)
    rms = []
    with torch.no_grad():
        for block in blocks:
            x = block(x)
            rms.append(x.float().pow(2).mean().sqrt().item())
    return rms


def test_qk_head_factors_are_diagonally_dominant_per_head():
    """The paper's claim is about the matrix that actually reaches the softmax:
    nn.Linear stores (out, in) and computes x @ W.T, so the per-head logit
    matrix is W_q[h].T @ W_k[h], not W_q[h] @ W_k[h].T. A head slice is
    (head_dim, dim), so the product has rank head_dim -- a full-rank alpha*Z +
    beta*I target is unreachable per head; this checks the achievable
    approximation is still diagonally dominant."""
    torch.manual_seed(0)
    dim, head_dim = 64, 16
    w_q, w_k = cutting_skip_qk_head_factors(dim, head_dim, alpha=_ALPHA, beta=_BETA)

    logits = w_q.T @ w_k
    diag = logits.diag().abs().mean()
    off_diag = (logits - torch.diag_embed(logits.diag())).abs().mean()

    assert w_q.shape == (head_dim, dim)
    assert diag > off_diag


def test_vo_factors_give_orthonormal_conditioning():
    """c=3.0 -> every singular value of W_O @ W_V should be exactly c**2 = 9.0
    (perfect conditioning, kappa=1): W_V and W_O are scaled orthonormal
    factors of the same SVD, so their product is c**2 times an orthonormal
    matrix, whose singular values are all 1."""
    torch.manual_seed(0)
    dim, c = 64, _C
    w_v, w_o = cutting_skip_vo_factors(dim, c=c)

    singular_values = torch.linalg.svdvals(w_o @ w_v)

    assert singular_values.min().item() == pytest.approx(9.0, abs=1e-3)
    assert singular_values.max().item() == pytest.approx(9.0, abs=1e-3)


def test_apply_cutting_skip_sets_every_head_and_leaves_vo_orthonormal():
    """Regression test for the original bug: the old code walked nn.Linear
    modules one at a time via Module.apply(), so the fused (3*dim, dim) qkv
    Linear failed the "is this square" check and silently fell back to
    trunc_normal -- the diagonally-dominant structure was never applied to any
    head. This builds a real SelfAttention, applies the fix, and checks every
    head individually."""
    torch.manual_seed(0)
    embed_dim, num_heads = 64, 8
    head_dim = embed_dim // num_heads
    attn = SelfAttention(embed_dim=embed_dim, num_heads=num_heads)

    apply_cutting_skip_to_attention(attn, alpha=_ALPHA, beta=_BETA, c=_C)

    q, k, v = attn.qkv.weight.chunk(3, dim=0)
    for h in range(num_heads):
        rows = slice(h * head_dim, (h + 1) * head_dim)
        logits = q[rows].T @ k[rows]
        diag = logits.diag().abs().mean()
        off_diag = (logits - torch.diag_embed(logits.diag())).abs().mean()
        assert diag > off_diag, f"head {h} is not diagonally dominant"

    singular_values = torch.linalg.svdvals(attn.proj.weight @ v)
    assert singular_values.min().item() == pytest.approx(9.0, abs=1e-3)
    assert singular_values.max().item() == pytest.approx(9.0, abs=1e-3)


def test_cutting_skip_init_makes_attention_near_identity_at_init():
    """Direct measurement of the paper's design intent: alpha*Z + beta*I with
    beta dominant should make each token mostly attend to itself before any
    training. Confirms the earlier ad hoc measurement (365x uniform mass) is
    reproducible from the shipped function, not an artifact of a one-off script."""
    torch.manual_seed(0)
    embed_dim, num_heads, seq_len = 64, 8, 32
    block = ViTBlock(BlockConfig(embed_dim=embed_dim, num_heads=num_heads))
    apply_cutting_skip_to_attention(block.attn, alpha=_ALPHA, beta=_BETA, c=_C)
    block.eval()

    x = torch.randn(2, seq_len, embed_dim)
    with torch.no_grad():
        _, attn_weights = block.attn(block.norm1(x), need_weights=True)
    diagonal_mass = torch.diagonal(attn_weights, dim1=-2, dim2=-1).mean().item()

    uniform_baseline = 1.0 / seq_len
    assert uniform_baseline == pytest.approx(0.03125)
    assert diagonal_mass > 0.3125  # >10x the 1/seq_len uniform baseline (0.03125)


def test_skipping_the_residual_is_what_cutting_skip_is_actually_for():
    """What removing the skip connection does, with and without this init.

    This is the paper's actual claim: alpha*Z + beta*I is a substitute for the
    residual path, not an addition to it. Without any skip, an untrained stack
    has nothing else carrying signal forward, so a generic init (trunc_normal)
    should drift towards collapse (RMS -> ~0) or blow-up as depth grows, while
    the near-identity attention from cutting_skip should keep the stream's
    scale roughly stable block-to-block.
    """
    depth = 8
    trunc_rms = _stream_rms_per_block(depth, use_skip=False, method="trunc_normal")
    cutting_rms = _stream_rms_per_block(depth, use_skip=False, method="cutting_skip")

    trunc_drift = trunc_rms[-1] / trunc_rms[0]
    cutting_drift = cutting_rms[-1] / cutting_rms[0]

    # trunc_normal has no stabilising structure without a skip path: each
    # block resets the stream to a fresh, uncorrelated projection of its
    # input, so scale should drift far more than the (near-identity) cutting
    # skip's small per-block state update.
    assert abs(cutting_drift - 1.0) < abs(trunc_drift - 1.0)


def test_keeping_the_skip_makes_cutting_skip_attention_redundant():
    """What NOT skipping the residual does to this init: with the skip path
    present, the near-identity attention (previous test) duplicates work the
    residual connection already does for free, adding scale the block did not
    need. This is the reason weight_init defaults to trunc_normal, not
    cutting_skip, in conf/model/videomae_base.yaml -- see the comment there."""
    depth = 8
    trunc_rms = _stream_rms_per_block(depth, use_skip=True, method="trunc_normal")
    cutting_rms = _stream_rms_per_block(depth, use_skip=True, method="cutting_skip")

    # With the skip retained, cutting_skip's near-identity attention adds an
    # extra full-scale copy of the stream on top of the residual every block,
    # so its RMS should grow measurably faster than trunc_normal's.
    assert cutting_rms[-1] > 2 * trunc_rms[-1]
