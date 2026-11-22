"""Weight initialisation for the ViT, including the Cutting-the-Skip attention init."""

from __future__ import annotations

import math

import torch
from torch import nn


def _topk_svd(
    m: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Top-`k` singular triplet of `m`, via a randomised solver.

    A full `torch.linalg.svd` of a 768x768 matrix per head per layer costs
    minutes of startup on a 12x12 model, paid again by every mesh worker.
    `svd_lowrank` returns the same leading subspace for a fraction of that.
    """
    u, s, v = torch.svd_lowrank(m, q=min(k + 8, m.shape[-1]), niter=4)
    return u[:, :k], s[:k], v[:, :k]


def cutting_skip_qk_head_factors(
    dim: int,
    head_dim: int,
    *,
    alpha: float = 1.8,
    beta: float = 1.0,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One head's `(head_dim, dim)` W_Q / W_K slices targeting M = alpha*Z + beta*I.

    The head's logit matrix is `W_Q[h].T @ W_K[h]` -- `nn.Linear` holds weights
    as `(out, in)` and applies `x @ W.T`, so it is this orientation, not
    `W_Q @ W_K.T`, that reaches the softmax. Being `(head_dim, dim)` slices,
    their product has rank `head_dim`, so a full-rank `M` is unreachable: the
    best available target is `M`'s rank-`head_dim` truncation, which is what
    the truncated SVD below constructs.
    """
    device = device or torch.device("cpu")
    dtype = dtype or torch.float32
    z = torch.randn(dim, dim, device=device, dtype=dtype) / math.sqrt(dim)
    target = alpha * z + beta * torch.eye(dim, device=device, dtype=dtype)
    u, s, v = _topk_svd(target, head_dim)
    root = torch.diag(torch.sqrt(s.clamp_min(1e-8)))
    return (u @ root).T, (v @ root).T


def cutting_skip_vo_factors(
    dim: int,
    *,
    c: float = 3.0,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """W_V, W_O as scaled orthonormal factors, giving kappa(W_O W_V) = 1.

    Sample Q ~ N(0, 1/d), take Q = U S V^T, keep `c*U` and `c*V^T` and discard
    S. Unlike the Q/K half this is orientation-independent: the product is
    orthonormal either way round, so `nn.Linear`'s transpose does not disturb
    the conditioning.
    """
    device = device or torch.device("cpu")
    dtype = dtype or torch.float32
    q = torch.randn(dim, dim, device=device, dtype=dtype) / math.sqrt(dim)
    u, _, vh = torch.linalg.svd(q, full_matrices=False)
    return c * u, c * vh


def apply_cutting_skip_to_attention(
    attn: nn.Module,
    *,
    alpha: float = 1.8,
    beta: float = 1.0,
    c: float = 3.0,
) -> None:
    """Initialise an existing SelfAttention in place, per head.

    Must run against a whole attention module rather than through
    `Module.apply`: the construction couples Q with K and V with O, and the
    fused `qkv` Linear is `(3*dim, dim)`, so a per-`nn.Linear` walk can see
    neither pairing.
    """
    dim = attn.qkv.in_features
    head_dim = dim // attn.num_heads
    device, dtype = attn.qkv.weight.device, attn.qkv.weight.dtype
    with torch.no_grad():
        w_q = torch.empty(dim, dim, device=device, dtype=dtype)
        w_k = torch.empty(dim, dim, device=device, dtype=dtype)
        for h in range(attn.num_heads):
            rows = slice(h * head_dim, (h + 1) * head_dim)
            q_h, k_h = cutting_skip_qk_head_factors(
                dim, head_dim, alpha=alpha, beta=beta, device=device, dtype=dtype
            )
            w_q[rows], w_k[rows] = q_h, k_h
        w_v, w_o = cutting_skip_vo_factors(dim, c=c, device=device, dtype=dtype)
        attn.qkv.weight.copy_(torch.cat([w_q, w_k, w_v], dim=0))
        attn.proj.weight.copy_(w_o)
        nn.init.zeros_(attn.qkv.bias)
        nn.init.zeros_(attn.proj.bias)


def cutting_skip_attention_init(
    *,
    embed_dim: int,
    num_heads: int,
    alpha: float = 1.8,
    beta: float = 1.0,
    c: float = 3.0,
):
    """Build a fresh SelfAttention carrying the Cutting-the-Skip init."""
    from stern.models.vit.attention import SelfAttention

    attn = SelfAttention(embed_dim=embed_dim, num_heads=num_heads)
    apply_cutting_skip_to_attention(attn, alpha=alpha, beta=beta, c=c)
    return attn


def init_module(module: nn.Module, method: str = "trunc_normal", std: float = 0.02):
    """Initialise one Linear, LayerNorm or Conv3d in place, using `method`."""
    if isinstance(module, nn.Linear):
        if method == "trunc_normal":
            nn.init.trunc_normal_(module.weight, std=std)
        elif method == "xavier_uniform":
            nn.init.xavier_uniform_(module.weight)
        elif method == "kaiming_uniform":
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        elif method == "normal":
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif method == "cutting_skip":
            # Attention weights are set separately, per block, by
            # apply_cutting_skip_to_attention; everything else (MLP,
            # projections) keeps the default.
            nn.init.trunc_normal_(module.weight, std=std)
        else:
            raise ValueError(f"unknown init method: {method}")

        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Conv3d):
        if method == "trunc_normal":
            nn.init.trunc_normal_(module.weight, std=std)
        elif method == "kaiming_uniform":
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        else:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def apply_weight_init(
    module: nn.Module,
    *,
    method: str = "trunc_normal",
    std: float = 0.02,
) -> None:
    """Apply a selected init policy while keeping the default behavior stable."""
    if isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    else:
        init_module(module, method=method, std=std)


__all__ = [
    "apply_cutting_skip_to_attention",
    "apply_weight_init",
    "cutting_skip_attention_init",
    "cutting_skip_qk_head_factors",
    "cutting_skip_vo_factors",
    "init_module",
]
