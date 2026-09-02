from __future__ import annotations

import math

import torch
from torch import nn


def cutting_skip_logits(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    alpha: float = 0.1,
    beta: float = 5.0,
) -> torch.Tensor:
    """Return the mimetic logits M = alpha * Z + beta * I.

    This matches the paper's stated recipe: build a random Gaussian matrix Z,
    then add an identity term to make the logits diagonally dominant. The final
    logit matrix is not itself an SVD decomposition; it is the random logits
    term plus the identity bias.
    """
    if q.shape != k.shape:
        raise ValueError(f"q and k must share shape, got {q.shape} and {k.shape}")
    d = q.shape[-1]
    z = q @ k.T
    identity = torch.eye(d, device=q.device, dtype=q.dtype)
    return alpha * z + beta * identity


def _cutting_skip_qk_factors(
    dim: int,
    *,
    alpha: float = 0.1,
    beta: float = 5.0,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Factorize the target QK^T structure M = alpha * Z + beta * I.

    We construct the target logit matrix, then factorize it via SVD so that
    W_Q W_K^T = M exactly.
    """
    device = device or torch.device("cpu")
    dtype = dtype or torch.float32
    z = torch.randn(dim, dim, device=device, dtype=dtype) / math.sqrt(dim)
    target = alpha * z + beta * torch.eye(dim, device=device, dtype=dtype)
    u, s, vh = torch.linalg.svd(target, full_matrices=False)
    sqrt_s = torch.sqrt(s.clamp_min(1e-8))
    w_q = u @ torch.diag(sqrt_s)
    w_k = vh.T @ torch.diag(sqrt_s)
    return w_q, w_k


def _cutting_skip_vo_factors(
    dim: int,
    *,
    c: float = 0.5,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Initialize W_V and W_O from a Gaussian matrix via SVD.

    Per the paper: sample Q ~ N(0, 1/d), compute Q = U S V^T, then assign
    W_V = c * U and W_O = c * V^T. This makes the product a scaled orthonormal
    matrix in the intended sense.
    """
    device = device or torch.device("cpu")
    dtype = dtype or torch.float32
    q = torch.randn(dim, dim, device=device, dtype=dtype) / math.sqrt(dim)
    u, _, vh = torch.linalg.svd(q, full_matrices=False)
    return c * u, c * vh


def cutting_skip_attention_init(
    *,
    embed_dim: int,
    num_heads: int,
    alpha: float = 0.1,
    beta: float = 5.0,
    c: float = 0.5,
):
    """Create a SelfAttention block using the paper's exact construction.

    For Q/K we first build the target logits M = alpha * Z + beta * I, then
    factorize M = U S V^T and split the square root of the singular values into
    W_Q and W_K so that W_Q W_K^T = M. For V/O we sample a random Gaussian,
    decompose it via SVD, and set W_V = c * U and W_O = c * V^T.
    """
    from video_encoding.models.ViT.attention import SelfAttention

    attn = SelfAttention(embed_dim=embed_dim, num_heads=num_heads)
    with torch.no_grad():
        w_q, w_k = _cutting_skip_qk_factors(
            embed_dim, alpha=alpha, beta=beta, device=attn.qkv.weight.device, dtype=attn.qkv.weight.dtype
        )
        w_v, w_o = _cutting_skip_vo_factors(
            embed_dim, c=c, device=attn.qkv.weight.device, dtype=attn.qkv.weight.dtype
        )

        attn.qkv.weight.copy_(torch.cat([w_q, w_k, w_v], dim=0))
        attn.proj.weight.copy_(w_o)
        nn.init.zeros_(attn.qkv.bias)
        nn.init.zeros_(attn.proj.bias)
    return attn


def init_module(module: nn.Module, method: str = "trunc_normal", std: float = 0.02):
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
            if module.weight.ndim == 2 and module.weight.shape[0] == module.weight.shape[1]:
                base = torch.randn_like(module.weight) / math.sqrt(max(module.weight.shape[-1], 1))
                logits = cutting_skip_logits(base, base, alpha=0.1, beta=5.0)
                module.weight.copy_(logits)
            else:
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


def init_linear_cutting_skip(
    module: nn.Linear,
    *,
    alpha: float = 0.1,
    beta: float = 5.0,
    c: float = 0.5,
    std: float = 0.02,
) -> None:
    """Apply the paper's Gaussian -> SVD construction to a square weight matrix.

    This uses the V/O-style construction: random Gaussian Q, SVD Q = U S V^T,
    then keep the orthonormal factors and scale them by c. This is the correct
    pattern for the paper's scaled-orthonormal construction.
    """
    with torch.no_grad():
        if module.weight.ndim != 2 or module.weight.shape[0] != module.weight.shape[1]:
            nn.init.trunc_normal_(module.weight, std=std)
        else:
            base = torch.randn_like(module.weight) / math.sqrt(max(module.weight.shape[-1], 1))
            u, _, vh = torch.linalg.svd(base, full_matrices=False)
            module.weight.copy_(c * (u @ vh))
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def apply_weight_init(
    module: nn.Module,
    *,
    method: str = "trunc_normal",
    alpha: float = 0.1,
    beta: float = 5.0,
    c: float = 0.5,
    std: float = 0.02,
) -> None:
    """Apply a selected init policy while keeping the default behavior stable."""
    if isinstance(module, nn.Linear):
        if method == "cutting_skip":
            init_linear_cutting_skip(module, alpha=alpha, beta=beta, c=c, std=std)
        else:
            init_module(module, method=method, std=std)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv3d):
        init_module(module, method=method, std=std)


__all__ = [
    "apply_weight_init",
    "cutting_skip_attention_init",
    "cutting_skip_logits",
    "init_linear_cutting_skip",
    "init_module",
]