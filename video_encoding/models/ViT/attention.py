from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SelfAttention(nn.Module):
    """Multi-head self-attention with an optional QK-norm (L2-normalising
    queries and keys before the scaled dot product, bounding attention logit
    magnitude) -- a hook nn.MultiheadAttention's fused implementation
    doesn't expose, which is why this hand-written replacement exists.

    Parameter layout (qkv, proj) intentionally mirrors nn.MultiheadAttention's
    (in_proj, out_proj) shapes exactly, so weights can be copied 1:1 between
    them -- see test_vit_attention.py's parity test.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        qk_norm: bool = False,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
            )
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout
        self.qk_norm = qk_norm

    def forward(
        self, x: torch.Tensor, need_weights: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.qk_norm:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)

        attn_weights = None
        if need_weights:
            logits = (q @ k.transpose(-2, -1)) * self.scale
            attn_weights = logits.softmax(dim=-1)
            attn_drop = F.dropout(attn_weights, p=self.dropout, training=self.training)
            out = attn_drop @ v
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0
            )

        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        return out, attn_weights
