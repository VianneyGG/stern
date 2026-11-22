from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from stern.models.vit.rope import apply_rotary


class SelfAttention(nn.Module):
    """Multi-head self-attention with an optional QK-norm (L2-normalising
    queries and keys before the scaled dot product, bounding attention logit
    magnitude) -- a hook nn.MultiheadAttention's fused implementation
    doesn't expose, which is why this hand-written replacement exists.

    QK-norm bounds q.k to [-1, 1], so the usual head_dim**-0.5 factor would
    leave logits in a range softmax flattens to near-uniform over any real
    token count -- attention effectively switched off rather than normalised.
    A learnable per-head `logit_scale` (init sqrt(head_dim)) replaces that
    factor and restores usable logit range; it is only created when
    `qk_norm=True`, keeping the default path's parameter set identical to
    nn.MultiheadAttention's.

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
        if qk_norm:
            self.logit_scale = nn.Parameter(
                torch.full((num_heads, 1, 1), self.head_dim**0.5)
            )

    def forward(
        self,
        x: torch.Tensor,
        need_weights: bool = False,
        attn_mask: torch.Tensor | None = None,
        rotary: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if rotary is not None:
            cos, sin = rotary
            q, k = apply_rotary(q, k, cos.unsqueeze(1), sin.unsqueeze(1))

        if self.qk_norm:
            # Fold the per-head scale into q: SDPA's `scale=` takes a float,
            # not a per-head tensor, so the two paths agree only this way.
            q = F.normalize(q, dim=-1) * self.logit_scale
            k = F.normalize(k, dim=-1)
            scale = 1.0
        else:
            scale = self.scale

        attn_weights = None
        if need_weights:
            logits = (q @ k.transpose(-2, -1)) * scale
            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    logits = logits.masked_fill(~attn_mask, float("-inf"))
                else:
                    logits = logits + attn_mask
            attn_weights = logits.softmax(dim=-1)
            attn_drop = F.dropout(attn_weights, p=self.dropout, training=self.training)
            out = attn_drop @ v
        else:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                scale=scale,
            )

        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        return out, attn_weights
