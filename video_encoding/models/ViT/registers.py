from __future__ import annotations

import torch
from torch import nn


class RegisterTokens(nn.Module):
    """Extra learnable tokens (Darcet et al. 2023, "Vision Transformers Need
    Registers") prepended to the sequence before the transformer blocks and
    stripped afterward -- they participate in attention but are never part
    of the patch-token grid, never masked, never reconstructed."""

    def __init__(self, embed_dim: int, num_registers: int):
        super().__init__()
        self.num_registers = num_registers
        self.tokens = nn.Parameter(torch.zeros(1, num_registers, embed_dim))
        if num_registers > 0:
            nn.init.trunc_normal_(self.tokens, std=0.02)

    def prepend(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_registers == 0:
            return x
        registers = self.tokens.expand(x.shape[0], -1, -1)
        return torch.cat([registers, x], dim=1)

    def strip(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_registers == 0:
            return x
        return x[:, self.num_registers :, :]
