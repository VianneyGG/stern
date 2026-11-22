"""Feedforward variants for ViT blocks, selected by `mlp_type`."""

from __future__ import annotations

from torch import nn


class GELUMlp(nn.Module):
    """Today's MLP, extracted as-is: Linear -> GELU -> Dropout -> Linear -> Dropout."""

    def __init__(self, embed_dim: int, mlp_ratio: float, dropout: float = 0.0):
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class SwiGLUMlp(nn.Module):
    """SwiGLU feedforward (Shazeer 2020): gated SiLU activation.

    Hidden dim carries the standard 2/3 correction: SwiGLU spends three
    projections (gate, up, down) where GELUMlp spends two, so an uncorrected
    embed_dim*mlp_ratio would hand this variant 1.5x the parameters and turn
    `mlp_type` into a capacity confound instead of an activation ablation.
    At 2/3 the two variants are weight-matched at equal mlp_ratio.
    """

    def __init__(self, embed_dim: int, mlp_ratio: float, dropout: float = 0.0):
        super().__init__()
        hidden = int(embed_dim * mlp_ratio * 2 / 3)
        self.gate = nn.Linear(embed_dim, hidden)
        self.up = nn.Linear(embed_dim, hidden)
        self.down = nn.Linear(hidden, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, x):
        h = self.act(self.gate(x)) * self.up(x)
        return self.dropout(self.down(h))


MLPS: dict[str, type[nn.Module]] = {"gelu": GELUMlp, "swiglu": SwiGLUMlp}


def build_mlp(
    mlp_type: str, embed_dim: int, mlp_ratio: float, dropout: float = 0.0
) -> nn.Module:
    if mlp_type not in MLPS:
        raise ValueError(f"unknown mlp_type {mlp_type!r}; choices: {sorted(MLPS)}")
    return MLPS[mlp_type](embed_dim, mlp_ratio, dropout)
