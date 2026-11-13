"""f_M: shared metadata encoder.

discrete category multi-hot + log1p(duration) -> 2-layer MLP -> d.
Similarity sim(q_i, q_j) is cosine. The raw-metadata cosine used for TopK *selection* lives
in the collate; here f_M provides the LEARNED embedding used for the prototype weighting
alpha^M over the already-selected candidates.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaEncoder(nn.Module):
    def __init__(self, meta_dim: int, d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(meta_dim, d), nn.ReLU(), nn.Linear(d, d))

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """m: (..., meta_dim) -> q: (..., d)."""
        return self.net(m)

    @staticmethod
    def cosine(a: torch.Tensor, b: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return F.cosine_similarity(a, b, dim=dim)
