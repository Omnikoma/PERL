"""Uncertainty-aware evidence representation.

For each source s in {P, A, C}:
  sigma^s = softplus(f^U_s(o^s)) + eps        dimension-wise positive uncertainty (R^d)
  r^s     = m^s * sigmoid( -(1/d) sum_q log (sigma^s_q)^2 )     evidence confidence score

f^U_s is a 2-layer MLP from the source-specific indicator vector o^s to d. m^P == 1.
`no_uncertainty`: bypass the heads, r^s = m^s (available => fully confident).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class UncertaintyHead(nn.Module):
    def __init__(self, o_dim, d, eps=1e-6):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(o_dim, d), nn.ReLU(), nn.Linear(d, d))
        self.eps = eps

    def forward(self, o):
        return torch.nn.functional.softplus(self.net(o)) + self.eps     # (B,d)


class Uncertainty(nn.Module):
    def __init__(self, d, o_dims=(4, 3, 2), no_uncertainty=False):
        super().__init__()
        self.no_uncertainty = no_uncertainty
        self.heads = nn.ModuleDict({
            "P": UncertaintyHead(o_dims[0], d),
            "A": UncertaintyHead(o_dims[1], d),
            "C": UncertaintyHead(o_dims[2], d)})

    def confidence(self, sigma, m):
        # r = m * sigmoid(-mean_q log sigma_q^2)
        return m * torch.sigmoid(-(sigma ** 2).clamp(min=1e-12).log().mean(dim=1))

    def forward(self, o_P, o_A, o_C, m_P, m_A, m_C):
        if self.no_uncertainty:
            B = o_P.size(0); d = self.heads["P"].net[-1].out_features
            one = torch.ones(B, d, device=o_P.device)
            return {"sigma_P": one, "sigma_A": one, "sigma_C": one,
                    "r_P": m_P, "r_A": m_A, "r_C": m_C}
        sP, sA, sC = self.heads["P"](o_P), self.heads["A"](o_A), self.heads["C"](o_C)
        return {"sigma_P": sP, "sigma_A": sA, "sigma_C": sC,
                "r_P": self.confidence(sP, m_P),
                "r_A": self.confidence(sA, m_A),
                "r_C": self.confidence(sC, m_C)}
