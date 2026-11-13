"""Evidence-driven progressive refinement.

h^(0) = mu^P
Delta^A = phi_A([mu^A; h0; mu^A - h0; mu^A ⊙ h0])
lambda^A = m^A * sigmoid(f^A_G([r^P; r^A; c^PA; rho~; ||Delta^A||]))     (scalar gate)
h^(1) = h0 + lambda^A Delta^A
Delta^C = phi_C([mu^C; h1; mu^C - h1; mu^C ⊙ h1])
lambda^C = m^C * sigmoid(f^C_G([r^C; c^C; rho~; ||Delta^C||]))
h = h1 + lambda^C Delta^C

Confidence scores r^s are fed to the gates WITHOUT stop-gradient (only CECL weights detach).

Alternative integration strategies (ablation.fusion_strategy) share the same interface:
  mean        : average of the available sources (m^s-masked)
  concat      : static concatenation [mu^P; mu^A; mu^C] -> MLP      (= `no_progressive`)
  reliability : one-shot confidence-weighted sum  sum_s r^s mu^s / sum_s r^s
  progressive : the gated residual refinement above                 (default)
For the non-progressive variants lambda^A = lambda^C = 0 is reported.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

FUSIONS = ("mean", "concat", "reliability", "progressive")


class _Residual(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(4 * d, d), nn.ReLU(), nn.Linear(d, d))

    def forward(self, mu, h):
        return self.phi(torch.cat([mu, h, mu - h, mu * h], dim=-1))


class Refinement(nn.Module):
    def __init__(self, d, fusion="progressive"):
        super().__init__()
        if fusion not in FUSIONS:
            raise ValueError(f"unknown fusion_strategy: {fusion} (choose from {FUSIONS})")
        self.fusion = fusion
        if fusion == "concat":
            self.fuse = nn.Sequential(nn.Linear(3 * d, d), nn.ReLU(), nn.Linear(d, d))
        elif fusion == "progressive":
            self.phi_A = _Residual(d)
            self.phi_C = _Residual(d)
            self.gA = nn.Sequential(nn.Linear(5, d), nn.ReLU(), nn.Linear(d, 1))
            self.gC = nn.Sequential(nn.Linear(4, d), nn.ReLU(), nn.Linear(d, 1))
        # mean / reliability are parameter-free

    @staticmethod
    def _weighted(mu_P, mu_A, mu_C, wP, wA, wC):
        W = torch.stack([wP, wA, wC], dim=1)                       # (B,3)
        M = torch.stack([mu_P, mu_A, mu_C], dim=1)                 # (B,3,d)
        return (W.unsqueeze(-1) * M).sum(1) / W.sum(1, keepdim=True).clamp(min=1e-6)

    def forward(self, mu_P, mu_A, mu_C, r_P, r_A, r_C, m_A, m_C, rho_norm):
        z = torch.zeros(mu_P.size(0), device=mu_P.device)
        if self.fusion == "concat":
            h = self.fuse(torch.cat([mu_P, mu_A, mu_C], dim=-1))
            return {"h": h, "lambda_A": z, "lambda_C": z, "h0": mu_P, "h1": mu_P}
        if self.fusion == "mean":
            h = self._weighted(mu_P, mu_A, mu_C, torch.ones_like(m_A), m_A, m_C)
            return {"h": h, "lambda_A": z, "lambda_C": z, "h0": mu_P, "h1": mu_P}
        if self.fusion == "reliability":
            h = self._weighted(mu_P, mu_A, mu_C, r_P, r_A, r_C)   # r^s already carries m^s
            return {"h": h, "lambda_A": z, "lambda_C": z, "h0": mu_P, "h1": mu_P}

        h0 = mu_P
        dA = self.phi_A(mu_A, h0)
        cPA = F.cosine_similarity(mu_P, mu_A, dim=-1)
        gin_A = torch.stack([r_P, r_A, cPA, rho_norm, dA.norm(dim=-1)], dim=1)
        lam_A = m_A * torch.sigmoid(self.gA(gin_A)).squeeze(-1)
        h1 = h0 + lam_A.unsqueeze(-1) * dA

        dC = self.phi_C(mu_C, h1)
        cC = F.cosine_similarity(mu_C, h1, dim=-1)
        gin_C = torch.stack([r_C, cC, rho_norm, dC.norm(dim=-1)], dim=1)
        lam_C = m_C * torch.sigmoid(self.gC(gin_C)).squeeze(-1)
        h = h1 + lam_C.unsqueeze(-1) * dC
        return {"h": h, "lambda_A": lam_A, "lambda_C": lam_C, "h0": h0, "h1": h1}
