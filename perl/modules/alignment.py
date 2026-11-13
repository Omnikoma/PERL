"""Confidence-aware cross-evidence alignment L_CECL.

Project each source into a shared space z^s = psi_s(mu^s) (2-layer MLP -> d, L2-normalized).
For adjacent pairs (P,A) and (A,C), over the batch instances where BOTH sources are available
(B_ab), compute an instance-level InfoNCE with negatives from the other B_ab instances, and
weight each pair by its DETACHED joint confidence omega = sg(r^a * r^b).

  L_CECL = sum_{(a,b)} [ sum_i omega_i * l_i / (sum_i omega_i + eps) ]

Empty B_ab -> 0. `no_cecl` is handled by the loss weight lambda_con = 0.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Projector(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class CrossEvidenceAlignment(nn.Module):
    def __init__(self, d, tau_con=0.1):
        super().__init__()
        self.tau = tau_con
        self.psi = nn.ModuleDict({"P": Projector(d), "A": Projector(d), "C": Projector(d)})

    def _pair_loss(self, za, zb, avail, ra, rb):
        """za,zb:(B,d) projected & normalized; avail:(B,) bool both-available; r*:(B,)."""
        idx = avail.nonzero(as_tuple=True)[0]
        if idx.numel() < 2:
            return za.new_zeros(())
        za, zb = za[idx], zb[idx]
        sim = (za @ zb.t()) / self.tau                       # (n,n)
        tgt = torch.arange(za.size(0), device=za.device)
        l = F.cross_entropy(sim, tgt, reduction="none")      # (n,)
        omega = (ra[idx] * rb[idx]).detach()                 # stop-gradient weights
        return (omega * l).sum() / (omega.sum() + 1e-9)

    def forward(self, mu_P, mu_A, mu_C, m_A, m_C, r_P, r_A, r_C):
        zP, zA, zC = self.psi["P"](mu_P), self.psi["A"](mu_A), self.psi["C"](mu_C)
        avail_PA = (m_A > 0)
        avail_AC = (m_A > 0) & (m_C > 0)
        loss = self._pair_loss(zP, zA, avail_PA, r_P, r_A) \
            + self._pair_loss(zA, zC, avail_AC, r_A, r_C)
        return loss, {"zP": zP, "zA": zA, "zC": zC}
