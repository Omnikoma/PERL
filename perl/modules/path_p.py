"""Path P: lifecycle-conditioned, target-conditioned multi-prototype population prior.

Inputs (from collate, per batch):
  pool_meta     (B, K, meta_dim)   metadata of the K retrieved historical analogs
  pool_win      (B, K, W, d_x)     their aligned behavioral windows (0-padded, front-aligned)
  pool_win_mask (B, K, W)          valid steps inside each window
  pool_valid    (B, K)             validity mask (fewer than K analogs -> padded)
  target_meta   (B, meta_dim), rho (B,) current lifecycle position of the target

Pipeline:
  g_P  : window -> h^P_j (GRU + masked mean pool) -> (B, K, d)
  alpha^M_ij = softmax(s^M_ij / tau_M) over valid analogs, s^M = cos(f_M(m_i), f_M(m_j))
  a_jk       = softmax(cos(h^P_j, c^P_k) / tau_P)                prototype assignment
  p^k, pi^k  from combined weights alpha_tilde = alpha^M * a_jk
  q    = f_Q([f_M(m_i); e_rho]),  beta^k = softmax_k(q^T W_P p^k / sqrt(d))
  mu^P = sum_k beta^k p^k                       (target-conditioned prototype selection)
  o^P  = [H(alpha^M), H(pi), sum_k pi^k ||p^k - mu^P||^2, max_j s^M_ij]

`prior = single_mean` replaces the prototypes by the retrieval-weighted mean sum_j alpha^M_ij h^P_j.
The historical alignment strategy is decided in the collate; this module is agnostic to it.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _entropy(p, dim=-1, eps=1e-9):
    return -(p * (p + eps).log()).sum(dim)


class PathP(nn.Module):
    def __init__(self, meta_dim, d, d_x, K_P=4, w=5, tau_M=0.1, tau_P=0.1,
                 meta_encoder=None, rho_max=64, prior="prototypes"):
        super().__init__()
        if prior not in ("prototypes", "single_mean"):
            raise ValueError(f"unknown population_prior: {prior}")
        self.d, self.K_P, self.tau_M, self.tau_P = d, K_P, tau_M, tau_P
        self.prior = prior
        self.rho_max = rho_max
        self.f_M = meta_encoder                       # shared f_M
        self.g_P = nn.GRU(d_x, d, batch_first=True)   # lightweight temporal encoder
        self.prototypes = nn.Parameter(torch.randn(K_P, d) * 0.02)
        # target-conditioned prototype selection
        self.e_rho = nn.Embedding(rho_max + 2, d)     # lifecycle-position embedding (clamped)
        self.f_Q = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, d))
        self.W_P = nn.Linear(d, d, bias=False)

    def encode_windows(self, win, win_mask):
        """win: (B,K,W,d_x), win_mask: (B,K,W) -> h^P: (B,K,d) via GRU + masked mean pool."""
        B, K, W, dx = win.shape
        out, _ = self.g_P(win.reshape(B * K, W, dx))                 # (B*K, W, d)
        m = win_mask.reshape(B * K, W, 1).float()
        h = (out * m).sum(1) / m.sum(1).clamp(min=1.0)
        return h.reshape(B, K, self.d)

    def forward(self, pool_meta, pool_win, pool_win_mask, pool_valid, target_meta, rho):
        B, K, _ = pool_meta.shape
        d = self.d
        valid = pool_valid.float()                     # (B,K)
        any_valid = valid.sum(1, keepdim=True).clamp(min=1.0)

        h_P = self.encode_windows(pool_win, pool_win_mask)       # (B,K,d)

        # s^M via learned f_M (cosine). NaN-safe masked softmax: exponentiate then zero out
        # invalid analogs (avoids softmax-over-all-(-inf) -> NaN for empty-pool rows).
        q_meta = self.f_M(target_meta)                            # (B,d)
        qi = q_meta.unsqueeze(1)                                  # (B,1,d)
        qj = self.f_M(pool_meta)                                  # (B,K,d)
        sM = F.cosine_similarity(qi, qj, dim=-1)                  # (B,K)
        sM_shift = (sM / self.tau_M) - (sM / self.tau_M).max(dim=1, keepdim=True).values
        ex = sM_shift.exp() * valid
        alphaM = ex / ex.sum(1, keepdim=True).clamp(min=1e-9)     # all-invalid row -> zeros

        # prototype assignment a_jk = softmax(cos(h^P_j, c_k)/tau_P)
        proto = self.prototypes.view(1, 1, self.K_P, d)
        hj = h_P.unsqueeze(2)                                     # (B,K,1,d)
        cos_jk = F.cosine_similarity(hj, proto, dim=-1)           # (B,K,K_P)
        a_jk = torch.softmax(cos_jk / self.tau_P, dim=-1)         # (B,K,K_P)

        # combined weights alpha_tilde_ijk = alpha^M_ij * a_jk
        at = alphaM.unsqueeze(-1) * a_jk                          # (B,K,K_P)
        at = at * valid.unsqueeze(-1)
        denom_k = at.sum(1)                                       # (B,K_P)
        # p^k = sum_j at_jk h^P_j / (sum_j at_jk + eps)
        p_k = torch.einsum("bkp,bkd->bpd", at, h_P) / (denom_k.unsqueeze(-1) + 1e-9)
        pi_k = denom_k / denom_k.sum(1, keepdim=True).clamp(min=1e-9)   # (B,K_P)

        if self.prior == "single_mean":
            mu_P = (alphaM.unsqueeze(-1) * h_P).sum(1)            # retrieval-weighted mean
            beta = pi_k
        else:
            # target-conditioned prototype selection
            rho_idx = rho.long().clamp(min=0, max=self.rho_max + 1)
            q = self.f_Q(torch.cat([q_meta, self.e_rho(rho_idx)], dim=-1))     # (B,d)
            logits = torch.einsum("bd,bpd->bp", self.W_P(q), p_k) / math.sqrt(d)
            beta = torch.softmax(logits, dim=-1)                  # (B,K_P)
            mu_P = (beta.unsqueeze(-1) * p_k).sum(1)              # (B,d)

        # o^P indicators
        H_alpha = _entropy(alphaM.clamp(min=1e-9))               # (B,)
        H_pi = _entropy(pi_k.clamp(min=1e-9))
        inter_var = (pi_k * ((p_k - mu_P.unsqueeze(1)) ** 2).sum(-1)).sum(1)  # (B,)
        max_sM = sM.masked_fill(~pool_valid, -1.0).max(dim=1).values          # (B,)
        o_P = torch.stack([H_alpha, H_pi, inter_var, max_sM], dim=1)          # (B,4)

        return {"mu_P": mu_P, "p_k": p_k, "pi_k": pi_k, "beta": beta, "o_P": o_P,
                "alphaM": alphaM, "sM": sM, "has_pool": (any_valid.squeeze(1) > 0).float()}
