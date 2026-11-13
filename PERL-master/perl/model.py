"""PERL model: wires the three evidence paths, uncertainty estimation, progressive
refinement, cross-evidence alignment, and the CTR head.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .modules.meta_encoder import MetaEncoder
from .modules.path_p import PathP
from .modules.path_a import PathA
from .modules.path_c import PathC
from .modules.uncertainty import Uncertainty
from .modules.refinement import Refinement
from .modules.alignment import CrossEvidenceAlignment
from .modules.ctr import UserRep, CTRHead


class PERL(nn.Module):
    def __init__(self, cfg, data):
        super().__init__()
        d = cfg.model.embed_dim
        self.cfg = cfg
        self.abl = cfg.ablation
        self.rho_max = cfg.model.rho_max

        self.f_M = MetaEncoder(data.meta_dim, d)
        self.path_p = PathP(data.meta_dim, d, data.d_x, K_P=cfg.model.K_P, w=cfg.model.w,
                            tau_M=cfg.model.tau_M, tau_P=cfg.model.tau_P, meta_encoder=self.f_M,
                            rho_max=self.rho_max,
                            prior=self.abl.get("population_prior", "prototypes"))
        self.path_a = PathA(cfg, data.d_x, d)
        self.path_c = PathC(data.n_users, d)
        self.unc = Uncertainty(d, o_dims=(4, 3, 2), no_uncertainty=self.abl.no_uncertainty)
        fusion = "concat" if self.abl.no_progressive else self.abl.get("fusion_strategy", "progressive")
        self.refine = Refinement(d, fusion=fusion)
        self.cecl = CrossEvidenceAlignment(d, tau_con=cfg.model.tau_con)
        self.user_rep = UserRep(data.n_users, d, self.f_M)
        self.ctr = CTRHead(d)

        self.pop_fallback = nn.Parameter(torch.zeros(d))   # mu^P when population removed/empty

    def forward(self, b):
        B = b["label"].size(0)
        d = self.cfg.model.embed_dim

        # ---- Path P ----
        if self.abl.no_population:
            mu_P = self.pop_fallback.expand(B, d)
            o_P = torch.zeros(B, 4, device=mu_P.device)
            m_P = torch.zeros(B, device=mu_P.device)       # r^P excluded from gate
        else:
            p = self.path_p(b["pool_meta"], b["pool_win"], b["pool_win_mask"], b["pool_valid"],
                            b["target_meta"], b["rho"])
            mu_P = p["mu_P"]; o_P = p["o_P"]
            # where no pool candidates exist, fall back to a learnable prior
            no_pool = (p["has_pool"] < 0.5).unsqueeze(-1)
            mu_P = torch.where(no_pool, self.pop_fallback.expand(B, d), mu_P)
            m_P = torch.ones(B, device=mu_P.device)

        # ---- Path A ----
        m_A = b["m_A"] * (0.0 if self.abl.no_temporal else 1.0)
        a = self.path_a(b["prefix"], b["prefix_mask"], m_A, b["L_it"], b["n_it"], b["stab"])
        mu_A, o_A = a["mu_A"], a["o_A"]

        # ---- Path C ----
        m_C = b["m_C"] * (0.0 if self.abl.no_collaborative else 1.0)
        c = self.path_c(b["collab_users"], b["collab_mask"], m_C, b["n_collab"], b["n_uniq_collab"])
        mu_C, o_C = c["mu_C"], c["o_C"]

        # ---- uncertainty / confidence ----
        u = self.unc(o_P, o_A, o_C, m_P if not self.abl.no_population else torch.ones(B, device=mu_P.device),
                     m_A, m_C)
        r_P, r_A, r_C = u["r_P"], u["r_A"], u["r_C"]
        if self.abl.no_population:
            r_P = torch.zeros(B, device=mu_P.device)

        # ---- progressive refinement ----
        rho_norm = (b["rho"] / self.rho_max).clamp(max=1.0)
        ref = self.refine(mu_P, mu_A, mu_C, r_P, r_A, r_C, m_A, m_C, rho_norm)
        h = ref["h"]

        # ---- CECL (training only consumer) ----
        cecl_loss, z = self.cecl(mu_P, mu_A, mu_C, m_A, m_C, r_P, r_A, r_C)

        # ---- CTR ----
        e_u = self.user_rep(b["user_id"], b["hist_meta"], b["hist_mask"])
        logits = self.ctr(e_u, h)

        return {
            "logits": logits, "cecl_loss": cecl_loss,
            "mu_P": mu_P, "mu_A": mu_A, "mu_C": mu_C,
            "sigma_P": u["sigma_P"], "sigma_A": u["sigma_A"], "sigma_C": u["sigma_C"],
            "r_P": r_P, "r_A": r_A, "r_C": r_C,
            "lambda_A": ref["lambda_A"], "lambda_C": ref["lambda_C"],
            "m_A": m_A, "m_C": m_C,
        }
