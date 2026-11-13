"""PERL training objective.

L = L_CTR + lambda_con * L_CECL + lambda_unc * L_unc + lambda_reg * L_reg

- L_CTR : binary cross-entropy on the CTR logits.
- L_CECL: confidence-weighted cross-evidence InfoNCE, computed inside the model.
- L_unc : monotone uncertainty calibration over ordered lifecycle pairs (t_a < t_b) of the
          SAME item, only for s in {A, C}:  sum max(0, sigma_bar^s_{t_b} - sigma_bar^s_{t_a}).
- L_reg : L2 over trainable parameters.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def ctr_loss(logits, labels):
    return F.binary_cross_entropy_with_logits(logits, labels.float())


def unc_loss(out_a, out_b, m_a):
    """Ordered pair (a=earlier cutoff, b=later). Penalize uncertainty that INCREASES with
    more observed history. Only A and C are constrained (P is excluded)."""
    total = out_a["logits"].new_zeros(())
    for s, msrc in (("A", "m_A"), ("C", "m_C")):
        sa = out_a[f"sigma_{s}"].mean(dim=1)       # sigma_bar at t_a
        sb = out_b[f"sigma_{s}"].mean(dim=1)       # sigma_bar at t_b
        m = out_a[msrc]                             # available at the earlier cutoff
        total = total + (m * F.relu(sb - sa)).sum()
    return total


def l2_reg(model):
    return sum((p ** 2).sum() for p in model.parameters() if p.requires_grad)


def total_loss(out_a, labels, cfg, model, out_b=None):
    lc = cfg.loss
    l_ctr = ctr_loss(out_a["logits"], labels)
    l_cecl = out_a["cecl_loss"] if not cfg.ablation.no_cecl else out_a["logits"].new_zeros(())
    l_unc = unc_loss(out_a, out_b, out_a["m_A"]) if out_b is not None else out_a["logits"].new_zeros(())
    l_reg = l2_reg(model)
    total = l_ctr + lc.lambda_con * l_cecl + lc.lambda_unc * l_unc + lc.lambda_reg * l_reg
    return total, {"ctr": float(l_ctr), "cecl": float(l_cecl),
                   "unc": float(l_unc), "reg": float(l_reg), "total": float(total)}
