"""CTR prediction and user representation.

e_{u,t}: user-id embedding + mean of the metadata representations f_M(m_j) of the
user's most-recent N items before t, added together. History items use ONLY metadata (no item
ID), so the user representation stays valid for unseen cold test items.

y_hat = sigmoid( f_CTR([e_{u,t}; h_{i,t}; e_{u,t} ⊙ h_{i,t}; c_{u,i,t}]) )
c_{u,i,t} defaults to empty. f_CTR is a 3-layer MLP.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class UserRep(nn.Module):
    def __init__(self, n_users, d, meta_encoder):
        super().__init__()
        self.user_emb = nn.Embedding(n_users + 1, d)
        self.f_M = meta_encoder

    def forward(self, user_id, hist_meta, hist_mask):
        """hist_meta:(B,H,meta_dim), hist_mask:(B,H) True=valid."""
        uid = self.user_emb(user_id.clamp(min=0))                 # (B,d)
        q = self.f_M(hist_meta)                                   # (B,H,d)
        m = hist_mask.float().unsqueeze(-1)
        hist = (q * m).sum(1) / m.sum(1).clamp(min=1.0)          # (B,d)
        return uid + hist


class CTRHead(nn.Module):
    def __init__(self, d, ctx_dim=0):
        super().__init__()
        inp = 3 * d + ctx_dim
        self.net = nn.Sequential(
            nn.Linear(inp, d), nn.ReLU(),
            nn.Linear(d, d), nn.ReLU(),
            nn.Linear(d, 1))

    def forward(self, e_u, h, ctx=None):
        feats = [e_u, h, e_u * h]
        if ctx is not None:
            feats.append(ctx)
        return self.net(torch.cat(feats, dim=-1)).squeeze(-1)     # logits (B,)
