"""Path C: collaborative evidence.

mu^C = f_C(R_{i,<t}),  m^C = 1[|R_{i,<t}| > 0].

neighborhood aggregation: mean-pool learnable user-id embeddings of the users who
interacted with item i before the cutoff, then a linear map to d. Since R_{i,<t} changes with
the cutoff t, f_C is a function OF THE INTERACTION SET (not a static item-ID embedding).

o^C = [log(1 + |R_{i,<t}|), Div(R)] with Div = (#unique users) / |R|.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PathC(nn.Module):
    def __init__(self, n_users, d):
        super().__init__()
        self.user_emb = nn.Embedding(n_users + 1, d, padding_idx=0)
        self.proj = nn.Linear(d, d)

    def forward(self, collab_users, collab_mask, m_C, n_collab, n_uniq):
        """collab_users:(B,C) ids, collab_mask:(B,C) True=valid."""
        emb = self.user_emb(collab_users.clamp(min=0))             # (B,C,d)
        m = collab_mask.float().unsqueeze(-1)
        pooled = (emb * m).sum(1) / m.sum(1).clamp(min=1.0)        # (B,d)
        mu_C = self.proj(pooled) * m_C.unsqueeze(-1)              # zero when |R|=0
        div = torch.where(n_collab > 0, n_uniq / n_collab.clamp(min=1.0),
                          torch.zeros_like(n_collab))
        o_C = torch.stack([torch.log1p(n_collab), div], dim=1)    # (B,2)
        return {"mu_C": mu_C, "o_C": o_C, "m_C": m_C}
