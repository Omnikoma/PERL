"""Path A: target-specific adaptive temporal evidence.

mu^A = g_A(Phi_TSFM(S_{i,t})),  m^A = 1[L_{i,t} > 0].

Phi_TSFM is Google TimesFM 2.5 loaded from REAL pretrained weights, FROZEN, with LoRA
injected on the fused QKV projection and fine-tuned END-TO-END via the PERL CTR loss. Only
the backbone hidden states (`output_embeddings`) are used — NO forecasting/decode call.

Multi-channel handling: TimesFM is univariate, so each of the d_x channels
is encoded as an independent series (stacked in the batch dim), its patch hidden states are
mean-pooled, and the d_x channel vectors are concatenated and linearly mapped to d.

Encoder choices: gru | tcn | transformer | frozen_tsfm | lora_tsfm.

o^A = [log(1+L), log(1+n), Stab(S)], where n counts pre-cutoff positive interactions and
Stab = 1/(1 + CV) of the nonnegative aggregate volumes over the prefix (computed in the collate).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class _TSFMEncoder(nn.Module):
    """Wraps TimesFM 2.5 backbone; returns per-series patch hidden states (no forecast)."""

    def __init__(self, cfg, lora: bool):
        super().__init__()
        import timesfm   # resolved from the vendored third_party/timesfm (see config.load_config)
        # load from the local weights directory (contains model.safetensors) — fully offline
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            cfg.paths.timesfm_model_dir, local_files_only=True)
        backbone = model.model.float()
        for p in backbone.parameters():
            p.requires_grad = False
        if lora:
            from peft import LoraConfig, get_peft_model
            lc = LoraConfig(r=cfg.path_a.lora_r, lora_alpha=cfg.path_a.lora_alpha,
                            lora_dropout=cfg.path_a.lora_dropout,
                            target_modules=list(cfg.path_a.lora_target_modules), bias="none")
            backbone = get_peft_model(backbone, lc)
        self.backbone = backbone
        self.patch = cfg.path_a.patch_len
        self.hidden = cfg.path_a.hidden_size
        self.max_prefix = cfg.path_a.max_prefix_len
        self.microbatch = int(getattr(cfg.path_a, "tsfm_microbatch", 1024))

    def forward(self, prefix, mask):
        """prefix:(B,L,dx) front-aligned, mask:(B,L) True=valid -> pooled (B, dx, hidden)."""
        B, L, dx = prefix.shape
        P = self.patch
        lengths = mask.sum(1).clamp(max=self.max_prefix)          # (B,)
        maxL = int(lengths.max().item()) if B else 0
        Lpad = max(P, int(math.ceil(max(maxL, 1) / P) * P))
        dev = prefix.device

        buf = torch.zeros(B, dx, Lpad, device=dev)
        pad = torch.ones(B, Lpad, dtype=torch.bool, device=dev)   # True=padding
        for b in range(B):
            Lb = int(lengths[b].item())
            if Lb > 0:
                obs = prefix[b, :Lb, :]                            # (Lb, dx)
                buf[b, :, Lpad - Lb:] = obs.transpose(0, 1)
                pad[b, Lpad - Lb:] = False
            else:
                pad[b, :] = False                                 # observed zeros; zeroed later
        series = buf.reshape(B * dx, Lpad)
        masks = pad.unsqueeze(1).expand(B, dx, Lpad).reshape(B * dx, Lpad)
        npatch = Lpad // P
        inp = series.reshape(B * dx, npatch, P)
        msk = masks.reshape(B * dx, npatch, P)
        # process (B*dx) series in sub-batches to bound per-forward memory
        pooled_chunks = []
        for s in range(0, B * dx, self.microbatch):
            ii, mm = inp[s:s + self.microbatch], msk[s:s + self.microbatch]
            emb = self.backbone(ii, mm)[0][1]                     # (n, npatch, hidden)
            pv = (~mm.all(dim=-1)).float().unsqueeze(-1)
            pooled_chunks.append((emb * pv).sum(1) / pv.sum(1).clamp(min=1.0))
        pooled = torch.cat(pooled_chunks, dim=0)                  # (B*dx, hidden)
        return pooled.reshape(B, dx, self.hidden)


class _SmallEncoder(nn.Module):
    """From-scratch temporal encoders over (B,L,dx): gru | tcn | transformer -> (B, dx*hid?)."""

    def __init__(self, kind, d_x, hidden):
        super().__init__()
        self.kind = kind
        self.hidden = hidden
        if kind == "gru":
            self.net = nn.GRU(d_x, hidden, batch_first=True)
        elif kind == "tcn":
            self.net = nn.Conv1d(d_x, hidden, kernel_size=3, padding=1)
        elif kind == "transformer":
            self.inp = nn.Linear(d_x, hidden)
            layer = nn.TransformerEncoderLayer(hidden, nhead=4, batch_first=True)
            self.net = nn.TransformerEncoder(layer, num_layers=2)
        else:
            raise ValueError(kind)

    def forward(self, prefix, mask):
        B, L, dx = prefix.shape
        m = mask.float().unsqueeze(-1)
        if self.kind == "gru":
            out, _ = self.net(prefix)
        elif self.kind == "tcn":
            out = self.net(prefix.transpose(1, 2)).transpose(1, 2)
        else:
            out = self.net(self.inp(prefix), src_key_padding_mask=~mask)
        pooled = (out * m).sum(1) / m.sum(1).clamp(min=1.0)       # (B, hidden)
        return pooled.unsqueeze(1)                                 # (B, 1, hidden)


class PathA(nn.Module):
    def __init__(self, cfg, d_x, d):
        super().__init__()
        self.kind = cfg.path_a.encoder
        self.d = d
        if self.kind in ("lora_tsfm", "frozen_tsfm"):
            self.enc = _TSFMEncoder(cfg, lora=(self.kind == "lora_tsfm"))
            proj_in = d_x * cfg.path_a.hidden_size
        else:
            self.enc = _SmallEncoder(self.kind, d_x, cfg.path_a.hidden_size)
            proj_in = cfg.path_a.hidden_size
        self.g_A = nn.Sequential(nn.Linear(proj_in, d), nn.ReLU(), nn.Linear(d, d))

    def forward(self, prefix, prefix_mask, m_A, L_it, n_it, stab):
        pooled = self.enc(prefix, prefix_mask)                     # (B, C, hidden)
        mu_A = self.g_A(pooled.reshape(pooled.size(0), -1))        # (B, d)
        mu_A = mu_A * m_A.unsqueeze(-1)                            # zero when L=0
        o_A = torch.stack([torch.log1p(L_it), torch.log1p(n_it), stab], dim=1)   # (B,3)
        return {"mu_A": mu_A, "o_A": o_A, "m_A": m_A}
