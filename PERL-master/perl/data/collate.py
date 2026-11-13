"""Batch builder: turns a list of (user, item, cutoff_ns, label) instances into padded
tensors by calling the causal slice functions in dataset.PERLData.

Retrieval note: TopK population candidates are SELECTED by cosine over the raw metadata
vectors (cacheable per (i, rho) and leak-free since it only uses static metadata + the causal
get_pool filter). The learned f_M still shapes the prototype weighting alpha^M over the
selected set inside the model.

Historical alignment (ablation.align_strategy): lifecycle_causal (default) | symmetric |
calendar | whole_trajectory; `ablation.no_lifecycle_align: true` is shorthand for
whole_trajectory. Lifecycle-based strategies restrict the pool to analogs that have already
reached the target's lifecycle position (L_{j,t} >= rho); calendar / whole_trajectory only
require L_{j,t} >= 1. The analog window length W is w for the first three and
path_a.max_prefix_len for whole_trajectory.
"""
from __future__ import annotations

import numpy as np
import torch


def _pad_seq(arrs, max_len, dim):
    """Stack variable-length (L_k, dim) arrays into (B, max_len, dim) + mask (B, max_len)."""
    B = len(arrs)
    out = np.zeros((B, max_len, dim), np.float32)
    mask = np.zeros((B, max_len), bool)
    for k, a in enumerate(arrs):
        L = min(len(a), max_len)
        if L > 0:
            out[k, :L] = a[-L:] if a.shape[0] > max_len else a[:L]
            mask[k, :L] = True
    return out, mask


def _cos_topk(qi, qjs, K):
    """Indices of top-K rows of qjs by cosine with qi (qjs: (M, d))."""
    if len(qjs) == 0:
        return np.zeros(0, int), np.zeros(0, np.float32)
    a = qi / (np.linalg.norm(qi) + 1e-8)
    b = qjs / (np.linalg.norm(qjs, axis=1, keepdims=True) + 1e-8)
    sim = b @ a
    k = min(K, len(sim))
    idx = np.argpartition(-sim, k - 1)[:k]
    idx = idx[np.argsort(-sim[idx])]
    return idx, sim[idx].astype(np.float32)


def align_strategy(cfg) -> str:
    abl = cfg["ablation"]
    if abl.get("no_lifecycle_align", False):
        return "whole_trajectory"
    return abl.get("align_strategy", "lifecycle_causal")


def build_batch(data, instances, cfg, device):
    """instances: list of (user_id, item_id, cutoff_ns, label). Returns dict of tensors."""
    K, w = cfg.model.K, cfg.model.w
    H = cfg.model.user_hist_len
    max_prefix = cfg.path_a.max_prefix_len
    strategy = align_strategy(cfg)
    W = max_prefix if strategy == "whole_trajectory" else w      # analog window length
    lifecycle_based = strategy in ("lifecycle_causal", "symmetric")
    C = 128  # collab user cap
    dx, md = data.d_x, data.meta_dim

    users, labels, cutoffs = [], [], []
    rhos, Lits, mA, mC, stabs = [], [], [], [], []
    prefixes, tgt_meta = [], []
    pool_meta, pool_win, pool_win_mask, pool_sim, pool_valid = [], [], [], [], []
    collab_u, hist_meta_rows = [], []
    n_it, n_collab, n_uniq_collab = [], [], []

    for (u, i, t, y) in instances:
        u, i, t, y = int(u), int(i), int(t), int(y)
        L = data.L_at(i, t)
        rho = L + 1
        users.append(u); labels.append(y); cutoffs.append(t)
        Lits.append(L); rhos.append(rho)

        # ---- Path A prefix ----
        S = data.get_prefix(i, t)                 # (L, d_x)
        prefixes.append(S)
        mA.append(1.0 if L > 0 else 0.0)
        # Stab(S) = 1 / (1 + CV) of the nonnegative aggregate volumes over the prefix
        vol = data.get_volume(i, t)
        if len(vol):
            cv = float(vol.std()) / (float(vol.mean()) + 1e-6)
            stabs.append(1.0 / (1.0 + cv))
        else:
            stabs.append(0.0)

        # ---- Path C collab ----
        cu, _, cl = data.get_collab(i, t)
        collab_u.append(cu)
        mC.append(1.0 if len(cu) > 0 else 0.0)
        n_it.append(int(cl.sum()))                # pre-cutoff positive interactions
        n_collab.append(len(cu))
        n_uniq_collab.append(int(len(np.unique(cu))) if len(cu) else 0)

        # ---- user history metadata ----
        hist = data.get_user_hist(u, t, H)
        hm = np.stack([data.meta_vec[int(h)] for h in hist]) if len(hist) else np.zeros((0, md), np.float32)
        hist_meta_rows.append(hm)

        # ---- Path P retrieval + aligned analog windows ----
        qi = data.meta_vec.get(i, np.zeros(md, np.float32))
        tgt_meta.append(qi)
        pool = data.get_pool(i, t, rho if lifecycle_based else 1)
        pm = np.zeros((K, md), np.float32); ps = np.zeros(K, np.float32); pv = np.zeros(K, bool)
        pw = np.zeros((K, W, dx), np.float32); pwm = np.zeros((K, W), bool)
        if len(pool):
            qjs = np.stack([data.meta_vec[int(j)] for j in pool])
            sel, sims = _cos_topk(qi, qjs, K)
            for k, (j, sm) in enumerate(zip(pool[sel], sims)):
                win = data.get_aligned_window(int(j), rho, w, t, strategy, W)
                if len(win) == 0:
                    continue                      # analog without a usable window -> empty slot
                n = min(len(win), W)
                pm[k] = data.meta_vec[int(j)]; ps[k] = sm; pv[k] = True
                pw[k, :n] = win[-n:]; pwm[k, :n] = True
        pool_meta.append(pm); pool_sim.append(ps); pool_valid.append(pv)
        pool_win.append(pw); pool_win_mask.append(pwm)

    # ---- stack to tensors ----
    pref_arr, pref_mask = _pad_seq(prefixes, max_prefix, dx)
    hist_arr, hist_mask = _pad_seq(hist_meta_rows, H, md)
    collab_arr = np.zeros((len(users), C), np.int64); collab_mask = np.zeros((len(users), C), bool)
    for k, cu in enumerate(collab_u):
        n = min(len(cu), C)
        if n:
            collab_arr[k, :n] = cu[-n:]; collab_mask[k, :n] = True

    t_ = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=device)
    return {
        "user_id": t_(users, torch.long), "label": t_(labels), "cutoff": t_(cutoffs, torch.long),
        "rho": t_(rhos), "L_it": t_(Lits), "stab": t_(stabs),
        "m_A": t_(mA), "m_C": t_(mC),
        "prefix": t_(pref_arr), "prefix_mask": t_(pref_mask, torch.bool),
        "target_meta": t_(np.stack(tgt_meta)),
        "pool_meta": t_(np.stack(pool_meta)), "pool_win": t_(np.stack(pool_win)),
        "pool_win_mask": t_(np.stack(pool_win_mask), torch.bool),
        "pool_sim": t_(np.stack(pool_sim)), "pool_valid": t_(np.stack(pool_valid), torch.bool),
        "collab_users": torch.as_tensor(collab_arr, device=device),
        "collab_mask": torch.as_tensor(collab_mask, device=device),
        "hist_meta": t_(hist_arr), "hist_mask": t_(hist_mask, torch.bool),
        "n_it": t_(n_it), "n_collab": t_(n_collab), "n_uniq_collab": t_(n_uniq_collab),
    }
