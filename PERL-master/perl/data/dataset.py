"""PERL data interface.

Loads the dataprocessing artifacts and exposes the cutoff-based slice functions used by
the model. EVERY slice is strictly causal ("< t"): nothing at or after the cutoff is ever
returned, identically for train / valid / test.

Core objects (all keyed by raw item/user id):
  series[i]      : float32 (L_i, d_x)   normalized behavioral channels, lifecycle_idx 1..L_i
  t0[i]          : pandas.Timestamp     item launch time
  item_inter[i]  : (times_ns sorted, users, labels) interactions with item i
  volume[i]      : float32 (L_i,)       raw aggregate interaction volume per interval
  user_inter[u]  : (times_ns sorted, items) interactions by user u
  meta_vec[i]    : float32 (meta_dim,)  static metadata vector for f_M / user history

Functions:
  get_prefix(i, t)      -> S_{i,t}  (L_{i,t}, d_x)      intervals with interval_end_ts <= t
  get_collab(i, t)      -> R_{i,<t} (users, times, labels) interactions with i strictly before t
  get_volume(i, t)      -> raw aggregate volumes of the observed intervals (for Stab)
  get_user_hist(u, t)   -> items the user engaged before t (most-recent-first, for e_{u,t})
  get_pool(i, t, rho)   -> H_{i,t} = { j != i : t0_j < t and L_{j,t} >= rho }, restricted to
                           training-era items (no validation / test items are ever retrieved)
  get_window(j, rho, w) -> [x_{j,tau}] for tau in the causal lifecycle window N_w(rho)
  get_aligned_window(j, rho, w, t, strategy, max_len)
                        -> analog window under lifecycle_causal | symmetric | calendar |
                           whole_trajectory alignment (all strictly before t)
  L_at(i, t)            -> number of observed lifecycle intervals before t (L_{i,t})
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ONE_DAY_NS = 86_400_000_000_000


class PERLData:
    def __init__(self, processed_dir, interval_days: int = 1, meta_num_cat: int = 31):
        d = Path(processed_dir)
        self.interval_ns = interval_days * ONE_DAY_NS
        self.meta_num_cat = meta_num_cat

        self.meta = pd.read_parquet(d / "item_meta.parquet")
        series = pd.read_parquet(d / "item_series.parquet")
        inter = pd.read_parquet(d / "interactions.parquet")
        self.norm_cols = [c for c in series.columns if c.endswith("_norm")]
        self.d_x = len(self.norm_cols)

        # ---- per-item normalized series arrays (lifecycle_idx 1..L_i) ----
        self.series, self.t0, self.L_i, self.volume = {}, {}, {}, {}
        # raw count channels (if present) define the per-interval aggregate volume used by
        # the trajectory-stability indicator; ratio-type channels are excluded
        raw_cols = [c[:-5] for c in self.norm_cols
                    if c[:-5] in series.columns and not c[:-5].endswith("progress")]
        series = series.sort_values(["video_id", "lifecycle_idx"])
        for vid, g in series.groupby("video_id"):
            self.series[int(vid)] = g[self.norm_cols].to_numpy(np.float32)
            self.L_i[int(vid)] = len(g)
            self.volume[int(vid)] = (g[raw_cols].to_numpy(np.float64).clip(min=0).sum(1).astype(np.float32)
                                     if raw_cols else np.zeros(len(g), np.float32))
        for _, r in self.meta.iterrows():
            self.t0[int(r["video_id"])] = pd.Timestamp(r["t0"]).value  # ns

        # ---- metadata vectors: [log1p(duration)] + category multi-hot ----
        self.meta_vec = {}
        for _, r in self.meta.iterrows():
            onehot = np.zeros(self.meta_num_cat, np.float32)
            cf = r["category_feats"]
            cf = [] if cf is None else list(cf)
            for c in cf:
                if 0 <= int(c) < self.meta_num_cat:
                    onehot[int(c)] = 1.0
            self.meta_vec[int(r["video_id"])] = np.concatenate(
                [[np.log1p(max(float(r["duration"]), 0.0))], onehot]).astype(np.float32)
        self.meta_dim = 1 + self.meta_num_cat

        # ---- interactions indexed by item and by user (sorted times) ----
        inter = inter.sort_values("ts")
        inter["ts_ns"] = inter["ts"].astype("int64")
        self.item_inter, self.user_inter = {}, {}
        if "label" not in inter.columns:
            inter["label"] = 1
        for vid, g in inter.groupby("video_id"):
            self.item_inter[int(vid)] = (g["ts_ns"].to_numpy(), g["user_id"].to_numpy(),
                                         g["label"].to_numpy(np.int64))
        for uid, g in inter.groupby("user_id"):
            self.user_inter[int(uid)] = (g["ts_ns"].to_numpy(), g["video_id"].to_numpy())

        # ---- retrieval repository: training-era items only, t0-sorted for "t0_j < t" ----
        if "split" in self.meta.columns:
            sp = dict(zip(self.meta["video_id"].astype(int), self.meta["split"]))
            tr_t0 = [self.t0[i] for i, s in sp.items() if s == "train"]
            train_end = max(tr_t0) if tr_t0 else np.iinfo(np.int64).max
            elig = [i for i, s in sp.items()
                    if s == "train" or (s == "pool" and self.t0[i] <= train_end)]
        else:
            elig = list(self.t0)
        self._pool_ids = np.array(sorted(elig, key=lambda k: self.t0[k]), dtype=np.int64)
        self._pool_t0 = np.array([self.t0[i] for i in self._pool_ids])

        self.n_users = int(inter["user_id"].max()) + 1
        self.n_items = int(self.meta["video_id"].max()) + 1

    # ---------------------------- slice functions -------------------------
    def L_at(self, i: int, t_ns: int) -> int:
        """Number of lifecycle intervals fully observed before cutoff t (L_{i,t})."""
        if i not in self.t0:
            return 0
        n = int((t_ns - self.t0[i]) // self.interval_ns)
        return int(np.clip(n, 0, self.L_i.get(i, 0)))

    def get_prefix(self, i: int, t_ns: int) -> np.ndarray:
        """S_{i,t}: observed behavioral prefix (L_{i,t}, d_x). Empty if zero-history."""
        L = self.L_at(i, t_ns)
        s = self.series.get(i)
        if s is None or L == 0:
            return np.zeros((0, self.d_x), np.float32)
        return s[:L]

    def get_window(self, j: int, rho: int, w: int) -> np.ndarray:
        """Causal lifecycle window N_w(rho) = [max(1,rho-w+1) .. rho] of historical item j."""
        s = self.series.get(j)
        if s is None:
            return np.zeros((0, self.d_x), np.float32)
        hi = min(rho, len(s))
        lo = max(1, rho - w + 1)
        if hi < lo:
            return np.zeros((0, self.d_x), np.float32)
        return s[lo - 1:hi]

    def get_aligned_window(self, j: int, rho: int, w: int, t_ns: int, strategy: str,
                           max_len: int) -> np.ndarray:
        """Behavioral window of historical item j transferred to a target at lifecycle
        position rho, under one of four alignment strategies. Every strategy returns only
        intervals of j already observed before the cutoff t (L_{j,t} = L_at(j, t)):
          lifecycle_causal : tau in [max(1, rho-w+1), rho]                       (default)
          symmetric        : tau in [rho - w//2, rho + w//2], clipped to [1, L_{j,t}]
          calendar         : the w most recent intervals of j before t
          whole_trajectory : the whole observed trajectory of j before t (last max_len)
        """
        s = self.series.get(j)
        Lj = self.L_at(j, t_ns)
        if s is None or Lj == 0:
            return np.zeros((0, self.d_x), np.float32)
        if strategy == "lifecycle_causal":
            lo, hi = max(1, rho - w + 1), min(rho, Lj)
        elif strategy == "symmetric":
            half = w // 2
            lo, hi = max(1, rho - half), min(rho + half, Lj)
        elif strategy == "calendar":
            lo, hi = max(1, Lj - w + 1), Lj
        elif strategy == "whole_trajectory":
            lo, hi = max(1, Lj - max_len + 1), Lj
        else:
            raise ValueError(f"unknown align_strategy: {strategy}")
        if hi < lo:
            return np.zeros((0, self.d_x), np.float32)
        return s[lo - 1:hi]

    def get_collab(self, i: int, t_ns: int):
        """R_{i,<t}: (users, times_ns, labels) of interactions with i strictly before t."""
        rec = self.item_inter.get(i)
        if rec is None:
            z = np.zeros(0, np.int64)
            return z, z, z
        times, users, labels = rec
        k = np.searchsorted(times, t_ns, side="left")  # strictly < t
        return users[:k], times[:k], labels[:k]

    def get_volume(self, i: int, t_ns: int) -> np.ndarray:
        """Raw aggregate interaction volume of the L_{i,t} observed intervals (for Stab)."""
        L = self.L_at(i, t_ns)
        v = self.volume.get(i)
        if v is None or L == 0:
            return np.zeros(0, np.float32)
        return v[:L]

    def get_user_hist(self, u: int, t_ns: int, n: int | None = None):
        """Items the user engaged strictly before t, most-recent-first (for e_{u,t})."""
        rec = self.user_inter.get(u)
        if rec is None:
            return np.zeros(0, np.int64)
        times, items = rec
        k = np.searchsorted(times, t_ns, side="left")
        hist = items[:k][::-1]
        return hist if n is None else hist[:n]

    def get_pool(self, i: int, t_ns: int, rho: int) -> np.ndarray:
        """H_{i,t} = { j != i : t0_j < t and L_{j,t} >= rho } over training-era items."""
        k = np.searchsorted(self._pool_t0, t_ns, side="left")  # t0_j < t
        cand = self._pool_ids[:k]
        if len(cand) == 0:
            return cand
        # L_{j,t} >= rho  <=>  (t - t0_j) // interval >= rho
        Lj = (t_ns - self._pool_t0[:k]) // self.interval_ns
        mask = (Lj >= rho) & (cand != i)
        return cand[mask]
