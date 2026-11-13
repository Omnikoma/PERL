"""Lifecycle masking for training.

For a training item i with full sequence length L_i, sample a pseudo lifecycle cutoff
l ~ Uniform{0, 1, ..., L_i} and truncate both the behavioral prefix S and the collaborative
history R to the corresponding timestamp t_l = t0_i + l * interval:
  l = 0        -> strict zero-history launch
  0 < l << L_i -> noisy few-shot cold-start
  l -> L_i     -> warm regime

The label for the masked instance is drawn from interval l+1 (the interval right after the
cutoff). We also draw a second l' != l per item to form the ordered (t_a < t_b) pairs needed
by the uncertainty-calibration loss L_unc.
"""
from __future__ import annotations

import numpy as np

ONE_DAY_NS = 86_400_000_000_000


def sample_cutoff(L_i: int, rng: np.random.Generator, interval_days: int = 1):
    """Return (l, l_prime) with l in {0..L_i-1} (needs a label interval l+1<=L_i) and l'!=l."""
    if L_i < 1:
        return 0, 0
    hi = max(L_i - 1, 0)                 # label interval l+1 must exist (<= L_i)
    l = int(rng.integers(0, hi + 1))
    lp = int(rng.integers(0, hi + 1))
    if lp == l:
        lp = (l + 1) % (hi + 1)
    return l, lp


def cutoff_ns(t0_ns: int, l: int, interval_days: int = 1) -> int:
    """Cutoff timestamp for observing exactly l intervals (end of interval l)."""
    return int(t0_ns + l * interval_days * ONE_DAY_NS)
