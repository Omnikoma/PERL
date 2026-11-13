"""Unit tests for PERL: causal slicing, path masking, loss edge cases.

Runs on CPU with a tiny synthetic processed dataset and the from-scratch GRU encoder (so the
tests do NOT need to download/load the 200M TimesFM weights). TimesFM-specific behavior is
exercised by the `--smoke` train/eval entry points instead.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from perl.config import Cfg
from perl.data.dataset import PERLData
from perl.data.collate import build_batch
from perl.model import PERL
from perl import losses

ONE_DAY_NS = 86_400_000_000_000


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    d = tmp_path_factory.mktemp("proc")
    t0 = pd.Timestamp("2020-07-05")
    items = [10, 11, 12, 13]
    dx_cols = [f"c{j}_norm" for j in range(3)]
    # series: item 10 has L=6, others L=4; continuous lifecycle_idx
    srows = []
    Ls = {10: 6, 11: 4, 12: 4, 13: 5}
    for it, L in Ls.items():
        for l in range(1, L + 1):
            srows.append({"video_id": it, "lifecycle_idx": l,
                          "interval_start_ts": t0 + pd.Timedelta(days=l - 1),
                          "interval_end_ts": t0 + pd.Timedelta(days=l), "imputed": 0,
                          **{c: float(l) for c in dx_cols}})
    pd.DataFrame(srows).to_parquet(d / "item_series.parquet")
    # meta
    meta = pd.DataFrame({"video_id": items,
                         "t0": [t0, t0 + pd.Timedelta(days=1), t0, t0],
                         "duration": [10.0, 20.0, 30.0, 40.0],
                         "category_feats": [[0], [1], [0, 1], [2]],
                         "caption": [None] * 4, "first_level_category": [None] * 4,
                         "split": ["train", "train", "test", "test"],
                         "is_cold_target": [1, 1, 1, 1], "L_i": [Ls[i] for i in items]})
    meta.to_parquet(d / "item_meta.parquet")
    # interactions: user 1 interacts with item 10 on day0 and day3
    inter = pd.DataFrame({
        "user_id": [1, 2, 1, 3, 2],
        "video_id": [10, 10, 10, 11, 12],
        "ts": [t0, t0, t0 + pd.Timedelta(days=3), t0 + pd.Timedelta(days=1), t0],
        "label": [1, 0, 1, 1, 0], "split": ["train"] * 5})
    inter.to_parquet(d / "interactions.parquet")
    for sp in ("train", "valid", "test"):
        pd.DataFrame({"user_id": [1], "video_id": [10],
                      "cutoff_ts": [t0 + pd.Timedelta(days=2)], "L_it": [2],
                      "stage": ["G1"], "label": [1]}).to_parquet(d / f"instances_{sp}.parquet")
    return PERLData(d, interval_days=1, meta_num_cat=3)


def _cfg():
    return Cfg({
        "seed": 0,
        "model": {"embed_dim": 16, "K": 3, "w": 2, "K_P": 2, "tau_con": 0.1,
                  "tau_M": 0.1, "tau_P": 0.1, "rho_max": 8, "user_hist_len": 5},
        "path_a": {"encoder": "gru", "patch_len": 32, "hidden_size": 16,
                   "lora_r": 4, "lora_alpha": 8, "lora_dropout": 0.0,
                   "lora_target_modules": ["qkv_proj"], "max_prefix_len": 64},
        "loss": {"lambda_con": 0.1, "lambda_unc": 0.1, "lambda_reg": 1e-5},
        "optim": {"batch_size": 8},
        "ablation": {"no_lifecycle_align": False, "no_population": False, "no_temporal": False,
                     "no_collaborative": False, "no_uncertainty": False, "no_progressive": False,
                     "no_cecl": False, "align_strategy": "lifecycle_causal",
                     "fusion_strategy": "progressive"}})


# ----------------------------- causal slicing -----------------------------
def test_prefix_excludes_future(tiny):
    t0 = tiny.t0[10]
    # cutoff at end of interval 2 => exactly 2 observed intervals, none at/after cutoff
    assert tiny.L_at(10, t0 + 2 * ONE_DAY_NS) == 2
    S = tiny.get_prefix(10, t0 + 2 * ONE_DAY_NS)
    assert S.shape[0] == 2
    # zero history at launch
    assert tiny.get_prefix(10, t0).shape[0] == 0


def test_collab_strictly_before_t(tiny):
    t0 = tiny.t0[10]
    users, times, labels = tiny.get_collab(10, t0 + 1 * ONE_DAY_NS)   # only day0 interactions
    assert (times < t0 + 1 * ONE_DAY_NS).all()
    assert set(users.tolist()) == {1, 2}
    assert int(labels.sum()) == 1            # n_{i,t} counts positives only (user 1)


def test_pool_is_training_era_only(tiny):
    t0 = tiny.t0[10]
    pool = tiny.get_pool(11, t0 + 3 * ONE_DAY_NS, rho=1)
    assert set(pool.tolist()) == {10}        # items 12/13 are test items -> never retrieved


def test_prototype_selection_and_stab(tiny):
    cfg = _cfg()
    torch.manual_seed(0)
    model = PERL(cfg, tiny).eval()
    t0 = tiny.t0[10]
    b = build_batch(tiny, [(1, 11, int(t0 + 3 * ONE_DAY_NS), 1)], cfg, torch.device("cpu"))
    assert 0.0 <= float(b["stab"][0]) <= 1.0 and int(b["n_it"][0]) == 1
    p = model.path_p(b["pool_meta"], b["pool_win"], b["pool_win_mask"], b["pool_valid"],
                     b["target_meta"], b["rho"])
    assert torch.allclose(p["beta"].sum(-1), torch.ones(1))
    assert p["mu_P"].shape == (1, cfg["model"]["embed_dim"])


def test_pool_validity(tiny):
    t0 = tiny.t0[10]
    t = t0 + 5 * ONE_DAY_NS
    pool = tiny.get_pool(10, t, rho=1)
    for j in pool:
        assert tiny.t0[int(j)] < t                 # launched before cutoff
        assert tiny.L_at(int(j), t) >= 1           # reached lifecycle position
        assert int(j) != 10                        # excludes self


def test_window_is_causal(tiny):
    win = tiny.get_window(10, rho=3, w=2)           # intervals 2..3
    assert win.shape[0] == 2


# --------------------------- model edge cases -----------------------------
def test_mA_zero_gives_zero_muA_and_h1(tiny):
    cfg = _cfg()
    torch.manual_seed(0)
    model = PERL(cfg, tiny).eval()
    t0 = tiny.t0[10]
    # zero-history instance => m_A should be 0
    b = build_batch(tiny, [(1, 10, int(t0), 1)], cfg, torch.device("cpu"))
    out = model(b)
    assert out["m_A"].item() == 0.0
    assert out["mu_A"].abs().sum().item() == 0.0
    assert out["lambda_A"].abs().sum().item() == 0.0


def test_no_temporal_collab_switches(tiny):
    cfg = _cfg(); cfg["ablation"]["no_temporal"] = True; cfg["ablation"]["no_collaborative"] = True
    model = PERL(cfg, tiny).eval()
    t0 = tiny.t0[10]
    b = build_batch(tiny, [(1, 10, int(t0 + 3 * ONE_DAY_NS), 1)], cfg, torch.device("cpu"))
    out = model(b)
    assert out["m_A"].item() == 0.0 and out["m_C"].item() == 0.0
    assert out["mu_A"].abs().sum().item() == 0.0


# --------------------------- loss edge cases ------------------------------
def test_cecl_zero_when_no_pairs(tiny):
    cfg = _cfg()
    model = PERL(cfg, tiny).eval()
    t0 = tiny.t0[10]
    # zero-history => m_A=0 => no valid (P,A) or (A,C) pairs => CECL == 0
    b = build_batch(tiny, [(1, 10, int(t0), 1)], cfg, torch.device("cpu"))
    out = model(b)
    assert float(out["cecl_loss"]) == 0.0


def test_unc_zero_when_monotone():
    # sigma decreasing from t_a to t_b => hinge is 0
    oa = {"sigma_A": torch.tensor([[0.9, 0.9]]), "sigma_C": torch.tensor([[0.8, 0.8]]),
          "m_A": torch.tensor([1.0]), "m_C": torch.tensor([1.0]),
          "logits": torch.zeros(1)}
    ob = {"sigma_A": torch.tensor([[0.3, 0.3]]), "sigma_C": torch.tensor([[0.2, 0.2]])}
    assert float(losses.unc_loss(oa, ob, oa["m_A"])) == 0.0


# --------------------------- alignment / fusion variants -------------------
@pytest.mark.parametrize("strategy", ["lifecycle_causal", "symmetric", "calendar", "whole_trajectory"])
def test_align_strategies_are_causal(tiny, strategy):
    cfg = _cfg(); cfg["ablation"]["align_strategy"] = strategy
    t0 = tiny.t0[10]
    t = int(t0 + 3 * ONE_DAY_NS)          # target item 11 (launched t0+1d): L=2, rho=3
    b = build_batch(tiny, [(1, 11, t, 1)], cfg, torch.device("cpu"))
    W = cfg["path_a"]["max_prefix_len"] if strategy == "whole_trajectory" else cfg["model"]["w"]
    assert b["pool_win"].shape[1:] == (cfg["model"]["K"], W, tiny.d_x)
    assert b["pool_valid"].any()
    # synthetic series store the lifecycle index as the channel value, and every analog has
    # L_{j,t} = 3 at this cutoff, so no transferred step may carry a value > 3.
    for k in range(cfg["model"]["K"]):
        if not b["pool_valid"][0, k]:
            continue
        vals = b["pool_win"][0, k][b["pool_win_mask"][0, k]][:, 0]
        assert vals.numel() > 0 and (vals <= 3).all()


def test_no_lifecycle_align_is_whole_trajectory(tiny):
    cfg = _cfg(); cfg["ablation"]["no_lifecycle_align"] = True
    t0 = tiny.t0[10]
    b = build_batch(tiny, [(1, 11, int(t0 + 3 * ONE_DAY_NS), 1)], cfg, torch.device("cpu"))
    assert b["pool_win"].shape[2] == cfg["path_a"]["max_prefix_len"]


@pytest.mark.parametrize("fusion", ["mean", "concat", "reliability", "progressive"])
def test_fusion_strategies_forward(tiny, fusion):
    cfg = _cfg(); cfg["ablation"]["fusion_strategy"] = fusion
    torch.manual_seed(0)
    model = PERL(cfg, tiny).eval()
    t0 = tiny.t0[10]
    b = build_batch(tiny, [(1, 11, int(t0 + 3 * ONE_DAY_NS), 1), (2, 10, int(t0), 0)],
                    cfg, torch.device("cpu"))
    out = model(b)
    assert out["logits"].shape == (2,) and torch.isfinite(out["logits"]).all()
    if fusion != "progressive":
        assert out["lambda_A"].abs().sum().item() == 0.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
