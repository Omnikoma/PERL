#!/usr/bin/env python3
"""
PERL — KuaiRec data processing (single-file pipeline).

Produces the model-ready artifacts consumed by perl/data/dataset.py:
  interactions.parquet     user_id, item_id, ts, label, split
  item_meta.parquet        item_id, t0, duration, category_feats, caption, split, ...
  item_series.parquet      item_id, lifecycle_idx (1..L_i continuous), interval_start/end_ts,
                           d_x raw channels + *_norm channels, imputed
  instances_{train,valid,test}.parquet  user_id, item_id, cutoff_ts, L_it, stage, label
  norm_stats.json, filter_funnel.md, dataset_stats.{md,json}, config.yaml,
  decision_record.md, acceptance_gate.json

Preprocessing decisions:
  - t0 = physical upload_dt (9,566 videos uploaded in-window; cold-start candidates)
  - interval = DAILY (Stage-A nonzero coverage median = 1.0)
  - positive label = watch_ratio >= 2.0  (strong-engagement conversion)
  - x_{i,l} = 16 multivariate DAILY channels (item_daily_features + big_matrix daily aggregates)

Usage:
  python3 -I dataprocessing.py --stage all          # Stage A diag + B-E build
  python3 -I dataprocessing.py --stage bcde         # build artifacts only
  python3 -I dataprocessing.py --stage a            # diagnostics only
  python3 -I dataprocessing.py --data-dir ... --out-dir ...

All thresholds are exposed in CFG and can be overridden from the command line.
"""
import argparse
import ast
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------- configuration ------------------------------
HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "data" / "KuaiRec"
DEFAULT_OUT = HERE / "processed" / "kuairec"

CFG = dict(
    watch_theta=2.0,
    win_start="2020-07-05", win_end="2020-09-05",   # daily-features window (x_{i,l} bound)
    kcore_k=5,
    burn_in_days=3,           # targets must upload >= win_start + burn_in
    l_min_days=3,             # keep target items with lifecycle length >= this
    split_q=(0.45, 0.55),     # upload_dt quantiles -> train(45)/valid(10)/test(45)
    max_cutoffs_per_item=16,
    n_neg=4,
    max_pos_per_cutoff=50,
    seed=42,
)

IDF_COUNT_COLS = ["show_cnt", "show_user_num", "play_cnt", "play_user_num", "play_duration",
                  "complete_play_cnt", "valid_play_cnt", "like_cnt", "comment_cnt",
                  "share_cnt", "collect_cnt", "follow_cnt"]
IDF_RATIO_COLS = ["play_progress"]
BM_COUNT_COLS = ["bm_exposure_cnt", "bm_user_num", "bm_pos_cnt"]
COUNT_COLS = IDF_COUNT_COLS + BM_COUNT_COLS
RATIO_COLS = IDF_RATIO_COLS
CHANNELS = COUNT_COLS + RATIO_COLS  # d_x = 16
STAGES = [(0, 3, "G1"), (4, 7, "G2"), (8, 14, "G3"), (15, 10**9, "G4")]


def stage_of(L):
    for lo, hi, n in STAGES:
        if lo <= L <= hi:
            return n
    return "G4"


def kcore(df, k, ucol="user_id", icol="video_id"):
    rounds = []
    while True:
        uc = df[ucol].value_counts(); ic = df[icol].value_counts()
        ku = set(uc[uc >= k].index); ki = set(ic[ic >= k].index)
        before = len(df)
        df = df[df[ucol].isin(ku) & df[icol].isin(ki)]
        rounds.append((before, len(df), int(df[ucol].nunique()), int(df[icol].nunique())))
        if len(df) == before:
            break
    return df, rounds


# ----------------------------- Stage A: diagnostics -----------------------
def stage_a(data_dir, out_dir, log):
    """Lightweight read-only diagnostics (t0 in/out window, daily coverage)."""
    data_dir = Path(data_dir)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    win_s, win_e = pd.Timestamp(CFG["win_start"]), pd.Timestamp(CFG["win_end"])
    idf = pd.read_csv(data_dir / "item_daily_features.csv",
                      usecols=["video_id", "date", "upload_dt", "show_cnt"])
    idf["d"] = pd.to_datetime(idf["date"].astype(str), format="%Y%m%d")
    upl = pd.to_datetime(idf.groupby("video_id")["upload_dt"].first(), errors="coerce").dt.normalize()
    in_win = (upl >= win_s) & (upl <= win_e)
    cand = set(upl[in_win].index)
    # daily nonzero coverage for candidates
    covs = []
    for vid, g in idf[idf["video_id"].isin(cand)].groupby("video_id"):
        t0i = upl.loc[vid]
        L = (win_e - t0i).days + 1
        if L <= 0:
            continue
        covs.append(int((g["show_cnt"] > 0).sum()) / L)
    covs = np.array(covs)
    rep = {"in_window_cold_candidates": int(in_win.sum()),
           "out_window": int((~in_win).sum()),
           "daily_cov_median": round(float(np.median(covs)), 4),
           "daily_cov_ge_0.8": int((covs >= 0.8).sum())}
    (out_dir / "stage_a_report.json").write_text(json.dumps(rep, indent=2))
    log(f"Stage A: {rep}")
    return rep


# ----------------------------- Stage B-E: build ---------------------------
def build(data_dir, out_dir, log):
    t0 = time.time()
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    funnel = []
    win_start = pd.Timestamp(CFG["win_start"])
    win_end_excl = pd.Timestamp(CFG["win_end"]) + pd.Timedelta(days=1)
    win_end_day = pd.Timestamp(CFG["win_end"])

    log("loading item_daily_features ...")
    idf = pd.read_csv(data_dir / "item_daily_features.csv",
                      usecols=["video_id", "date", "upload_dt"] + IDF_COUNT_COLS + IDF_RATIO_COLS)
    idf["d"] = pd.to_datetime(idf["date"].astype(str), format="%Y%m%d")
    upl = pd.to_datetime(idf.groupby("video_id")["upload_dt"].first(), errors="coerce").dt.normalize()
    in_win = (upl >= win_start) & (upl < win_end_excl)
    cand_items = set(upl[in_win].index)
    log(f"in-window cold candidates: {len(cand_items):,}")

    log("loading big_matrix (interactions) ...")
    bm = pd.read_csv(data_dir / "big_matrix.csv",
                     usecols=["user_id", "video_id", "time", "watch_ratio"])
    bm["ts"] = pd.to_datetime(bm["time"])
    bm = bm[(bm["ts"] >= win_start) & (bm["ts"] < win_end_excl)]
    bm["label"] = (bm["watch_ratio"] >= CFG["watch_theta"]).astype(int)
    bm = bm[bm["video_id"].isin(cand_items)].copy()
    funnel.append(("window-clip + in-window items", len(bm),
                   bm["user_id"].nunique(), bm["video_id"].nunique()))

    bm, kc_rounds = kcore(bm, CFG["kcore_k"])
    funnel.append((f"k-core(k={CFG['kcore_k']})", len(bm),
                   bm["user_id"].nunique(), bm["video_id"].nunique()))
    kept_items = set(bm["video_id"].unique())
    log(f"after k-core: rows={len(bm):,} users={bm['user_id'].nunique():,} items={len(kept_items):,}")

    # big_matrix daily aggregates
    bm["day"] = bm["ts"].dt.normalize()
    bm_daily = bm.groupby(["video_id", "day"]).agg(
        bm_exposure_cnt=("user_id", "size"), bm_user_num=("user_id", "nunique"),
        bm_pos_cnt=("label", "sum")).reset_index()

    log("building item_series (continuous l=1..L_i, 16 channels) ...")
    idf_k = idf[idf["video_id"].isin(kept_items)].copy()
    idf_g = {v: g for v, g in idf_k.groupby("video_id")}
    bmd_g = {v: g for v, g in bm_daily.groupby("video_id")}
    upl_day = {v: upl.loc[v] for v in kept_items}
    series_rows, day_count = [], {}
    for vid in kept_items:
        t0i = upl_day[vid]
        L = (win_end_day - t0i).days + 1
        if L <= 0:
            continue
        days = pd.date_range(t0i, periods=L, freq="D")
        base = pd.DataFrame({"video_id": vid, "day": days, "lifecycle_idx": np.arange(1, L + 1)})
        g = idf_g.get(vid)
        if g is not None:
            base = base.merge(g[["d"] + IDF_COUNT_COLS + IDF_RATIO_COLS].rename(columns={"d": "day"}),
                              on="day", how="left")
        bd = bmd_g.get(vid)
        if bd is not None:
            base = base.merge(bd.drop(columns="video_id"), on="day", how="left")
        base["imputed"] = base[IDF_COUNT_COLS[0]].isna().astype(int) if IDF_COUNT_COLS[0] in base else 1
        for c in CHANNELS:
            if c not in base:
                base[c] = 0.0
        base[CHANNELS] = base[CHANNELS].fillna(0.0)
        day_count[vid] = L
        series_rows.append(base)
    series = pd.concat(series_rows, ignore_index=True)
    series["interval_start_ts"] = series["day"]
    series["interval_end_ts"] = series["day"] + pd.Timedelta(days=1)
    log(f"series rows: {len(series):,} over {series['video_id'].nunique():,} items")

    # split by upload_dt quantile over eligible targets
    burn = win_start + pd.Timedelta(days=CFG["burn_in_days"])
    elig = [v for v in kept_items
            if upl_day[v] >= burn and day_count.get(v, 0) >= CFG["l_min_days"]]
    q1, q2 = np.quantile(np.array([np.int64(upl_day[v].value) for v in elig]), CFG["split_q"])
    def split_of(v):
        x = np.int64(upl_day[v].value)
        return "train" if x <= q1 else ("valid" if x <= q2 else "test")
    item_split = {v: split_of(v) for v in elig}
    log(f"eligible targets: {len(elig):,}  split={pd.Series(item_split).value_counts().to_dict()}")
    funnel.append(("eligible targets (burn-in + L>=l_min)", len(elig), 0, len(elig)))

    # lifecycle-stratified instances
    log("generating instances ...")
    rng = np.random.default_rng(CFG["seed"])
    all_users = bm["user_id"].unique()
    bm["lc"] = ((bm["day"] - bm["video_id"].map(upl_day)).dt.days + 1)
    bm_g = {v: g for v, g in bm.groupby("video_id")}
    inst = {s: [] for s in ("train", "valid", "test")}
    for vid, split in item_split.items():
        t0i, Li = upl_day[vid], day_count[vid]
        by_int = {lc: g for lc, g in bm_g[vid].groupby("lc")}
        cands = list(range(0, min(Li - 1, CFG["max_cutoffs_per_item"])))
        if Li - 1 > CFG["max_cutoffs_per_item"]:
            cands = sorted(rng.choice(range(0, Li - 1), CFG["max_cutoffs_per_item"], replace=False).tolist())
        for Lit in cands:
            g = by_int.get(Lit + 1)
            if g is None:
                continue
            pos_u = g.loc[g["label"] == 1, "user_id"].unique()
            neg_u = g.loc[g["label"] == 0, "user_id"].unique()
            if len(pos_u) == 0:
                continue
            if len(pos_u) > CFG["max_pos_per_cutoff"]:
                pos_u = rng.choice(pos_u, CFG["max_pos_per_cutoff"], replace=False)
            # cutoff = end of the last OBSERVED interval (= start of the label interval L_it+1);
            # get_prefix(i, cutoff) returns intervals with interval_end_ts <= cutoff => exactly
            # the L_it observed intervals, never the label interval (strict causality, no leak).
            cutoff_ts = t0i + pd.Timedelta(days=Lit)
            st = stage_of(Lit)
            need = len(pos_u) * CFG["n_neg"]
            negs = (rng.choice(neg_u, need, replace=False) if len(neg_u) >= need
                    else np.concatenate([neg_u, rng.choice(all_users, need - len(neg_u), replace=True)]))
            for u in pos_u:
                inst[split].append((u, vid, cutoff_ts, Lit, st, 1))
            for u in negs:
                inst[split].append((u, vid, cutoff_ts, Lit, st, 0))
    inst_df = {s: pd.DataFrame(inst[s], columns=["user_id", "video_id", "cutoff_ts",
                                                 "L_it", "stage", "label"]) for s in inst}
    for s in inst_df:
        d = inst_df[s]
        log(f"  {s}: {len(d):,} instances, {d['video_id'].nunique() if len(d) else 0} items")

    # normalization fit on train item series
    train_items = {v for v, s in item_split.items() if s == "train"}
    tr = series[series["video_id"].isin(train_items)]
    norm = {}
    for c in COUNT_COLS:
        v = np.log1p(np.clip(tr[c].to_numpy(np.float64), 0, None))
        mu, sd = float(v.mean()), float(v.std() or 1.0)
        norm[c] = {"transform": "log1p_zscore", "mean": mu, "std": sd}
        series[c + "_norm"] = (np.log1p(np.clip(series[c].to_numpy(np.float64), 0, None)) - mu) / sd
    for c in RATIO_COLS:
        v = tr[c].to_numpy(np.float64)
        mu, sd = float(v.mean()), float(v.std() or 1.0)
        norm[c] = {"transform": "zscore", "mean": mu, "std": sd}
        series[c + "_norm"] = (series[c].to_numpy(np.float64) - mu) / sd

    # item_meta
    log("building item_meta ...")
    cats = pd.read_csv(data_dir / "item_categories.csv")
    cats["feat_list"] = cats["feat"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
    try:
        capt = pd.read_csv(data_dir / "kuairec_caption_category.csv", engine="python",
                           on_bad_lines="skip",
                           usecols=lambda c: c in ("video_id", "caption", "first_level_category_name"))
    except Exception:
        capt = pd.DataFrame(columns=["video_id", "caption", "first_level_category_name"])
    dur = pd.read_csv(data_dir / "item_daily_features.csv",
                      usecols=["video_id", "video_duration"]).groupby("video_id")["video_duration"].median()
    meta = pd.DataFrame({"video_id": sorted(kept_items)})
    meta["t0"] = meta["video_id"].map(lambda v: upl_day[v])
    meta["duration"] = meta["video_id"].map(dur).fillna(0.0)
    fl = cats.set_index("video_id")["feat_list"]
    meta["category_feats"] = meta["video_id"].map(lambda v: fl.get(v, []))
    ci = capt.set_index("video_id") if len(capt) else None
    meta["caption"] = meta["video_id"].map(
        lambda v: (ci["caption"].get(v) if ci is not None and v in ci.index else None))
    meta["first_level_category"] = meta["video_id"].map(
        lambda v: (ci["first_level_category_name"].get(v) if ci is not None and v in ci.index else None))
    meta["split"] = meta["video_id"].map(lambda v: item_split.get(v, "pool"))
    meta["is_cold_target"] = meta["video_id"].isin(item_split).astype(int)
    meta["L_i"] = meta["video_id"].map(day_count).fillna(0).astype(int)

    # write artifacts
    log("writing artifacts ...")
    bm_out = bm[["user_id", "video_id", "ts", "label"]].copy()
    bm_out["split"] = bm_out["video_id"].map(lambda v: item_split.get(v, "pool"))
    bm_out.to_parquet(out_dir / "interactions.parquet", index=False)
    meta.to_parquet(out_dir / "item_meta.parquet", index=False)
    scols = (["video_id", "lifecycle_idx", "interval_start_ts", "interval_end_ts", "imputed"]
             + CHANNELS + [c + "_norm" for c in CHANNELS])
    series[scols].to_parquet(out_dir / "item_series.parquet", index=False)
    for s in inst_df:
        inst_df[s].to_parquet(out_dir / f"instances_{s}.parquet", index=False)
    (out_dir / "norm_stats.json").write_text(json.dumps(norm, indent=2))

    _write_funnel(out_dir, funnel, kc_rounds)
    _write_stats(out_dir, bm, series, meta, inst_df, item_split, day_count)
    (out_dir / "config.yaml").write_text(
        "\n".join([f"{k}: {v}" for k, v in CFG.items()] + [f"channels: {CHANNELS}"]) + "\n")

    tst = inst_df["test"]
    g1 = int((tst["stage"] == "G1").sum()); g2 = int((tst["stage"] == "G2").sum())
    gate = {"test_items": int(tst["video_id"].nunique()), "test_instances": int(len(tst)),
            "G1": g1, "G2": g2,
            "pass": bool(tst["video_id"].nunique() >= 3000 and len(tst) >= 1e5 and g1 >= 2e4 and g2 >= 2e4)}
    (out_dir / "acceptance_gate.json").write_text(json.dumps(gate, indent=2))
    log(f"ACCEPTANCE GATE: {gate}")
    log(f"DONE build total={time.time()-t0:.0f}s -> {out_dir}")
    return gate


def _write_funnel(out_dir, funnel, kc):
    L = ["# KuaiRec — Filter Funnel\n", "| step | rows | users | items |", "|---|---|---|---|"]
    for n, r, u, i in funnel:
        L.append(f"| {n} | {r:,} | {u:,} | {i:,} |")
    L.append("\n## k-core rounds")
    for b, a, u, i in kc:
        L.append(f"- {b:,} -> {a:,} (users={u:,}, items={i:,})")
    (out_dir / "filter_funnel.md").write_text("\n".join(L) + "\n")


def _write_stats(out_dir, bm, series, meta, inst_df, item_split, day_count):
    nU, nI, nInter = bm["user_id"].nunique(), bm["video_id"].nunique(), len(bm)
    nPos = int(bm["label"].sum())
    s = {"basic": {"users": int(nU), "items": int(nI), "interactions": int(nInter),
                   "positives": nPos, "pos_rate": round(nPos / nInter, 4),
                   "sparsity": round(1 - nInter / (nU * nI), 6)},
         "lifecycle": {"interval": "daily", "d_x": len(CHANNELS), "channels": CHANNELS,
                       "avg_L_days": round(float(np.mean(list(day_count.values()))), 2),
                       "imputed_frac": round(float(series["imputed"].mean()), 4)},
         "splits": {sp: {"target_items": int(sum(1 for v in item_split.values() if v == sp)),
                         "instances": int(len(inst_df[sp])),
                         "stages": {k: int(v) for k, v in inst_df[sp]["stage"].value_counts().items()}}
                    for sp in ("train", "valid", "test")},
         "cold": {"cold_target_items": int(meta["is_cold_target"].sum())}}
    (out_dir / "dataset_stats.json").write_text(json.dumps(s, indent=2, ensure_ascii=False))
    md = [f"# KuaiRec — Dataset Stats\n", "## Basic",
          f"- users {nU:,} · items {nI:,} · interactions {nInter:,} · pos_rate {s['basic']['pos_rate']}",
          "\n## Splits", "| split | items | instances | stages |", "|---|---|---|---|"]
    for sp in ("train", "valid", "test"):
        d = s["splits"][sp]
        md.append(f"| {sp} | {d['target_items']:,} | {d['instances']:,} | {d['stages']} |")
    (out_dir / "dataset_stats.md").write_text("\n".join(md) + "\n")
    _write_decision(out_dir)


def _write_decision(out_dir):
    (out_dir / "decision_record.md").write_text(
        "# KuaiRec — Decision Record\n\n"
        "- t0 = physical upload_dt (in-window cold candidates).\n"
        "- interval = daily (Stage-A nonzero coverage median 1.0).\n"
        "- positive label = watch_ratio >= 2.0.\n"
        "- x_{i,l} = 16 multivariate daily channels (item_daily_features + big_matrix daily agg).\n"
        f"- k-core k={CFG['kcore_k']}; burn-in {CFG['burn_in_days']}d; L_min {CFG['l_min_days']}d; "
        f"split quantiles {CFG['split_q']}.\n"
        "- instances: one cutoff per lifecycle interval (cap "
        f"{CFG['max_cutoffs_per_item']}), label from next interval, n_neg={CFG['n_neg']}.\n")


def main():
    ap = argparse.ArgumentParser(description="PERL KuaiRec data processing")
    ap.add_argument("--stage", choices=["a", "bcde", "all"], default="all")
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    # threshold overrides (e.g. for the tiny KuaiRec_example subset)
    ap.add_argument("--kcore-k", type=int, default=None)
    ap.add_argument("--l-min-days", type=int, default=None)
    ap.add_argument("--burn-in-days", type=int, default=None)
    ap.add_argument("--max-cutoffs-per-item", type=int, default=None)
    args = ap.parse_args()
    for k_cli, k_cfg in (("kcore_k", "kcore_k"), ("l_min_days", "l_min_days"),
                         ("burn_in_days", "burn_in_days"),
                         ("max_cutoffs_per_item", "max_cutoffs_per_item")):
        v = getattr(args, k_cli)
        if v is not None:
            CFG[k_cfg] = v
    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
    if args.stage in ("a", "all"):
        stage_a(args.data_dir, args.out_dir, log)
    if args.stage in ("bcde", "all"):
        build(args.data_dir, args.out_dir, log)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    main()
