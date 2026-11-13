#!/usr/bin/env python3
"""
Build a tiny KuaiRec_example dataset for an end-to-end smoke run of the whole pipeline.

Output: data/KuaiRec_example/ with the four CSVs read by dataprocessing.py, filtered to the
selected videos (user_features.csv is not needed by the pipeline and is not copied).

Usage:  python3 -I make_example_dataset.py --n 40
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE / "data" / "KuaiRec"
DST = HERE / "data" / "KuaiRec_example"
WIN_START, WIN_END = pd.Timestamp("2020-07-05"), pd.Timestamp("2020-09-05")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="number of videos to keep (30-50)")
    ap.add_argument("--min_life", type=int, default=20, help="min lifecycle length (days)")
    ap.add_argument("--min_cov", type=float, default=0.95, help="min daily nonzero coverage")
    args = ap.parse_args()
    DST.mkdir(parents=True, exist_ok=True)

    # ---- per-video density metrics from item_daily_features (fast) ----
    idf = pd.read_csv(SRC / "item_daily_features.csv",
                      usecols=["video_id", "date", "upload_dt", "show_cnt"])
    idf["d"] = pd.to_datetime(idf["date"].astype(str), format="%Y%m%d")
    upl = pd.to_datetime(idf.groupby("video_id")["upload_dt"].first(), errors="coerce").dt.normalize()
    in_win = upl[(upl >= WIN_START) & (upl <= WIN_END)].index

    rows = []
    for vid, g in idf[idf["video_id"].isin(in_win)].groupby("video_id"):
        t0 = upl.loc[vid]
        L = (WIN_END - t0).days + 1
        if L <= 0:
            continue
        cov = int((g["show_cnt"] > 0).sum()) / L
        rows.append((vid, t0, L, cov, float(g["show_cnt"].sum())))
    stats = pd.DataFrame(rows, columns=["video_id", "t0", "L", "cov", "show_sum"])

    # densest candidates: long lifecycle + near-full coverage, ranked by platform activity
    cand = stats[(stats["L"] >= args.min_life) & (stats["cov"] >= args.min_cov)]
    cand = cand.sort_values("show_sum", ascending=False).head(max(args.n * 5, 100))
    # spread across launch dates so train/valid/test (by upload_dt) are non-empty
    cand = cand.sort_values("t0")
    idx = np.linspace(0, len(cand) - 1, args.n).round().astype(int)
    sel = cand.iloc[np.unique(idx)]
    sel_ids = set(sel["video_id"].tolist())
    print(f"candidates={len(cand)}  selected={len(sel_ids)} videos")
    print(f"  lifecycle L: min={sel['L'].min()} max={sel['L'].max()} median={int(sel['L'].median())}")
    print(f"  cov: min={sel['cov'].min():.3f}  launch span: {sel['t0'].min().date()} .. {sel['t0'].max().date()}")

    # ---- filter item_daily_features (keep full daily rows for selected videos) ----
    idf_full = pd.read_csv(SRC / "item_daily_features.csv")
    idf_full[idf_full["video_id"].isin(sel_ids)].to_csv(DST / "item_daily_features.csv", index=False)

    # ---- filter big_matrix (all interactions for selected videos) via streaming ----
    parts = []
    for ch in pd.read_csv(SRC / "big_matrix.csv", chunksize=2_000_000):
        parts.append(ch[ch["video_id"].isin(sel_ids)])
    bm = pd.concat(parts, ignore_index=True)
    bm.to_csv(DST / "big_matrix.csv", index=False)
    print(f"  big_matrix rows kept: {len(bm):,}  users: {bm['user_id'].nunique():,}")

    # ---- metadata CSVs ----
    pd.read_csv(SRC / "item_categories.csv").query("video_id in @sel_ids").to_csv(
        DST / "item_categories.csv", index=False)
    try:
        capt = pd.read_csv(SRC / "kuairec_caption_category.csv", engine="python", on_bad_lines="skip")
        capt[capt["video_id"].isin(sel_ids)].to_csv(DST / "kuairec_caption_category.csv", index=False)
    except Exception as e:
        print("  caption filter skipped:", e)
        pd.DataFrame(columns=["video_id", "caption", "first_level_category_name"]).to_csv(
            DST / "kuairec_caption_category.csv", index=False)
    print(f"DONE -> {DST}")


if __name__ == "__main__":
    main()
