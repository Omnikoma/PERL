"""PERL evaluation protocol.

Item-level cold-start test instances are grouped into four lifecycle stages by L_{i,t}:
  G1: 0-3, G2: 4-7, G3: 8-14, G4: >14.

Metrics:
  - AUC: per-instance, over the candidate set, reported Overall / per-stage / Early / Mature.
  - HR@10, NDCG@10: for each POSITIVE instance (u, i, t), a candidate set of {i} + 99 sampled
    negative items is scored at the same (u, t); the positive's rank gives HR@10 / NDCG@10.
  - Diagnostics: per-stage mean sigma^P/sigma^A/sigma^C and lambda^A/lambda^C.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from .config import load_config, set_seed, get_device
from .data.dataset import PERLData
from .data.collate import build_batch
from .model import PERL

STAGE_ORDER = ["G1", "G2", "G3", "G4"]


def _rows(df):
    return list(zip(df["user_id"].to_numpy(), df["video_id"].to_numpy(),
                    df["cutoff_ts"].astype("int64").to_numpy(),
                    df["label"].to_numpy(), df["stage"].to_numpy()))


@torch.no_grad()
def _score(model, data, rows4, cfg, device):
    """Score (u,i,cutoff) rows -> probabilities + diagnostics, batched."""
    bs = cfg.optim.batch_size
    probs, diag = [], {k: [] for k in ("sigma_P", "sigma_A", "sigma_C", "lambda_A", "lambda_C")}
    for i in range(0, len(rows4), bs):
        chunk = [(u, it, ct, y) for (u, it, ct, y) in rows4[i:i + bs]]
        b = build_batch(data, chunk, cfg, device)
        out = model(b)
        probs.append(torch.sigmoid(out["logits"]).float().cpu().numpy())
        for k in ("sigma_P", "sigma_A", "sigma_C"):
            diag[k].append(out[k].mean(dim=1).float().cpu().numpy())
        for k in ("lambda_A", "lambda_C"):
            diag[k].append(out[k].float().cpu().numpy())
    probs = np.concatenate(probs) if probs else np.zeros(0)
    diag = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in diag.items()}
    return probs, diag


@torch.no_grad()
def ranking_metrics(model, data, pos_rows, cfg, device, all_items, max_pos=2000):
    """HR@10 / NDCG@10: per positive, candidate = {i} + 99 sampled negative items at (u,t)."""
    rng = np.random.default_rng(cfg.seed)
    # cap negatives to the catalog size (small example datasets may have < 99 items)
    K = min(cfg.eval.num_candidate_negatives, max(len(all_items) - 1, 1))
    topk = cfg.eval.topk
    if len(pos_rows) > max_pos:
        pos_rows = [pos_rows[i] for i in rng.choice(len(pos_rows), max_pos, replace=False)]
    hr, ndcg = [], []
    bs = max(1, cfg.optim.batch_size // (K + 1))
    buf = []
    def flush(buf):
        rows = [r for grp in buf for r in grp]
        probs, _ = _score(model, data, rows, cfg, device)
        off = 0
        for grp in buf:
            n = len(grp)
            s = probs[off:off + n]; off += n
            rank = int((s > s[0]).sum())  # position of the positive (index 0)
            hr.append(1.0 if rank < topk else 0.0)
            ndcg.append(1.0 / np.log2(rank + 2) if rank < topk else 0.0)
    for (u, i, t, y, st) in pos_rows:
        negs = rng.choice(all_items, K, replace=False)
        grp = [(u, int(i), int(t), 1)] + [(u, int(j), int(t), 0) for j in negs]
        buf.append(grp)
        if len(buf) >= bs:
            flush(buf); buf = []
    if buf:
        flush(buf)
    return float(np.mean(hr)) if hr else 0.0, float(np.mean(ndcg)) if ndcg else 0.0


def evaluate(cfg, ckpt=None, smoke=False):
    set_seed(cfg.seed)
    device = get_device()
    proc = Path(cfg.paths.processed_dir)
    data = PERLData(proc, interval_days=1, meta_num_cat=31)
    test = pd.read_parquet(proc / "instances_test.parquet")
    if smoke:
        test = test.groupby("stage", group_keys=False).head(300)
    rows = _rows(test)

    model = PERL(cfg, data).to(device)
    if ckpt and Path(ckpt).exists():
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        print(f"loaded checkpoint {ckpt}")
    model.eval()

    # ---- per-instance AUC + diagnostics ----
    probs, diag = _score(model, data, [(u, i, t, y) for (u, i, t, y, s) in rows], cfg, device)
    labels = np.array([y for (_, _, _, y, _) in rows])
    stages = np.array([s for (_, _, _, _, s) in rows])

    def auc(mask):
        yy, pp = labels[mask], probs[mask]
        return roc_auc_score(yy, pp) if len(np.unique(yy)) > 1 else float("nan")

    report = {"overall": {}, "per_stage": {}, "diagnostics": {}}
    report["overall"]["AUC"] = auc(np.ones(len(labels), bool))
    all_items = data.meta["video_id"].to_numpy()
    for st in STAGE_ORDER:
        m = stages == st
        if m.sum() == 0:
            continue
        a = auc(m)
        pos_rows = [r for r in rows if r[4] == st and r[3] == 1]
        hr, nd = ranking_metrics(model, data, pos_rows, cfg, device, all_items,
                                 max_pos=300 if smoke else 2000)
        report["per_stage"][st] = {"AUC": a, "HR@10": hr, "NDCG@10": nd, "n": int(m.sum())}
        report["diagnostics"][st] = {k: float(np.nanmean(diag[k][m])) for k in diag}

    def avg(metric, keys):
        vals = [report["per_stage"][k][metric] for k in keys if k in report["per_stage"]]
        return float(np.nanmean(vals)) if vals else float("nan")
    for metric in ("AUC", "HR@10", "NDCG@10"):
        report["overall"][f"Early_{metric}"] = avg(metric, ["G1", "G2"])
        report["overall"][f"Mature_{metric}"] = avg(metric, ["G3", "G4"])

    out = proc / "eval_report.json"
    out.write_text(json.dumps(report, indent=2))
    _print(report)
    print(f"saved -> {out}")
    return report


def _print(r):
    print("\n==== PERL Evaluation ====")
    print(f"Overall AUC: {r['overall']['AUC']:.4f}")
    print(f"{'stage':<6}{'AUC':>8}{'HR@10':>9}{'NDCG@10':>9}{'n':>8}")
    for st in STAGE_ORDER:
        if st in r["per_stage"]:
            s = r["per_stage"][st]
            print(f"{st:<6}{s['AUC']:>8.4f}{s['HR@10']:>9.4f}{s['NDCG@10']:>9.4f}{s['n']:>8}")
    print(f"Early  AUC {r['overall']['Early_AUC']:.4f} | Mature AUC {r['overall']['Mature_AUC']:.4f}")
    print("per-stage diagnostics (sigma^P/A/C, lambda^A/C):")
    for st in STAGE_ORDER:
        if st in r["diagnostics"]:
            d = r["diagnostics"][st]
            print(f"  {st}: sP={d['sigma_P']:.3f} sA={d['sigma_A']:.3f} sC={d['sigma_C']:.3f} "
                  f"lA={d['lambda_A']:.3f} lC={d['lambda_C']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ckpt = args.ckpt or str(Path(cfg.paths.checkpoint_dir) / "perl_best.pt")
    evaluate(cfg, ckpt=ckpt, smoke=args.smoke)


if __name__ == "__main__":
    main()
