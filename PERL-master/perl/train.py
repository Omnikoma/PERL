"""PERL training loop.

- Training instances come from instances_train.parquet (lifecycle-stratified cutoffs).
- For the uncertainty-calibration loss L_unc, each instance is paired on the fly with a
  LATER cutoff of the same item (more observed history), giving an ordered (t_a < t_b) pair.
- Optimizer Adam, lr 1e-3, gradient clipping, early stopping on validation CTR-AUC.
- The TimesFM backbone is frozen; only its LoRA adapters + the PERL modules train.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from .config import load_config, set_seed, get_device
from .data.dataset import PERLData
from .data.collate import build_batch
from .data.lifecycle_mask import ONE_DAY_NS
from .model import PERL
from . import losses


def _rows(df):
    return list(zip(df["user_id"].to_numpy(), df["video_id"].to_numpy(),
                    df["cutoff_ts"].astype("int64").to_numpy(), df["label"].to_numpy()))


def _later_cutoff(data, item, cutoff_ns, rng):
    """Sample a strictly-later cutoff (more observed intervals) for the same item, for L_unc."""
    L_i = data.L_i.get(int(item), 0)
    L_now = data.L_at(int(item), int(cutoff_ns))
    if L_now >= L_i - 1:
        return None
    lp = int(rng.integers(L_now + 1, L_i))
    return int(data.t0[int(item)] + lp * ONE_DAY_NS)


@torch.no_grad()
def eval_auc(model, data, rows, cfg, device, max_n=20000):
    model.eval()
    if len(rows) > max_n:
        idx = np.random.default_rng(0).choice(len(rows), max_n, replace=False)
        rows = [rows[i] for i in idx]
    scores, labels = [], []
    bs = cfg.optim.batch_size
    for i in range(0, len(rows), bs):
        b = build_batch(data, rows[i:i + bs], cfg, device)
        out = model(b)
        scores.append(torch.sigmoid(out["logits"]).float().cpu().numpy())
        labels.append(b["label"].cpu().numpy())
    s, y = np.concatenate(scores), np.concatenate(labels)
    return roc_auc_score(y, s) if len(np.unique(y)) > 1 else 0.5


def train(cfg, smoke=False):
    set_seed(cfg.seed)
    device = get_device()
    proc = Path(cfg.paths.processed_dir)
    data = PERLData(proc, interval_days=1, meta_num_cat=31)
    tr = _rows(pd.read_parquet(proc / "instances_train.parquet"))
    va = _rows(pd.read_parquet(proc / "instances_valid.parquet"))
    print(f"train rows {len(tr):,} | valid rows {len(va):,} | device {device}")

    model = PERL(cfg, data).to(device)
    ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {ntr:,}")
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.optim.lr)

    rng = np.random.default_rng(cfg.seed)
    bs = cfg.optim.batch_size
    max_epochs = 1 if smoke else cfg.optim.max_epochs
    ckpt_dir = Path(cfg.paths.checkpoint_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_auc, patience = 0.0, 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        order = rng.permutation(len(tr))
        n_steps = 3 if smoke else len(order) // bs
        t0 = time.time()
        for step in range(n_steps):
            sel = order[step * bs:(step + 1) * bs]
            rows_a = [tr[i] for i in sel]
            b_a = build_batch(data, rows_a, cfg, device)
            out_a = model(b_a)
            labels = b_a["label"]

            # ---- ordered later-cutoff view for L_unc ----
            out_b = None
            if cfg.loss.lambda_unc > 0:
                rows_b = []
                for (u, it, ct, y) in rows_a:
                    cb = _later_cutoff(data, it, ct, rng)
                    rows_b.append((u, it, cb if cb is not None else ct, y))
                b_b = build_batch(data, rows_b, cfg, device)
                out_b = model(b_b)

            loss, parts = losses.total_loss(out_a, labels, cfg, model, out_b=out_b)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            opt.step()
            if step % 1 == 0 and smoke or step % 50 == 0:
                print(f"  epoch {epoch} step {step}/{n_steps} "
                      f"loss={parts['total']:.4f} ctr={parts['ctr']:.4f} "
                      f"cecl={parts['cecl']:.4f} unc={parts['unc']:.4f}")
            if smoke:
                _smoke_asserts(out_a, parts, model)
        if smoke:
            print("SMOKE OK: finite losses, LoRA grads present, m_A=0 => mu_A=0 verified.")
            return model
        auc = eval_auc(model, data, va, cfg, device)
        print(f"epoch {epoch} done in {time.time()-t0:.0f}s | val AUC {auc:.4f}")
        if auc > best_auc:
            best_auc, patience = auc, 0
            torch.save(model.state_dict(), ckpt_dir / "perl_best.pt")
            print(f"  ✓ saved best (val AUC {best_auc:.4f})")
        else:
            patience += 1
            if patience >= cfg.optim.early_stop_patience:
                print(f"early stop at epoch {epoch} (best val AUC {best_auc:.4f})")
                break
    return model


def _smoke_asserts(out, parts, model):
    assert np.isfinite(parts["total"]), "non-finite loss"
    # m_A=0 => mu_A exactly 0
    zero = (out["m_A"] == 0)
    if zero.any():
        assert out["mu_A"][zero].abs().sum().item() == 0.0, "mu_A must be 0 when m_A=0"
    # LoRA / trainable grads exist
    g = [p.grad for n, p in model.named_parameters() if p.requires_grad and p.grad is not None]
    assert len(g) > 0, "no gradients"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    train(cfg, smoke=args.smoke)


if __name__ == "__main__":
    main()
