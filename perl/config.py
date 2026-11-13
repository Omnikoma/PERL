"""Configuration loading for PERL.

Reads a YAML config (see configs/kuairec.yaml) into a nested namespace and resolves
paths relative to the project root. Also centralizes seeding and the vendored-TimesFM
import path so every entry point (train/evaluate/tests) behaves identically.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Cfg(dict):
    """dict with attribute access (recursively) for ergonomic cfg.model.embed_dim usage."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Cfg(v) if isinstance(v, dict) else v

    def __setattr__(self, k, v):
        self[k] = v


def load_config(path: str | Path) -> Cfg:
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)
    cfg = Cfg(raw)
    # resolve paths relative to project root
    p = cfg["paths"]
    for k in ("processed_dir", "checkpoint_dir", "timesfm_src", "timesfm_model_dir"):
        if k in p:
            p[k] = str((PROJECT_ROOT / p[k]).resolve())
    # make the vendored TimesFM package importable (takes precedence over any global install)
    import sys
    if "timesfm_src" in p and p["timesfm_src"] not in sys.path:
        sys.path.insert(0, p["timesfm_src"])
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    return cfg


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
