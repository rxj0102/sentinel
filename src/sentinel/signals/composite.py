"""Pre-registered composite score: equal weight average of signed, standardized signals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sentinel.config import PortfolioConfig
from sentinel.data.universe import mask_to_members
from sentinel.signals.base import Panel, winsorize, zscore
from sentinel.signals.library import REGISTRY

PRIMARY_HORIZON = 21


def composite_score(panel: Panel, membership: pd.DataFrame, cfg: PortfolioConfig) -> pd.DataFrame:
    """Winsorize, z-score and sign every signal, then average the available ones.

    The sign is the locked 21 day sign, so a higher composite always means a higher expected
    return. At least cfg.composite_min_signals must be available, else the score is NaN.
    The overhang signal is included at full weight although marked lower confidence.
    """
    parts = []
    for _, (fn, spec) in REGISTRY.items():
        s = mask_to_members(fn(panel), membership)
        z = zscore(winsorize(s, cfg.winsorize_lower, cfg.winsorize_upper))
        parts.append(z * spec.sign_at(PRIMARY_HORIZON))
    stack = np.stack([p.to_numpy() for p in parts])
    count = np.isfinite(stack).sum(axis=0)
    total = np.where(np.isfinite(stack), stack, 0.0).sum(axis=0)
    mean = total / np.maximum(count, 1)
    out = np.where(count >= cfg.composite_min_signals, mean, np.nan)
    return pd.DataFrame(out, index=panel.close.index, columns=panel.close.columns)
