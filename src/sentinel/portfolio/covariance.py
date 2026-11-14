"""Ledoit-Wolf shrinkage covariance. No tuning parameter: the intensity is analytic."""

from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf


def ledoit_wolf(returns: np.ndarray) -> tuple[np.ndarray, float]:
    """Shrink the sample covariance toward a scaled identity with the Ledoit-Wolf intensity.

    returns is (observations x assets) with no missing values. The result is positive
    definite even when assets outnumber observations, and preserves the sample trace.
    """
    if returns.ndim != 2 or returns.shape[0] < 3 or returns.shape[1] < 1:
        raise ValueError("returns must be a 2d array with at least 3 observations")
    if not np.isfinite(returns).all():
        raise ValueError("returns contain missing or infinite values")
    est = LedoitWolf().fit(returns)
    return np.asarray(est.covariance_, dtype=float), float(est.shrinkage_)


def risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Share of portfolio variance contributed by each asset, summing to one (long or short)."""
    marginal = cov @ w
    total = float(w @ marginal)
    return np.asarray(w * marginal / total) if total > 0 else np.full(len(w), np.nan)
