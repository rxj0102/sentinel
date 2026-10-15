"""Sharpe statistics, deflated Sharpe, and multiple testing corrections.

Formulas follow Bailey and Lopez de Prado (2012, 2014). All Sharpe ratios inside the
probability formulas are per period (daily) values, not annualized.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

from sentinel.backtest.trials import TrialRegistry

_N = NormalDist()
EULER_GAMMA = 0.5772156649015329


def sharpe_per_period(r: pd.Series) -> float:
    x = r.dropna()
    sd = float(x.std(ddof=1))
    return float(x.mean() / sd) if sd > 0 else float("nan")


def annualize_sharpe(sr: float, periods: int = 252) -> float:
    return sr * math.sqrt(periods)


def moments(r: pd.Series) -> tuple[float, float]:
    """Skewness and raw (not excess) kurtosis."""
    x = r.dropna()
    return float(x.skew()), float(x.kurt()) + 3.0  # type: ignore[arg-type]


def probabilistic_sharpe(
    sr_hat: float, sr_star: float, n_obs: int, skew: float = 0.0, kurtosis: float = 3.0
) -> float:
    """Probability that the true Sharpe exceeds sr_star given the estimate and its moments."""
    denom = 1.0 - skew * sr_hat + (kurtosis - 1.0) / 4.0 * sr_hat**2
    if n_obs < 3 or denom <= 0:
        return float("nan")
    z = (sr_hat - sr_star) * math.sqrt(n_obs - 1) / math.sqrt(denom)
    return _N.cdf(z)


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """Expected maximum Sharpe among n_trials independent trials with zero true skill."""
    if n_trials <= 1 or var_sr <= 0:
        return 0.0
    g = EULER_GAMMA
    a = _N.inv_cdf(1.0 - 1.0 / n_trials)
    b = _N.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(var_sr) * ((1.0 - g) * a + g * b)


def deflated_sharpe(r: pd.Series, registry: TrialRegistry) -> dict[str, float]:
    """Deflated Sharpe probability using the registry trial count and Sharpe variance."""
    x = r.dropna()
    sr = sharpe_per_period(x)
    skew, kurt = moments(x)
    n_trials = registry.n_trials
    sr_star = expected_max_sharpe(n_trials, registry.sharpe_variance())
    return {
        "sharpe_annualized": annualize_sharpe(sr),
        "sr_star_annualized": annualize_sharpe(sr_star),
        "n_trials": float(n_trials),
        "n_evaluations": float(registry.n_evaluations),
        "dsr_probability": probabilistic_sharpe(sr, sr_star, len(x), skew, kurt),
        "psr_vs_zero": probabilistic_sharpe(sr, 0.0, len(x), skew, kurt),
    }


def sharpe_pvalue(r: pd.Series) -> float:
    """One sided p-value that the Sharpe is above zero (1 minus the PSR against zero)."""
    x = r.dropna()
    skew, kurt = moments(x)
    return 1.0 - probabilistic_sharpe(sharpe_per_period(x), 0.0, len(x), skew, kurt)


def holm_adjust(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (controls family wise error)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj.tolist()


def benjamini_hochberg(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values (controls false discovery rate)."""
    m = len(pvals)
    order = np.argsort(pvals)[::-1]
    adj = np.empty(m)
    running = 1.0
    for pos, idx in enumerate(order):
        rank = m - pos
        running = min(running, pvals[idx] * m / rank)
        adj[idx] = running
    return adj.tolist()


def max_drawdown(r: pd.Series) -> float:
    wealth = (1 + r.fillna(0)).cumprod()
    return float((wealth / wealth.cummax() - 1).min())
