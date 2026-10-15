"""Quantile long-short construction, turnover and capacity for signal research."""

from __future__ import annotations

import numpy as np
import pandas as pd


def quantile_weights(signal: pd.DataFrame, quantile: float = 0.2, sign: int = 1) -> pd.DataFrame:
    """Dollar neutral equal weight long top and short bottom quantile of sign * signal.

    Long leg sums to +1, short leg to -1. Dates without enough names get zero weights.
    """
    s = signal * sign
    rank = s.rank(axis=1, pct=True, method="first")
    n = s.notna().sum(axis=1)
    long = (rank > 1 - quantile).astype(float).where(s.notna(), 0.0)
    short = (rank <= quantile).astype(float).where(s.notna(), 0.0)
    ln, sn = long.sum(axis=1).replace(0, np.nan), short.sum(axis=1).replace(0, np.nan)
    w = long.div(ln, axis=0) - short.div(sn, axis=0)
    return w.where(n >= 10, 0.0).fillna(0.0)


def turnover(weights: pd.DataFrame) -> pd.Series:
    """One way turnover per period: half the sum of absolute weight changes."""
    return 0.5 * weights.diff().abs().sum(axis=1).iloc[1:]


def quantile_weights_long_only(
    signal: pd.DataFrame, quantile: float = 0.2, sign: int = 1
) -> pd.DataFrame:
    """Equal weight long only top quantile of sign * signal, summing to one."""
    w = quantile_weights(signal, quantile, sign)
    return w.clip(lower=0.0)


def gross_ls_return(weights: pd.DataFrame, ret: pd.DataFrame, delay: int = 1) -> pd.Series:
    """Daily return before costs under the documented timing rule.

    Weights decided at close t are traded at close t + delay and first earn the return of
    day t + delay + 1. So the holding shift is delay + 1. (An earlier version shifted by
    delay only, which let a trade at the signal close earn the next return. Fixed in Phase 3
    and not used by any Phase 2 output.) Missing returns are treated as zero (position is
    flat), a simplification.
    """
    held = weights.shift(delay + 1)
    return (held * ret.fillna(0.0)).sum(axis=1).iloc[delay + 1 :]


def capacity(
    weights: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    participation: float = 0.05,
    pct: float = 0.05,
) -> float:
    """AUM at which the pct-th percentile of per-name participation hits the cap.

    For each date and name, AUM_max = participation * ADV / |w|. The result is the median
    over dates of the pct percentile across names, a robust version of the binding constraint.
    ADV is the trailing 21 day mean of dollar volume, lagged one day.
    """
    adv = dollar_volume.rolling(21, min_periods=10).mean().shift(1)
    aw = weights.abs().where(weights.abs() > 0)
    cap = (participation * adv) / aw
    per_date = cap.quantile(pct, axis=1).dropna()
    return float(per_date.median()) if len(per_date) else float("nan")
