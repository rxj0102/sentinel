"""Information coefficient analysis and decay."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sentinel.signals.base import forward_returns


def rank_ic(signal: pd.DataFrame, fwd: pd.DataFrame, min_obs: int = 10) -> pd.Series:
    """Spearman correlation across tickers at each date."""
    sr = signal.rank(axis=1)
    fr = fwd.where(signal.notna()).rank(axis=1)
    sr = sr.where(fr.notna())
    n = sr.notna().sum(axis=1)
    sm = sr.sub(sr.mean(axis=1), axis=0)
    fm = fr.sub(fr.mean(axis=1), axis=0)
    num = (sm * fm).sum(axis=1)
    den = ((sm**2).sum(axis=1) * (fm**2).sum(axis=1)).pow(0.5)
    ic = num / den.replace(0, np.nan)
    return ic.where(n >= min_obs).dropna()


def newey_west_tstat(x: pd.Series, lags: int) -> float:
    """t-statistic of the mean with a Bartlett kernel long run variance."""
    v = x.dropna().to_numpy(dtype=float)
    n = len(v)
    if n < 3:
        return float("nan")
    e = v - v.mean()
    lrv = float(e @ e) / n
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1 - lag / (lags + 1)
        lrv += 2 * w * float(e[lag:] @ e[:-lag]) / n
    se = np.sqrt(lrv / n)
    return float(v.mean() / se) if se > 0 else float("nan")


def ic_summary(ic: pd.Series, horizon: int, expected_sign: int) -> dict[str, float]:
    """Mean IC, IC information ratio, overlap robust t-stat, hit rate against expected sign."""
    lags = max(horizon - 1, 0)
    return {
        "mean_ic": float(ic.mean()),
        "ic_std": float(ic.std(ddof=1)),
        "icir": float(ic.mean() / ic.std(ddof=1)) if ic.std(ddof=1) > 0 else float("nan"),
        "t_stat_nw": newey_west_tstat(ic, lags),
        "hit_rate_expected_sign": float((np.sign(ic) == expected_sign).mean()),
        "n_dates": float(len(ic)),
        "expected_sign": float(expected_sign),
    }


def decay_table(
    signal: pd.DataFrame,
    close: pd.DataFrame,
    expected_sign: int,
    horizons: tuple[int, ...] = (1, 5, 10, 21, 63),
    delay: int = 1,
) -> pd.DataFrame:
    """IC summary per forward horizon. Rows are non-overlapping only in the t-stat sense,
    the Newey West lag handles the overlap."""
    rows = []
    for h in horizons:
        ic = rank_ic(signal, forward_returns(close, h, delay))
        rows.append({"horizon": h, **ic_summary(ic, h, expected_sign)})
    return pd.DataFrame(rows)
