"""Pre-registered replacement for the assumed information coefficient (Amendment 2).

Expanding window, point in time: at rebalance row t only dates s whose forward label ends on
or before t are used. Dates are spaced ic_horizon rows apart so labels do not overlap. The
estimate is the mean per-date Spearman rank IC of the composite score against the forward
return. Development window data only, the caller passes holdout-clipped frames.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sentinel.signals.base import forward_returns
from sentinel.signals.ic import newey_west_tstat, rank_ic


@dataclass(frozen=True)
class IcEstimate:
    estimate: float
    std_error: float
    n_dates: int


def per_date_ic(score: pd.DataFrame, close: pd.DataFrame, horizon: int, delay: int) -> pd.Series:
    """Rank IC at every row where it is defined, indexed by row position."""
    ic = rank_ic(score, forward_returns(close, horizon, delay))
    pos = score.index.get_indexer(ic.index)
    return pd.Series(ic.to_numpy(), index=pos)


def estimate_at(ic_by_pos: pd.Series, t: int, horizon: int, delay: int) -> IcEstimate:
    """Estimate usable at row t. Only non-overlapping dates with labels ending by t."""
    usable = ic_by_pos[(ic_by_pos.index % horizon == 0) & (ic_by_pos.index + delay + horizon <= t)]
    n = len(usable)
    if n == 0:
        return IcEstimate(float("nan"), float("nan"), 0)
    se = float("nan")
    if n >= 3:
        t_stat = newey_west_tstat(usable, 0)
        mean = float(usable.mean())
        se = abs(mean / t_stat) if t_stat and np.isfinite(t_stat) else float("nan")
    return IcEstimate(float(usable.mean()), se, n)


def resolve_ic(est: IcEstimate, assumed: float, min_dates: int) -> tuple[float, str]:
    """Value to use and why: estimate, fallback to the assumption, or floored at zero."""
    if est.n_dates < min_dates or not np.isfinite(est.estimate):
        return assumed, "fallback_assumed"
    if est.estimate < 0:
        return 0.0, "floored_zero"
    return est.estimate, "estimate"
