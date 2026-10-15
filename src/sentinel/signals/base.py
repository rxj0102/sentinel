"""Shared plumbing for signals: the wide panel, forward returns, cross-sectional transforms.

Timing convention (the single anti look-ahead rule for signals):
a signal at row t may use data from rows <= t. The position is opened at the close of
t + delay (delay defaults to 1) and the forward return is measured from there.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Panel:
    """Wide frames, index = trading dates, columns = tickers."""

    close: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    volume: pd.DataFrame

    @property
    def ret(self) -> pd.DataFrame:
        return self.close.pct_change(fill_method=None)

    @property
    def dollar_volume(self) -> pd.DataFrame:
        return self.close * self.volume


def panel_from_prices(prices: pd.DataFrame, calendar: pd.DatetimeIndex | None = None) -> Panel:
    def wide(col: str) -> pd.DataFrame:
        w = prices.pivot(index="date", columns="ticker", values=col)
        return w if calendar is None else w.reindex(calendar)

    return Panel(close=wide("adj_close"), high=wide("high"), low=wide("low"), volume=wide("volume"))


def forward_returns(close: pd.DataFrame, horizon: int, delay: int = 1) -> pd.DataFrame:
    """Return from close[t+delay] to close[t+delay+horizon], stored at row t."""
    if horizon < 1 or delay < 0:
        raise ValueError("horizon must be at least 1 and delay non-negative")
    entry = close.shift(-delay)
    exit_ = close.shift(-(delay + horizon))
    return exit_ / entry - 1.0


def winsorize(x: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    lo = x.quantile(lower, axis=1)
    hi = x.quantile(upper, axis=1)
    return x.clip(lower=lo, upper=hi, axis=0)


def zscore(x: pd.DataFrame) -> pd.DataFrame:
    mu = x.mean(axis=1)
    sd = x.std(axis=1, ddof=0).replace(0, np.nan)
    return x.sub(mu, axis=0).div(sd, axis=0)


@dataclass(frozen=True)
class SignalSpec:
    """Metadata that drives model cards and pre-registered tests."""

    name: str
    expected_sign: int  # sign of the relation between signal value and next period return
    hypothesis: str
    construction: str
    reference: str
    min_history: int
    data: str = "prices and volume (yfinance)"
    caveats: tuple[str, ...] = field(default_factory=tuple)
