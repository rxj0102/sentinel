"""Deterministic synthetic market used for CI and tests. Contains no real data.

It reproduces the structural problems of free data on purpose: index members that delist,
and delisted names whose price history is entirely absent (as on Yahoo).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sentinel.data.pit import with_lag
from sentinel.data.schemas import MembershipChangeSchema, PriceSchema


@dataclass(frozen=True)
class SyntheticMarket:
    prices: pd.DataFrame
    changes: pd.DataFrame
    initial_members: set[str]
    initial_date: pd.Timestamp
    delisted: dict[str, pd.Timestamp]  # ticker -> last trading day
    dropped: set[str]  # delisted tickers whose prices are entirely missing
    calendar: pd.DatetimeIndex


def make_market(
    n_initial: int = 30,
    n_reserve: int = 20,
    start: str = "2010-01-04",
    periods: int = 750,
    seed: int = 0,
    delist_frac: float = 0.3,
    dropped_frac: float = 0.6,
    pit_lag_days: int = 1,
) -> SyntheticMarket:
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range(start, periods=periods)
    n = n_initial + n_reserve
    tickers = [f"S{i:03d}" for i in range(n)]

    mkt = rng.normal(0.0003, 0.01, periods)
    beta = rng.uniform(0.6, 1.4, n)
    idio = rng.normal(0.0, 0.012, (periods, n))
    rets = mkt[:, None] * beta[None, :] + idio

    # Delistings: among initial members, some stop trading with a terminal crash.
    initial = tickers[:n_initial]
    n_del = int(round(delist_frac * n_initial))
    del_names = list(rng.choice(initial, size=n_del, replace=False))
    last_day: dict[str, pd.Timestamp] = {}
    for t in del_names:
        k = int(rng.integers(max(periods // 4, 1), max(periods - 20, 2)))
        i = tickers.index(t)
        rets[k, i] = -0.4
        last_day[t] = cal[k]
    dropped = set(del_names[: int(round(dropped_frac * n_del))])

    close = 50.0 * np.exp(np.cumsum(rets, axis=0))
    rows = []
    for i, t in enumerate(tickers):
        if t in dropped:
            continue
        end = int(cal.searchsorted(last_day[t])) + 1 if t in last_day else periods
        c = close[:end, i]
        spread = np.abs(rng.normal(0.004, 0.002, end))
        hi, lo = c * (1 + spread), c * (1 - spread)
        rows.append(
            pd.DataFrame(
                {
                    "date": cal[:end],
                    "ticker": t,
                    "open": c * (1 + rng.normal(0, 0.002, end)),
                    "high": hi,
                    "low": lo,
                    "close": c,
                    "adj_close": c,
                    "volume": rng.integers(1_000_000, 5_000_000, end).astype(float),
                }
            )
        )
    prices = with_lag(pd.concat(rows, ignore_index=True), "date", pit_lag_days)
    prices["open"] = prices["open"].clip(upper=prices["high"]).clip(lower=prices["low"])

    # Change log: every delister is removed the day after its last print and replaced
    # by a reserve name; plus a few ordinary swaps.
    reserve = iter(tickers[n_initial:])
    ev = []
    for t, d in sorted(last_day.items(), key=lambda kv: kv[1]):
        nxt = cal[int(cal.searchsorted(d)) + 1]
        ev.append({"date": nxt, "added": next(reserve), "removed": t})
    for _ in range(3 if periods > 120 else 0):
        k = int(rng.integers(50, periods - 50))
        out = str(rng.choice([t for t in initial if t not in last_day]))
        if any(e["removed"] == out for e in ev):
            continue
        ev.append({"date": cal[k], "added": next(reserve), "removed": out})
    changes = pd.DataFrame(ev).sort_values("date").reset_index(drop=True)
    MembershipChangeSchema.validate(changes)
    PriceSchema.validate(prices)
    return SyntheticMarket(
        prices=prices,
        changes=changes,
        initial_members=set(initial),
        initial_date=cal[0],
        delisted=last_day,
        dropped=dropped,
        calendar=cal,
    )
