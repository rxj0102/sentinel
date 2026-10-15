"""Behavioral signals. Each function maps a Panel to a wide signal frame.

Raw signals are NOT sign-flipped: expected_sign in the spec says which way the relation to
future returns should go, and IC analysis reports against it. Row t uses data through t only.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from sentinel.signals.base import Panel, SignalSpec

SignalFn = Callable[[Panel], pd.DataFrame]


def max_lottery(panel: Panel, window: int = 21, k: int = 5) -> pd.DataFrame:
    """Mean of the k largest daily returns in the window (Bali, Cakici, Whitelaw 2011)."""
    r = panel.ret

    def topk_mean(a: np.ndarray) -> float:
        a = a[~np.isnan(a)]
        return float(np.sort(a)[-k:].mean()) if len(a) >= k else np.nan

    return r.rolling(window, min_periods=window).apply(topk_mean, raw=True)


def high52(panel: Panel, window: int = 252) -> pd.DataFrame:
    """Close relative to the trailing high (George and Hwang 2004). 1.0 means at the high."""
    hi = panel.close.rolling(window, min_periods=window).max()
    return panel.close / hi


def reversal(panel: Panel, window: int = 21) -> pd.DataFrame:
    """Past window return (Jegadeesh 1990). Expected to predict the opposite sign."""
    return panel.close / panel.close.shift(window) - 1.0


def overhang(panel: Panel, vol_window: int = 252) -> pd.DataFrame:
    """Capital gains overhang proxy (Grinblatt and Han 2005): (P - RP) / P.

    RP is a volume weighted reference price. With no shares outstanding data, the turnover
    weight is approximated as (1 / vol_window) times volume over its trailing mean, capped
    at one. This proxy is rough and is documented as a limitation.
    """
    vbar = panel.volume.rolling(vol_window, min_periods=vol_window // 2).mean()
    w = (panel.volume / vbar / vol_window).clip(upper=1.0)
    p = panel.close.to_numpy()
    wv = w.to_numpy()
    rp = np.full_like(p, np.nan)
    for t in range(p.shape[0]):
        prev = rp[t - 1] if t else np.full(p.shape[1], np.nan)
        init = np.isnan(prev)
        cur = np.where(init, p[t], (1 - np.nan_to_num(wv[t])) * prev + np.nan_to_num(wv[t]) * p[t])
        rp[t] = np.where(np.isnan(p[t]), prev, cur)
    ref = pd.DataFrame(rp, index=panel.close.index, columns=panel.close.columns)
    ov = (panel.close - ref) / panel.close
    valid = vbar.notna()
    return ov.where(valid)


def realized_skew(panel: Panel, window: int = 63) -> pd.DataFrame:
    """Skewness of daily returns over the window (Amaya et al 2015)."""
    return panel.ret.rolling(window, min_periods=window).skew()


def abnormal_volume(panel: Panel, short: int = 5, long: int = 252) -> pd.DataFrame:
    """Log ratio of short to long average volume. Attention proxy (Barber and Odean 2008)."""
    s = panel.volume.rolling(short, min_periods=short).mean()
    lg = panel.volume.rolling(long, min_periods=long // 2).mean()
    ratio = s.where(s > 0) / lg.where(lg > 0)
    return pd.DataFrame(np.log(ratio), index=ratio.index, columns=ratio.columns)


REGISTRY: dict[str, tuple[SignalFn, SignalSpec]] = {
    "max_lottery": (
        max_lottery,
        SignalSpec(
            "max_lottery",
            -1,
            "Investors overpay for lottery payoffs, so stocks with extreme recent up days are "
            "overpriced and earn lower future returns. Limits to arbitrage via short constraints.",
            "Mean of the five largest daily returns in the past 21 trading days.",
            "Bali, Cakici, Whitelaw (2011)",
            22,
            caveats=("Effect concentrates in small illiquid names, weaker in large caps.",),
        ),
    ),
    "high52": (
        high52,
        SignalSpec(
            "high52",
            1,
            "Investors anchor on the 52 week high and underreact to good news for stocks "
            "near it, so proximity to the high predicts higher returns.",
            "Adjusted close divided by the trailing 252 day maximum close.",
            "George and Hwang (2004)",
            253,
            caveats=("Replaces analyst herding, see deviations in README.",),
        ),
    ),
    "reversal": (
        reversal,
        SignalSpec(
            "reversal",
            -1,
            "Overreaction and liquidity provision: last month winners underperform next "
            "month, losers outperform.",
            "Return over the past 21 trading days.",
            "Jegadeesh (1990)",
            22,
            caveats=("Very high turnover, likely killed by costs. Kept as a cost honesty case.",),
        ),
    ),
    "overhang": (
        overhang,
        SignalSpec(
            "overhang",
            1,
            "Disposition effect: unrealized gains create selling pressure that depresses price, "
            "which later drifts up toward fundamentals, so high overhang predicts higher returns.",
            "Price minus a volume weighted reference price, divided by price.",
            "Grinblatt and Han (2005)",
            253,
            caveats=("Turnover weights are approximated without shares outstanding.",),
            confidence="lower",
            confidence_note=(
                "Free data has no shares outstanding history, so the turnover weight in the "
                "reference price is a volume ratio proxy. The proxy can misstate the reference "
                "price, so any result for this signal carries lower confidence than the others."
            ),
        ),
    ),
    "realized_skew": (
        realized_skew,
        SignalSpec(
            "realized_skew",
            -1,
            "Lottery preference through skewness: positively skewed stocks are overpriced.",
            "Skewness of daily returns over 63 trading days.",
            "Amaya, Christoffersen, Jacobs, Vasquez (2015)",
            64,
            caveats=("Backtestable substitute for option implied skew, which is forward only.",),
        ),
    ),
    "abnormal_volume": (
        abnormal_volume,
        SignalSpec(
            "abnormal_volume",
            1,
            "Attention driven retail buying pushes prices up over the next few weeks, then "
            "reverses at longer horizons. Signs are locked per horizon in docs/hypotheses.md.",
            "Log of the 5 day over 252 day average volume.",
            "Barber and Odean (2008)",
            253,
            caveats=("Google Trends version is best effort and not in the core pipeline.",),
            signs_by_horizon={1: 1, 5: 1, 10: 1, 21: 1, 63: -1},
        ),
    ),
}


def compute_all(panel: Panel) -> dict[str, pd.DataFrame]:
    return {name: fn(panel) for name, (fn, _) in REGISTRY.items()}
