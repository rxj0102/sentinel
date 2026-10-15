"""Trading cost, slippage, impact, borrow and position limit mechanics.

All parameters come from BacktestConfig. Defaults are assumptions, not measurements.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sentinel.config import BacktestConfig


def apply_limits(w: pd.DataFrame, max_position: float, max_gross: float) -> pd.DataFrame:
    """Cap each name at max_position (absolute), then scale down if gross exceeds max_gross."""
    capped = w.clip(lower=-max_position, upper=max_position)
    gross = capped.abs().sum(axis=1)
    scale = (max_gross / gross).where(gross > max_gross, 1.0).fillna(1.0)
    return capped.mul(scale, axis=0)


def cost_inputs(
    ret: pd.DataFrame, dollar_volume: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trailing 21 day volatility and average dollar volume, both lagged one day."""
    sigma = ret.rolling(21, min_periods=10).std().shift(1)
    adv = dollar_volume.rolling(21, min_periods=10).mean().shift(1)
    return sigma, adv


def trading_costs(
    exec_w: pd.DataFrame,
    sigma: pd.DataFrame,
    adv: pd.DataFrame,
    cfg: BacktestConfig,
    cost_multiplier: float = 1.0,
) -> pd.DataFrame:
    """Cost per day as a fraction of NAV, split by component. Booked on the trade date.

    commission, spread and slippage are linear in traded weight. Impact follows a square
    root model: bps = 1e4 * impact_coeff * sigma * sqrt(trade dollars / ADV). Names without
    enough history for sigma or ADV get zero impact and are counted by the engine.
    """
    trade = exec_w.diff().abs()
    trade.iloc[0] = exec_w.iloc[0].abs()
    m = cost_multiplier / 1e4
    out = pd.DataFrame(index=exec_w.index)
    out["commission"] = trade.sum(axis=1) * cfg.commission_bps * m
    out["spread"] = trade.sum(axis=1) * cfg.half_spread_bps * m
    out["slippage"] = trade.sum(axis=1) * cfg.slippage_bps * m
    participation = (trade * cfg.aum) / adv.where(adv > 0)
    impact_frac = cfg.impact_coeff * sigma * np.sqrt(participation)
    out["impact"] = (trade * impact_frac).sum(axis=1) * cost_multiplier
    return out


def borrow_cost(exec_w: pd.DataFrame, annual_rate: float) -> pd.Series:
    """Daily borrow cost on the short book as a fraction of NAV. The rate is an assumption."""
    short_value = (-exec_w.clip(upper=0.0)).sum(axis=1)
    return short_value * annual_rate / 252.0


def participation_breaches(
    exec_w: pd.DataFrame, adv: pd.DataFrame, cfg: BacktestConfig
) -> tuple[int, int]:
    """Count of trades above the participation cap and total nonzero trades."""
    trade = exec_w.diff().abs() * cfg.aum
    part = trade / adv.where(adv > 0)
    nonzero = trade > 0
    return int((part > cfg.max_participation).sum().sum()), int(nonzero.sum().sum())
