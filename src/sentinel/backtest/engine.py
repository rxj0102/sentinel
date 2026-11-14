"""Vectorized backtest engine with costs, borrow, limits, holdout guard and trial counting.

Timing rule: weights decided at close t are traded at close t + delay and first earn the
return of day t + delay + 1. Costs are booked on the trade date.

The engine refuses to run without a TrialSpec and a TrialRegistry, so no evaluation is
silent, and it asks the HoldoutGuard before it touches any date.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sentinel.backtest import costs as cm
from sentinel.backtest.holdout import HoldoutGrant, HoldoutGuard
from sentinel.backtest.stats import (
    annualize_sharpe,
    max_drawdown,
    moments,
    sharpe_per_period,
)
from sentinel.backtest.trials import COUNTED_KINDS, TrialRegistry, TrialSpec
from sentinel.config import BacktestConfig
from sentinel.data.storage import frame_hash
from sentinel.signals.base import Panel


def rebalance(weights: pd.DataFrame, every: int) -> pd.DataFrame:
    """Keep the target weights on every k-th row and hold them in between."""
    mask = np.arange(len(weights)) % every == 0
    return weights.where(pd.Series(mask, index=weights.index), np.nan).ffill()


def terminal_positions(close: pd.DataFrame) -> np.ndarray:
    """Position of the first missing day after each ticker last valid price.

    Returns an integer per column, equal to len(close) + 10 when the ticker is still priced on
    the last day (it never vanishes inside the sample) or never has a price. A ticker with a
    price gap that later resumes is not terminal, only a series that ends is.
    """
    valid = close.notna().to_numpy()
    n = valid.shape[0]
    has = valid.any(axis=0)
    last = np.where(has, n - 1 - np.argmax(valid[::-1], axis=0), -1)
    return np.where(has & (last < n - 1), last + 1, n + 10)


@dataclass(frozen=True)
class BacktestResult:
    gross: pd.Series
    net: pd.Series
    costs: pd.DataFrame  # commission, spread, slippage, impact, borrow
    turnover: pd.Series  # one way, fraction of NAV per day
    gross_exposure: pd.Series
    participation_breaches: int
    nonzero_trades: int
    missing_cost_inputs: int
    borrow_rate: float
    cost_multiplier: float
    trial_hash: str
    missing_treatment: str = "delisting_adverse"
    delisting_events: int = 0  # held positions closed because the name vanished
    delisting_pnl: float = 0.0  # P&L booked from those events, fraction of NAV
    gap_days_held: int = 0  # held position days with a missing return that later resumed


def run_backtest(
    weights: pd.DataFrame,
    panel: Panel,
    cfg: BacktestConfig,
    *,
    guard: HoldoutGuard,
    registry: TrialRegistry,
    trial: TrialSpec,
    unlock: HoldoutGrant | None = None,
    cost_multiplier: float = 1.0,
    borrow_rate: float | None = None,
) -> BacktestResult:
    """Run one backtest. Every call is registered as a trial before any computation."""
    guard.authorize(weights.index.max(), unlock, caller="run_backtest")
    rate = cfg.borrow_rate_annual if borrow_rate is None else borrow_rate
    spec = TrialSpec(
        trial.kind,
        trial.name,
        {
            **trial.params,
            "cfg": cfg.model_dump(),
            "cost_multiplier": cost_multiplier,
            "borrow_rate": rate,
            "weights_hash": frame_hash(weights.reset_index(drop=True)),
            "close_hash": frame_hash(panel.close.reset_index(drop=True)),
        },
    )
    digest = registry.register(spec)

    ret = panel.ret.reindex(weights.index)
    target = rebalance(
        weights.reindex(columns=panel.close.columns).fillna(0.0), cfg.rebalance_every
    )
    limited = cm.apply_limits(target, cfg.max_position, cfg.max_gross)
    exec_w = limited.shift(cfg.delay).fillna(0.0)  # position held from the trade close
    n_rows = len(exec_w)
    term = terminal_positions(panel.close.reindex(weights.index).reindex(columns=exec_w.columns))
    gone = np.arange(n_rows)[:, None] >= term[None, :]
    exec_w = exec_w.mask(pd.DataFrame(gone, index=exec_w.index, columns=exec_w.columns), 0.0)
    held = exec_w.shift(1).fillna(0.0)
    pnl = (held * ret.fillna(0.0)).to_numpy().copy()
    d_events, d_pnl = 0, 0.0
    treatment = cfg.missing_return_treatment
    if treatment != "zero":
        held_np = held.to_numpy()
        for j in np.flatnonzero(term < n_rows):
            h = held_np[term[j], j]
            if h != 0.0:
                cell = (
                    -abs(h) * abs(cfg.delisting_return)
                    if treatment == "delisting_adverse"
                    else h * cfg.delisting_return
                )
                pnl[term[j], j] = cell
                d_events += 1
                d_pnl += cell
    nan_ret = ret.isna().to_numpy()
    not_terminal = np.arange(n_rows)[:, None] < term[None, :]
    gap_days = int(((held.to_numpy() != 0) & nan_ret & not_terminal).sum())
    gross = pd.Series(pnl.sum(axis=1), index=exec_w.index)

    sigma, adv = cm.cost_inputs(panel.ret, panel.dollar_volume)
    sigma, adv = sigma.reindex(weights.index), adv.reindex(weights.index)
    comp = cm.trading_costs(exec_w, sigma, adv, cfg, cost_multiplier)
    comp["borrow"] = cm.borrow_cost(exec_w, rate)
    net = gross - comp.sum(axis=1)

    trade = exec_w.diff().abs()
    missing = int(((trade > 0) & (sigma.isna() | adv.isna())).sum().sum())
    breaches, nonzero = cm.participation_breaches(exec_w, adv, cfg)
    start = exec_w.abs().sum(axis=1).gt(0).idxmax() if exec_w.abs().sum().sum() > 0 else None
    sl = slice(start, None)
    result = BacktestResult(
        gross=gross.loc[sl],
        net=net.loc[sl],
        costs=comp.loc[sl],
        turnover=(0.5 * trade.sum(axis=1)).loc[sl],
        gross_exposure=exec_w.abs().sum(axis=1).loc[sl],
        participation_breaches=breaches,
        nonzero_trades=nonzero,
        missing_cost_inputs=missing,
        borrow_rate=rate,
        cost_multiplier=cost_multiplier,
        trial_hash=digest,
        missing_treatment=treatment,
        delisting_events=d_events,
        delisting_pnl=d_pnl,
        gap_days_held=gap_days,
    )
    if trial.kind in COUNTED_KINDS or trial.kind == "sensitivity":
        skew, kurt = moments(result.net)
        registry.record_result(
            digest, sharpe_per_period(result.net), len(result.net.dropna()), skew, kurt
        )
    return result


def summarize(res: BacktestResult, periods: int = 252) -> dict[str, float]:
    n = res.net.dropna()
    return {
        "ann_return_net": (
            float(np.prod(1 + n.to_numpy()) ** (periods / len(n)) - 1) if len(n) else np.nan
        ),
        "ann_vol_net": float(n.std(ddof=1) * np.sqrt(periods)),
        "sharpe_gross": annualize_sharpe(sharpe_per_period(res.gross.loc[n.index]), periods),
        "sharpe_net": annualize_sharpe(sharpe_per_period(n), periods),
        "max_drawdown_net": max_drawdown(n),
        "mean_one_way_turnover": float(res.turnover.mean()),
        "annual_cost_drag": float(res.costs.sum(axis=1).mean() * periods),
        "annual_borrow_cost": float(res.costs["borrow"].mean() * periods),
        "annual_impact_cost": float(res.costs["impact"].mean() * periods),
        "participation_breach_share": (
            res.participation_breaches / res.nonzero_trades if res.nonzero_trades else np.nan
        ),
        "delisting_events": float(res.delisting_events),
        "delisting_pnl": res.delisting_pnl,
        "gap_days_held": float(res.gap_days_held),
        "borrow_rate_assumed": res.borrow_rate,
        "cost_multiplier": res.cost_multiplier,
        "n_days": float(len(n)),
    }


def sensitivity_grid(
    weights: pd.DataFrame,
    panel: Panel,
    cfg: BacktestConfig,
    *,
    guard: HoldoutGuard,
    registry: TrialRegistry,
    name: str,
    unlock: HoldoutGrant | None = None,
) -> pd.DataFrame:
    """Net Sharpe and cost drag across the borrow and cost multiplier grids.

    Every cell is registered as a sensitivity trial. Cells reprice the same weights, so they
    are reported separately and do not enter the deflated Sharpe trial count.
    """
    rows = []
    for borrow in cfg.borrow_grid:
        for mult in cfg.cost_multipliers:
            res = run_backtest(
                weights,
                panel,
                cfg,
                guard=guard,
                registry=registry,
                trial=TrialSpec("sensitivity", f"{name}|borrow={borrow}|mult={mult}", {}),
                unlock=unlock,
                cost_multiplier=mult,
                borrow_rate=borrow,
            )
            s = summarize(res)
            rows.append(
                {
                    "signal": name,
                    "borrow_rate_assumed": borrow,
                    "cost_multiplier": mult,
                    "sharpe_net": s["sharpe_net"],
                    "ann_return_net": s["ann_return_net"],
                    "annual_cost_drag": s["annual_cost_drag"],
                }
            )
    return pd.DataFrame(rows)
