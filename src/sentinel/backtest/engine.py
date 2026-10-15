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
    held = exec_w.shift(1).fillna(0.0)
    gross = (held * ret.fillna(0.0)).sum(axis=1)

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
