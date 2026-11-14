"""Run configuration. One YAML file fully determines a run."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    start: date
    dev_end: date
    holdout_start: date
    holdout_end: date | None = None  # written on the day the holdout is locked
    seed: int = 0
    raw_dir: Path = Path("data/raw")
    curated_dir: Path = Path("data/curated")
    pit_lag_days: int = 1  # calendar days between a price date and its availability
    max_stale_days: int = 5
    extreme_return: float = 0.5  # absolute daily return flagged as suspicious

    @model_validator(mode="after")
    def _check_windows(self) -> DataConfig:
        if not self.start < self.dev_end:
            raise ValueError("start must precede dev_end")
        if not self.dev_end < self.holdout_start:
            raise ValueError("holdout_start must follow dev_end")
        if self.holdout_end is not None and self.holdout_end <= self.holdout_start:
            raise ValueError("holdout_end must follow holdout_start")
        return self


class BacktestConfig(BaseModel):
    """Execution assumptions. Every default is an assumption, not a measurement.

    The borrow rate in particular has no free data source. It is a flat assumption and every
    result must be shown across the sensitivity grid below, never at a single value.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    delay: int = 1  # trade at the close this many days after the signal date
    rebalance_every: int = 21  # pre-registered monthly rebalance, trading days
    quantile: float = 0.2  # pre-registered
    aum: float = 10_000_000.0
    commission_bps: float = 0.5
    half_spread_bps: float = 2.0
    slippage_bps: float = 1.0
    impact_coeff: float = 0.1  # square root impact: coeff * daily sigma * sqrt(trade / ADV)
    max_position: float = 0.03  # per name weight cap, share of NAV
    max_gross: float = 2.0
    max_participation: float = 0.05  # diagnostic cap on trade size as share of ADV
    borrow_rate_annual: float = 0.01  # ASSUMPTION, flat, no free data exists
    borrow_rate_is_assumption: bool = True
    cost_multipliers: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    borrow_grid: tuple[float, ...] = (0.0, 0.005, 0.01, 0.02, 0.05)
    # Held positions in vanished names. Default is conservative and tied to the milder
    # survivorship scenario. ASSUMPTION, not an estimate. zero is for comparison only.
    missing_return_treatment: Literal["delisting_adverse", "delisting_stock_return", "zero"] = (
        "delisting_adverse"
    )
    delisting_return: float = -0.30

    @model_validator(mode="after")
    def _check(self) -> BacktestConfig:
        if self.delay < 0 or self.rebalance_every < 1:
            raise ValueError("delay must be non-negative and rebalance_every at least 1")
        if not 0 < self.quantile < 0.5:
            raise ValueError("quantile must be in (0, 0.5)")
        if self.max_position <= 0 or self.max_gross <= 0 or self.aum <= 0:
            raise ValueError("limits and aum must be positive")
        if not -1.0 < self.delisting_return < 0.0:
            raise ValueError("delisting_return must be between -1 and 0")
        if not self.borrow_rate_is_assumption:
            raise ValueError("borrow rate is an assumption and must stay labeled as one")
        return self


class PortfolioConfig(BaseModel):
    """Pre-registered portfolio construction settings (docs/hypotheses.md, Amendment 1).

    ic_assumed is an assumption, not an estimate. Nothing here is tuned on results.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    ic_assumed: float = 0.02
    cov_window: int = 252
    cov_min_obs: int = 252
    risk_aversion: float = 10.0
    leg_quantile: float = 0.2
    composite_min_signals: int = 4
    min_names: int = 10
    exposure_tol: float = 0.10
    turnover_cap_one_way: float = 0.5
    winsorize_lower: float = 0.01
    winsorize_upper: float = 0.99
    # IC replacement procedure, docs/hypotheses.md Amendment 2. Default keeps the assumption.
    ic_source: Literal["assumed", "expanding_estimate"] = "assumed"
    ic_horizon: int = 21
    ic_min_dates: int = 24
    # Real runs always enforce the feasibility check. This switch only affects synthetic runs.
    require_full_gross_feasibility: bool = True

    @model_validator(mode="after")
    def _check(self) -> PortfolioConfig:
        if self.ic_horizon < 1 or self.ic_min_dates < 2:
            raise ValueError("ic_horizon must be at least 1 and ic_min_dates at least 2")
        if not 0 < self.leg_quantile < 0.5:
            raise ValueError("leg_quantile must be in (0, 0.5)")
        if self.cov_min_obs > self.cov_window or self.cov_window < 20:
            raise ValueError("cov_min_obs must not exceed cov_window, window at least 20")
        if self.risk_aversion <= 0 or self.turnover_cap_one_way <= 0 or self.exposure_tol <= 0:
            raise ValueError("risk_aversion, turnover cap and exposure tolerance must be positive")
        if not 1 <= self.composite_min_signals <= 6 or self.min_names < 2:
            raise ValueError("composite_min_signals in 1..6 and min_names at least 2")
        return self


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    data: DataConfig
    backtest: BacktestConfig = BacktestConfig()
    portfolio: PortfolioConfig = PortfolioConfig()


def load_config(path: Path) -> DataConfig:
    raw = yaml.safe_load(path.read_text())
    raw.pop("backtest", None)
    raw.pop("portfolio", None)
    return DataConfig.model_validate(raw)


def load_run_config(path: Path) -> RunConfig:
    """Flat data keys plus an optional backtest section in one YAML file."""
    raw = yaml.safe_load(path.read_text())
    bt = raw.pop("backtest", {}) or {}
    pf = raw.pop("portfolio", {}) or {}
    return RunConfig(
        data=DataConfig.model_validate(raw),
        backtest=BacktestConfig(**bt),
        portfolio=PortfolioConfig(**pf),
    )
