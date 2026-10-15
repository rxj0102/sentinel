from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sentinel.backtest.holdout import HoldoutGuard
from sentinel.backtest.trials import TrialRegistry
from sentinel.config import BacktestConfig
from sentinel.signals.base import Panel

START = pd.Timestamp("2020-01-01")


def make_panel(n: int = 120, names: tuple[str, ...] = ("A", "B"), seed: int = 0) -> Panel:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(START, periods=n)
    ret = pd.DataFrame(rng.normal(0.0005, 0.01, (n, len(names))), index=idx, columns=list(names))
    close = 100 * (1 + ret).cumprod()
    vol = pd.DataFrame(2_000_000.0, index=idx, columns=list(names))
    return Panel(close=close, high=close * 1.01, low=close * 0.99, volume=vol)


def zero_cost_cfg(**kw: float) -> BacktestConfig:
    base = {
        "commission_bps": 0.0,
        "half_spread_bps": 0.0,
        "slippage_bps": 0.0,
        "impact_coeff": 0.0,
        "borrow_rate_annual": 0.0,
        "rebalance_every": 1,
        "max_position": 1.0,
        "max_gross": 10.0,
    }
    return BacktestConfig(**{**base, **kw})


@pytest.fixture
def guard(tmp_path: Path) -> HoldoutGuard:
    far = START + pd.Timedelta(days=3650)
    return HoldoutGuard(far, far + pd.Timedelta(days=365), tmp_path / "access_log.csv")


@pytest.fixture
def registry() -> TrialRegistry:
    return TrialRegistry()
