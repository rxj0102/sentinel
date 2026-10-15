"""A delisted name with a missing return must not look like a flat position."""

import numpy as np
import pandas as pd
import pytest

from sentinel.backtest.engine import run_backtest, summarize, terminal_positions
from sentinel.backtest.trials import TrialSpec
from sentinel.config import BacktestConfig
from sentinel.signals.base import Panel

from .conftest import START, zero_cost_cfg

N, LAST_A = 80, 50  # A trades through row 50 and then vanishes


def _panel(gap: tuple[int, int] | None = None, vanish: bool = True) -> Panel:
    idx = pd.bdate_range(START, periods=N)
    ret = pd.DataFrame(0.0, index=idx, columns=["A", "B"])
    close = 100 * (1 + ret).cumprod()
    if vanish:
        close.loc[idx[LAST_A + 1] :, "A"] = np.nan
    if gap:
        close.iloc[gap[0] : gap[1], 0] = np.nan
    vol = pd.DataFrame(2_000_000.0, index=idx, columns=["A", "B"])
    return Panel(close=close, high=close, low=close, volume=vol)


def _run(guard, registry, side: float, treatment: str, panel: Panel | None = None, name="t"):  # type: ignore[no-untyped-def]
    p = panel or _panel()
    w = pd.DataFrame(0.0, index=p.close.index, columns=["A", "B"])
    w["A"] = side * 0.4
    w["B"] = -side * 0.4
    cfg = zero_cost_cfg(
        missing_return_treatment=treatment, delisting_return=-0.30, rebalance_every=1
    )
    return run_backtest(
        w,
        p,
        cfg,
        guard=guard,
        registry=registry,
        trial=TrialSpec("parameter_set", f"{name}-{side}-{treatment}", {}),
    )


def test_terminal_positions() -> None:
    close = _panel(gap=(20, 22)).close
    term = terminal_positions(close)
    assert term[0] == LAST_A + 1  # A vanishes, the earlier gap does not count
    assert term[1] > N  # B is priced on the last day


def test_default_is_conservative_and_tied_to_milder_scenario() -> None:
    cfg = BacktestConfig()
    assert cfg.missing_return_treatment == "delisting_adverse"
    assert cfg.delisting_return == pytest.approx(-0.30)
    import inspect

    from sentinel.data.quality import MILDER, survivorship_sensitivity  # noqa: F401

    default_delist = (
        inspect.signature(survivorship_sensitivity).parameters["delist_returns"].default
    )
    assert max(default_delist) == cfg.delisting_return  # the milder survivorship scenario


def test_long_in_delisted_name_takes_the_loss_not_a_flat_return(guard, registry) -> None:  # type: ignore[no-untyped-def]
    treated = _run(guard, registry, +1.0, "delisting_adverse")
    flat = _run(guard, registry, +1.0, "zero")
    assert flat.gross.sum() == pytest.approx(0.0)  # zero treatment looks flat: the flaw
    assert treated.delisting_events == 1
    assert treated.delisting_pnl == pytest.approx(-0.4 * 0.30)
    assert treated.gross.sum() == pytest.approx(-0.4 * 0.30)
    assert treated.gross.sum() < flat.gross.sum()
    assert summarize(treated)["delisting_events"] == 1.0


def test_short_in_delisted_name_is_also_adverse_by_default(guard, registry) -> None:  # type: ignore[no-untyped-def]
    adverse = _run(guard, registry, -1.0, "delisting_adverse")
    stock = _run(guard, registry, -1.0, "delisting_stock_return")
    assert adverse.gross.sum() == pytest.approx(-0.4 * 0.30)  # loss despite being short
    assert stock.gross.sum() == pytest.approx(+0.4 * 0.30)  # a short gains if the stock falls


def test_position_is_closed_after_delisting_and_not_earning_again(guard, registry) -> None:  # type: ignore[no-untyped-def]
    res = _run(guard, registry, +1.0, "delisting_adverse")
    after = res.gross.index > pd.bdate_range(START, periods=N)[LAST_A + 1]
    assert (res.gross[after] == 0).all()
    assert res.gross_exposure.iloc[-1] == pytest.approx(0.4)  # only B remains


def test_gap_that_resumes_is_flat_and_counted_not_a_delisting(guard, registry) -> None:  # type: ignore[no-untyped-def]
    p = _panel(gap=(20, 23), vanish=False)
    res = _run(guard, registry, +1.0, "delisting_adverse", panel=p, name="gap")
    assert res.delisting_events == 0 and res.gross.sum() == pytest.approx(0.0)
    assert res.gap_days_held >= 3


def test_untouched_when_position_not_held(guard, registry) -> None:  # type: ignore[no-untyped-def]
    p = _panel()
    w = pd.DataFrame({"A": 0.0, "B": 0.4}, index=p.close.index)
    res = run_backtest(
        w,
        p,
        zero_cost_cfg(),
        guard=guard,
        registry=registry,
        trial=TrialSpec("parameter_set", "notheld", {}),
    )
    assert res.delisting_events == 0 and res.delisting_pnl == 0.0
