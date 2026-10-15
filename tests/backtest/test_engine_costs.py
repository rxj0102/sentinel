import numpy as np
import pandas as pd
import pytest

from sentinel.backtest import costs as cm
from sentinel.backtest.engine import rebalance, run_backtest, sensitivity_grid, summarize
from sentinel.backtest.trials import TrialSpec
from sentinel.config import BacktestConfig, load_run_config
from sentinel.signals.base import Panel

from .conftest import START, make_panel, zero_cost_cfg


def test_apply_limits() -> None:
    w = pd.DataFrame({"a": [0.5, 0.01], "b": [-0.5, -0.01], "c": [0.2, 0.01]})
    out = cm.apply_limits(w, max_position=0.1, max_gross=0.25)
    assert out.abs().max().max() <= 0.1 + 1e-12
    assert out.abs().sum(axis=1).iloc[0] == pytest.approx(
        0.25
    )  # capped 0.3 then scaled? gross 0.3 -> 0.25
    assert out.iloc[1].tolist() == pytest.approx([0.01, -0.01, 0.01])


def test_trading_cost_known_answer() -> None:
    cfg = BacktestConfig(aum=1_000_000.0)
    idx = pd.bdate_range(START, periods=3)
    exec_w = pd.DataFrame({"A": [0.0, 0.1, 0.1]}, index=idx)
    sigma = pd.DataFrame({"A": 0.02}, index=idx)
    adv = pd.DataFrame({"A": 1_000_000.0}, index=idx)
    c = cm.trading_costs(exec_w, sigma, adv, cfg)
    assert c["commission"].iloc[1] == pytest.approx(0.1 * 0.5e-4)
    assert c["spread"].iloc[1] == pytest.approx(0.1 * 2.0e-4)
    assert c["slippage"].iloc[1] == pytest.approx(0.1 * 1.0e-4)
    assert c["impact"].iloc[1] == pytest.approx(0.1 * 0.1 * 0.02 * np.sqrt(0.1))
    assert c.iloc[2].sum() == 0  # no trade, no cost
    c2 = cm.trading_costs(exec_w, sigma, adv, cfg, cost_multiplier=2.0)
    assert c2.iloc[1].sum() == pytest.approx(2 * c.iloc[1].sum())


def test_borrow_cost_known_answer_and_label() -> None:
    idx = pd.bdate_range(START, periods=2)
    w = pd.DataFrame({"A": [0.5, 0.5], "B": [-0.5, -0.5]}, index=idx)
    assert cm.borrow_cost(w, 0.0252).iloc[0] == pytest.approx(0.5 * 0.0252 / 252)
    assert cm.borrow_cost(w, 0.0).sum() == 0
    assert BacktestConfig().borrow_rate_is_assumption is True
    with pytest.raises(ValueError, match="assumption"):
        BacktestConfig(borrow_rate_is_assumption=False)


def test_config_validation_and_yaml_defaults() -> None:
    for bad in ({"quantile": 0.6}, {"delay": -1}, {"rebalance_every": 0}, {"aum": 0}):
        with pytest.raises(ValueError):
            BacktestConfig(**bad)
    from pathlib import Path

    cfg = load_run_config(Path(__file__).resolve().parents[2] / "configs" / "ci_fixture.yaml")
    assert cfg.backtest.borrow_rate_is_assumption and cfg.backtest.rebalance_every == 21
    assert cfg.backtest.borrow_grid[0] == 0.0 and len(cfg.backtest.cost_multipliers) >= 3


def test_rebalance_holds_between_dates() -> None:
    w = pd.DataFrame({"A": np.arange(7.0)})
    out = rebalance(w, 3)
    assert out["A"].tolist() == [0, 0, 0, 3, 3, 3, 6]


def _spike_panel(spike_row: int) -> Panel:
    p = make_panel(40, ("A", "B"), seed=1)
    ret = pd.DataFrame(0.0, index=p.close.index, columns=["A", "B"])
    ret.iloc[spike_row, 0] = 0.10
    close = 100 * (1 + ret).cumprod()
    return Panel(close=close, high=close, low=close, volume=p.volume)


def test_timing_trade_at_t_plus_delay_earns_from_t_plus_delay_plus_one(guard, registry) -> None:  # type: ignore[no-untyped-def]
    cfg = zero_cost_cfg(delay=1)
    w = pd.DataFrame(0.0, index=_spike_panel(0).close.index, columns=["A", "B"])
    w.iloc[0] = [1.0, 0.0]  # decided at row 0, traded at close of row 1, earns from row 2

    def total(row: int) -> float:
        res = run_backtest(
            w,
            _spike_panel(row),
            cfg,
            guard=guard,
            registry=registry,
            trial=TrialSpec("parameter_set", f"spike{row}", {}),
        )
        return float(res.gross.sum())

    assert total(1) == pytest.approx(0.0)  # earned too early, must not count
    assert total(2) == pytest.approx(0.10)
    assert total(5) == pytest.approx(0.0)  # weights were held only if rebalanced, here fixed row 0
    # rebalance_every=1 forward fills the row 0 target, so a later spike is also earned
    ff = zero_cost_cfg(delay=1, rebalance_every=40)
    res = run_backtest(
        w,
        _spike_panel(5),
        ff,
        guard=guard,
        registry=registry,
        trial=TrialSpec("parameter_set", "ff", {}),
    )
    assert res.gross.sum() == pytest.approx(0.10)


def test_net_equals_gross_without_costs_and_is_lower_with_costs(guard, registry) -> None:  # type: ignore[no-untyped-def]
    panel = make_panel(150, ("A", "B", "C", "D"))
    rng = np.random.default_rng(0)
    w = pd.DataFrame(
        rng.normal(size=(150, 4)) * 0.02, index=panel.close.index, columns=list("ABCD")
    )
    free = run_backtest(
        w,
        panel,
        zero_cost_cfg(rebalance_every=5),
        guard=guard,
        registry=registry,
        trial=TrialSpec("signal", "free", {}),
    )
    assert (free.net - free.gross).abs().max() < 1e-15
    costly = run_backtest(
        w,
        panel,
        BacktestConfig(rebalance_every=5, max_position=1.0, max_gross=10.0),
        guard=guard,
        registry=registry,
        trial=TrialSpec("signal", "costly", {}),
    )
    assert costly.net.sum() < costly.gross.sum()
    assert set(costly.costs.columns) == {"commission", "spread", "slippage", "impact", "borrow"}
    assert costly.costs.sum().sum() > 0


def test_position_limits_bind_in_engine(guard, registry) -> None:  # type: ignore[no-untyped-def]
    panel = make_panel(60)
    w = pd.DataFrame({"A": 0.9, "B": -0.9}, index=panel.close.index)
    res = run_backtest(
        w,
        panel,
        zero_cost_cfg(max_position=0.03),
        guard=guard,
        registry=registry,
        trial=TrialSpec("signal", "lim", {}),
    )
    assert res.gross_exposure.max() == pytest.approx(0.06)


def test_engine_has_no_lookahead(guard, registry) -> None:  # type: ignore[no-untyped-def]
    panel = make_panel(150, ("A", "B", "C"))
    rng = np.random.default_rng(4)
    w = pd.DataFrame(rng.normal(size=(150, 3)) * 0.02, index=panel.close.index, columns=list("ABC"))
    cfg = BacktestConfig(rebalance_every=7, max_position=1.0, max_gross=10.0)
    base = run_backtest(
        w, panel, cfg, guard=guard, registry=registry, trial=TrialSpec("signal", "b", {})
    )
    cut = 100
    new_ret = panel.ret.copy()
    new_ret.iloc[cut + 1 :] = rng.normal(0, 0.05, new_ret.iloc[cut + 1 :].shape)
    close2 = 100 * (1 + new_ret.fillna(0)).cumprod()
    w2 = w.copy()
    w2.iloc[cut + 1 :] = rng.normal(size=w2.iloc[cut + 1 :].shape)
    other = run_backtest(
        w2,
        Panel(close2, close2, close2, panel.volume),
        cfg,
        guard=guard,
        registry=registry,
        trial=TrialSpec("signal", "c", {}),
    )
    upto = base.net.index[base.net.index <= panel.close.index[cut - 3]]
    pd.testing.assert_series_equal(base.net.loc[upto], other.net.loc[upto], check_exact=False)


def test_sensitivity_grid_is_monotone_and_not_counted(guard, registry) -> None:  # type: ignore[no-untyped-def]
    panel = make_panel(200, ("A", "B", "C", "D"), seed=5)
    rng = np.random.default_rng(1)
    w = pd.DataFrame(rng.normal(size=(200, 4)), index=panel.close.index, columns=list("ABCD"))
    w = w.sub(w.mean(axis=1), axis=0) * 0.02
    cfg = BacktestConfig(rebalance_every=3, max_position=1.0, max_gross=10.0, aum=50_000_000.0)
    grid = sensitivity_grid(w, panel, cfg, guard=guard, registry=registry, name="x")
    assert len(grid) == len(cfg.borrow_grid) * len(cfg.cost_multipliers)
    assert registry.n_trials == 0 and registry.n_sensitivity_cells == len(grid)
    piv = grid.pivot(index="borrow_rate_assumed", columns="cost_multiplier", values="sharpe_net")
    assert (piv.diff(axis=0).dropna() <= 1e-12).all().all()  # more borrow, lower Sharpe
    assert (piv.diff(axis=1).dropna(axis=1) <= 1e-12).all().all()  # more costs, lower Sharpe
    assert set(grid["borrow_rate_assumed"]) == set(cfg.borrow_grid)
    s = summarize(
        run_backtest(
            w, panel, cfg, guard=guard, registry=registry, trial=TrialSpec("signal", "z", {})
        )
    )
    assert s["annual_cost_drag"] > 0 and 0 <= s["participation_breach_share"] <= 1
