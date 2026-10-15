import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from sentinel.backtest.holdout import HoldoutGuard
from sentinel.backtest.trials import TrialRegistry
from sentinel.cli import app
from sentinel.config import PortfolioConfig, RunConfig, load_run_config
from sentinel.data.synthetic import make_market
from sentinel.data.universe import membership_matrix
from sentinel.portfolio.build import METHODS, build_weights, constraint_set
from sentinel.portfolio.compare import HEADLINE_CELL, adopt_cell_as_configuration, run_portfolios
from sentinel.signals.base import Panel, panel_from_prices
from sentinel.signals.composite import composite_score

ROOT = Path(__file__).resolve().parents[2]
RUN = load_run_config(ROOT / "configs" / "ci_fixture.yaml")
runner = CliRunner()


@pytest.fixture(scope="module")
def market():  # type: ignore[no-untyped-def]
    m = make_market(periods=620, seed=11)
    panel = panel_from_prices(m.prices, m.calendar)
    mem = membership_matrix(m.initial_members, m.initial_date, m.changes, m.calendar)
    score = composite_score(panel, mem, RUN.portfolio)
    return m, panel, mem, score


@pytest.fixture(scope="module")
def built(market):  # type: ignore[no-untyped-def]
    _, panel, mem, score = market
    return {meth: build_weights(panel, score, mem, RUN, meth) for meth in METHODS}


def test_composite_is_signed_standardized_and_respects_min_signals(market) -> None:  # type: ignore[no-untyped-def]
    _, panel, mem, score = market
    assert score.notna().sum().sum() > 0
    strict = composite_score(panel, mem, PortfolioConfig(composite_min_signals=6))
    loose = composite_score(panel, mem, PortfolioConfig(composite_min_signals=1))
    assert strict.notna().sum().sum() <= score.notna().sum().sum() <= loose.notna().sum().sum()
    assert not score[~mem.reindex_like(score).astype(bool)].notna().any().any()  # members only


def test_composite_ignores_the_future(market) -> None:  # type: ignore[no-untyped-def]
    _, panel, mem, score = market
    cut = 400
    rng = np.random.default_rng(0)
    frames = []
    for f in (panel.close, panel.high, panel.low, panel.volume):
        g = f.copy()
        g.iloc[cut + 1 :] = rng.uniform(1, 1000, g.iloc[cut + 1 :].shape)
        frames.append(g)
    other = composite_score(Panel(*frames), mem, RUN.portfolio)
    np.testing.assert_allclose(
        score.iloc[: cut + 1].to_numpy(), other.iloc[: cut + 1].to_numpy(), equal_nan=True
    )


@pytest.mark.parametrize("method", METHODS)
def test_weights_satisfy_every_constraint_at_every_rebalance(market, built, method) -> None:  # type: ignore[no-untyped-def]
    _, panel, mem, _ = market
    cons = constraint_set(RUN)
    w = built[method].weights
    active = w[(w != 0).any(axis=1)]
    assert len(active) >= 5
    prev = np.zeros(w.shape[1])
    tol = 1e-5
    for _, row in active.iterrows():
        v = row.to_numpy()
        assert abs(v.sum()) < tol and np.abs(v).sum() <= cons.max_gross + tol
        assert np.abs(v).max() <= cons.max_position + tol
        assert 0.5 * np.abs(v - prev).sum() <= cons.turnover_cap_one_way + tol
        prev = v
    d = built[method].diagnostics
    assert (d["max_abs_exposure"].dropna() <= cons.exposure_tol + tol).all()
    # Only point in time members are ever held.
    held = w != 0
    assert not (held & ~mem.reindex_like(w).astype(bool)).any().any()


def test_weights_exist_only_on_rebalance_rows_after_warmup(market, built) -> None:  # type: ignore[no-untyped-def]
    every, window = RUN.backtest.rebalance_every, RUN.portfolio.cov_window
    for b in built.values():
        rows = np.flatnonzero((b.weights != 0).any(axis=1))
        assert (rows % every == 0).all() and rows.min() >= window


def test_constructions_differ_but_share_the_same_inputs(built) -> None:  # type: ignore[no-untyped-def]
    a, b, c = (built[m].weights.to_numpy() for m in METHODS)
    assert not np.allclose(a, b) and not np.allclose(b, c) and not np.allclose(a, c)


def test_build_has_no_lookahead(market, built) -> None:  # type: ignore[no-untyped-def]
    _, panel, mem, score = market
    cut = 400
    rng = np.random.default_rng(1)
    frames = []
    for f in (panel.close, panel.high, panel.low, panel.volume):
        g = f.copy()
        g.iloc[cut + 1 :] = rng.uniform(1, 1000, g.iloc[cut + 1 :].shape)
        frames.append(g)
    p2 = Panel(*frames)
    s2 = composite_score(p2, mem, RUN.portfolio)
    again = build_weights(p2, s2, mem, RUN, "mv_lw")
    np.testing.assert_allclose(
        built["mv_lw"].weights.iloc[: cut + 1].to_numpy(),
        again.weights.iloc[: cut + 1].to_numpy(),
        atol=1e-6,
    )


def test_unknown_method_and_small_universe(market) -> None:  # type: ignore[no-untyped-def]
    _, panel, mem, score = market
    with pytest.raises(ValueError):
        build_weights(panel, score, mem, RUN, "bogus")
    tiny = RunConfig(data=RUN.data, backtest=RUN.backtest, portfolio=PortfolioConfig(min_names=500))
    res = build_weights(panel, score, mem, tiny, "hrp")
    assert (res.weights == 0).all().all()
    assert set(res.diagnostics["status"]) == {"universe_too_small"}


@pytest.fixture(scope="module")
def tables(market, tmp_path_factory):  # type: ignore[no-untyped-def]
    _, panel, mem, _ = market
    tmp = tmp_path_factory.mktemp("cmp")
    far = pd.Timestamp("2100-01-01")
    guard = HoldoutGuard(far, far, tmp / "log.csv")
    reg = TrialRegistry()
    return run_portfolios(panel, mem, RUN, guard=guard, registry=reg), reg


def test_headline_uses_only_the_pre_registered_default_cell(tables) -> None:  # type: ignore[no-untyped-def]
    t, _ = tables
    head, sens = t["portfolio_comparison"], t["portfolio_sensitivity"]
    bt = RUN.backtest
    assert (head["cell"] == HEADLINE_CELL).all()
    assert (head["borrow_rate_assumed"] == bt.borrow_rate_annual).all()
    assert (head["cost_multiplier"] == 1.0).all()
    assert len(head) == 4  # one headline row per variant, never a best cell
    # Every cell is reported, and the headline equals its own cell in the grid.
    assert len(sens) == 4 * len(bt.borrow_grid) * len(bt.cost_multipliers)
    for _, h in head.iterrows():
        cell = sens[
            (sens["signal"] == f"composite|{h['variant']}")
            & (sens["borrow_rate_assumed"] == bt.borrow_rate_annual)
            & (sens["cost_multiplier"] == 1.0)
        ]
        assert len(cell) == 1 and cell["sharpe_net"].iloc[0] == pytest.approx(h["sharpe_net"])


def test_every_portfolio_variant_is_a_trial_and_cells_are_not(tables) -> None:  # type: ignore[no-untyped-def]
    t, reg = tables
    assert reg.n_trials == 4  # equal weight baseline plus three constructions
    assert reg.n_sensitivity_cells == 4 * 20
    dsr = t["portfolio_deflated_sharpe"]
    assert (dsr["n_trials"] == 4).all() and {"p_holm", "p_bh"} <= set(dsr.columns)
    assert set(t["portfolio_diagnostics"]["method"]) == set(METHODS)


def test_adopting_a_cell_counts_as_a_trial() -> None:
    reg = TrialRegistry()
    before = reg.n_trials
    adopt_cell_as_configuration(reg, "composite|mv_lw", borrow_rate=0.0, cost_multiplier=0.5)
    assert reg.n_trials == before + 1


def test_cli_end_to_end_labels_synthetic_and_stacks_trials(tmp_path: Path) -> None:
    fx, out = tmp_path / "fx", tmp_path / "res"
    assert (
        runner.invoke(app, ["data", "make-fixture", "--out", str(fx), "--periods", "700"]).exit_code
        == 0
    )
    cfg = str(ROOT / "configs" / "ci_fixture.yaml")
    common = ["--fixture", str(fx), "--out", str(out), "--config", cfg]
    r1 = runner.invoke(app, ["backtest", "run", *common])
    assert r1.exit_code == 0, r1.output
    r2 = runner.invoke(app, ["portfolio", "run", *common])
    assert r2.exit_code == 0, r2.output
    syn = out / "synthetic"
    assert json.loads((syn / "portfolio_comparison.meta.json").read_text())["synthetic"] is True
    counts = pd.read_csv(syn / "portfolio_trial_counts.csv").iloc[0]
    assert counts["n_trials_counted"] == 12 + 4  # backtest trials plus portfolio variants
