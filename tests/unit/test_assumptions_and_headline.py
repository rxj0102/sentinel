import re
from pathlib import Path

import pytest

from sentinel.config import BacktestConfig, PortfolioConfig, load_run_config

ROOT = Path(__file__).resolve().parents[2]
ASSUMPTIONS = (ROOT / "ASSUMPTIONS.md").read_text()
EXEMPT = {"borrow_rate_is_assumption"}  # a flag, not a numeric default


def test_every_default_is_listed_in_assumptions() -> None:
    for model in (BacktestConfig, PortfolioConfig):
        for name in model.model_fields:
            if name in EXEMPT:
                continue
            if name in {"winsorize_lower", "winsorize_upper"}:
                continue  # listed together as winsorize quantiles in the amendment
            assert f"| {name} |" in ASSUMPTIONS, f"{name} missing from ASSUMPTIONS.md"
    assert (
        "not an estimate" in ASSUMPTIONS.lower()
        or "Nothing in this file is an estimate" in ASSUMPTIONS
    )


def test_assumptions_has_no_sourcing_claims_without_a_source() -> None:
    for field in ("commission_bps", "half_spread_bps", "slippage_bps", "borrow_rate_annual"):
        row = next(ln for ln in ASSUMPTIONS.splitlines() if ln.startswith(f"| {field} |"))
        assert "ASSUMPTION" in row or "Placeholder" in row or "placeholder" in row


def test_yaml_matches_preregistered_table() -> None:
    doc = (ROOT / "docs" / "hypotheses.md").read_text()
    table = dict(re.findall(r"^\| (\w+) \| ([^|]+) \|$", doc, re.M))
    cfg = load_run_config(ROOT / "configs" / "ci_fixture.yaml")
    default = load_run_config(ROOT / "configs" / "default.yaml")
    for c in (cfg, default):
        p = c.portfolio
        assert float(table["ic_assumed"]) == p.ic_assumed
        assert (
            int(table["cov_window"]) == p.cov_window == p.cov_min_obs == int(table["cov_min_obs"])
        )
        assert float(table["risk_aversion"]) == p.risk_aversion
        assert float(table["leg_quantile"]) == p.leg_quantile
        assert int(table["min_names"]) == p.min_names
        assert float(table["exposure_tol"]) == p.exposure_tol
        assert float(table["turnover_cap_one_way"]) == p.turnover_cap_one_way
        assert int(table["composite_min_signals"]) == p.composite_min_signals
        assert float(table["max_gross"]) == c.backtest.max_gross
        assert float(table["max_position"]) == c.backtest.max_position
        assert float(table["delisting_return"]) == c.backtest.delisting_return
        lo, hi = table["winsorize_quantiles"].split()
        assert (float(lo), float(hi)) == (p.winsorize_lower, p.winsorize_upper)


def test_portfolio_config_validation() -> None:
    for bad in (
        {"leg_quantile": 0.7},
        {"cov_min_obs": 300},
        {"risk_aversion": 0},
        {"composite_min_signals": 9},
        {"min_names": 1},
    ):
        with pytest.raises(ValueError):
            PortfolioConfig(**bad)
    with pytest.raises(ValueError):
        BacktestConfig(delisting_return=0.1)
