import json
import re
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
from sentinel.docs_check import check_docs
from sentinel.portfolio import compare
from sentinel.portfolio.build import METHODS, BuildResult, build_weights, leg_size
from sentinel.portfolio.feasibility import (
    FeasibilityError,
    check_feasibility,
    enforcement,
    required_leg_names,
    required_universe,
)
from sentinel.portfolio.ic_estimate import (
    IcEstimate,
    estimate_at,
    per_date_ic,
    resolve_ic,
)
from sentinel.portfolio.report import MODEL_DESCRIPTION, infeasible_header
from sentinel.signals.base import Panel, panel_from_prices
from sentinel.signals.composite import composite_score

ROOT = Path(__file__).resolve().parents[2]
RUN = load_run_config(ROOT / "configs" / "ci_fixture.yaml")
runner = CliRunner()

# ---- 1. naming --------------------------------------------------------------------------


def test_variant_names_are_leg_wise_everywhere() -> None:
    assert METHODS == ("mv_lw", "leg_risk_parity", "leg_hrp")
    bare = re.compile(r"(?<![A-Za-z_])(" + "risk" + "_parity|" + "h" + "rp" + r")(?![A-Za-z_])")
    offenders = [
        str(f.relative_to(ROOT))
        for d in ("src", "tests", "configs")
        for f in (ROOT / d).rglob("*.*")
        if f.suffix in {".py", ".yaml"} and bare.search(f.read_text())
    ]
    assert offenders == []
    assert "NOT true long-short risk parity" in MODEL_DESCRIPTION


def test_docs_check_enforces_the_qualifier(tmp_path: Path) -> None:
    def errors(text: str) -> list[str]:
        (tmp_path / "docs").mkdir(exist_ok=True)
        (tmp_path / "README.md").write_text(text)
        (tmp_path / "docs" / "REAL_DATA_GATE.md").write_text("Status: OPEN\n")
        return check_docs(tmp_path)

    assert errors("We use leg-wise risk parity and leg-wise HRP here.\n") == []
    assert errors("They are not true long-short risk parity.\n") == []
    assert errors("We compare risk parity today.\n")
    assert errors("The HRP variant wins.\n")
    assert errors("hierarchical risk parity is used.\n")
    assert errors("    HRP in an indented code block is fine\n") == []


# ---- 2. IC replacement procedure ---------------------------------------------------------


def _planted(
    n: int = 600, k: int = 80, effect: float = 0.01, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    z = pd.DataFrame(rng.normal(size=(n, k)), index=idx)
    ret = pd.DataFrame(rng.normal(0, 0.02, (n, k)), index=idx)
    # The return on day s + 2 loads on the score at s, matching delay 1, horizon 1 at the start.
    ret = ret + effect * z.shift(2).fillna(0.0)
    return z, 100 * (1 + ret).cumprod()


def test_estimator_recovers_planted_ic_and_uses_non_overlapping_dates() -> None:
    z, close = _planted(effect=0.02)
    ic = per_date_ic(z, close, horizon=1, delay=1)
    est = estimate_at(ic, t=500, horizon=1, delay=1)
    assert est.n_dates > 400 and est.estimate > 0.1 and est.std_error > 0
    spaced = estimate_at(per_date_ic(z, close, 21, 1), t=500, horizon=21, delay=1)
    assert spaced.n_dates <= 500 // 21 + 1


def test_estimate_is_point_in_time() -> None:
    z, close = _planted(effect=0.02)
    t = 400
    base = estimate_at(per_date_ic(z, close, 21, 1), t, 21, 1)
    changed = close.copy()
    changed.iloc[t + 1 :] = np.random.default_rng(9).uniform(1, 500, changed.iloc[t + 1 :].shape)
    again = estimate_at(per_date_ic(z, changed, 21, 1), t, 21, 1)
    assert again == base  # labels that end after t never enter the estimate
    assert estimate_at(per_date_ic(z, close, 21, 1), 5, 21, 1).n_dates == 0


def test_resolve_ic_rules() -> None:
    assert resolve_ic(IcEstimate(0.05, 0.01, 30), 0.02, 24) == (0.05, "estimate")
    assert resolve_ic(IcEstimate(0.05, 0.01, 10), 0.02, 24) == (0.02, "fallback_assumed")
    assert resolve_ic(IcEstimate(-0.03, 0.01, 30), 0.02, 24) == (0.0, "floored_zero")
    assert resolve_ic(IcEstimate(float("nan"), float("nan"), 0), 0.02, 24)[1] == "fallback_assumed"


@pytest.fixture(scope="module")
def market():  # type: ignore[no-untyped-def]
    m = make_market(periods=620, seed=11)
    panel = panel_from_prices(m.prices, m.calendar)
    mem = membership_matrix(m.initial_members, m.initial_date, m.changes, m.calendar)
    return panel, mem, composite_score(panel, mem, RUN.portfolio)


def _with(pc: PortfolioConfig) -> RunConfig:
    return RunConfig(data=RUN.data, backtest=RUN.backtest, portfolio=pc)


def test_ic_source_switch_in_the_builder_and_it_counts_as_a_trial(market, tmp_path) -> None:  # type: ignore[no-untyped-def]
    panel, mem, score = market
    # Short history on the fixture, so with the pre-registered minimum every rebalance falls back.
    strict = build_weights(
        panel, score, mem, _with(PortfolioConfig(ic_source="expanding_estimate")), "mv_lw"
    )
    assert set(strict.diagnostics["ic_source_used"]) == {"fallback_assumed"}
    assert (strict.diagnostics["ic_used"] == 0.02).all()
    loose = build_weights(
        panel,
        score,
        mem,
        _with(PortfolioConfig(ic_source="expanding_estimate", ic_min_dates=2)),
        "mv_lw",
    )
    assert set(loose.diagnostics["ic_source_used"]) <= {
        "estimate",
        "floored_zero",
        "fallback_assumed",
    }
    assert (loose.diagnostics["ic_source_used"] != "fallback_assumed").any()
    default = build_weights(panel, score, mem, RUN, "mv_lw")
    assert set(default.diagnostics["ic_source_used"]) == {"assumed"}
    # Applying the estimate registers new trials.
    far = pd.Timestamp("2100-01-01")
    guard = HoldoutGuard(far, far, tmp_path / "log.csv")
    reg = TrialRegistry()
    compare.run_portfolios(panel, mem, RUN, guard=guard, registry=reg)
    before = reg.n_trials
    compare.run_portfolios(
        panel,
        mem,
        _with(PortfolioConfig(ic_source="expanding_estimate", ic_min_dates=2)),
        guard=guard,
        registry=reg,
    )
    assert reg.n_trials > before


# ---- 3. feasibility ----------------------------------------------------------------------


def test_required_names_formula_and_leg_size() -> None:
    assert required_leg_names(0.03, 2.0) == 34
    assert required_universe(0.03, 2.0, 0.2) == 170
    assert leg_size(170, 0.2) == 34 and leg_size(169, 0.2) == 33
    assert required_leg_names(0.05, 2.0) == 20


def _counts(vals: list[int]) -> pd.Series:
    return pd.Series(vals, index=pd.bdate_range("2020-01-01", periods=len(vals)))


def test_feasibility_fails_loudly_with_numbers_and_date() -> None:
    with pytest.raises(FeasibilityError) as exc:
        check_feasibility(_counts([300, 150, 300]), RUN, enforce=True)
    msg = str(exc.value)
    assert "FEASIBILITY CHECK FAILED" in msg and "34 names" in msg and "170" in msg
    assert "150" in msg and "2020-01-02" in msg
    ok = check_feasibility(_counts([170, 400]), RUN, enforce=True)
    assert ok.passes and ok.min_observed == 170
    soft = check_feasibility(_counts([30, 40]), RUN, enforce=False)
    assert not soft.passes and not soft.enforced and "FAIL" in soft.summary()
    with pytest.raises(FeasibilityError):
        check_feasibility(_counts([]), RUN, enforce=True)


def test_real_runs_always_enforce_and_configs_are_set_as_expected() -> None:
    assert enforcement(synthetic=False, require_flag=False) is True
    assert enforcement(synthetic=True, require_flag=False) is False
    assert enforcement(synthetic=True, require_flag=True) is True
    assert load_run_config(
        ROOT / "configs" / "default.yaml"
    ).portfolio.require_full_gross_feasibility
    assert not RUN.portfolio.require_full_gross_feasibility


@pytest.mark.parametrize("command", ["backtest", "portfolio"])
def test_cli_refuses_a_real_run_on_a_small_universe(command, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    fx = tmp_path / "fx"
    assert (
        runner.invoke(app, ["data", "make-fixture", "--out", str(fx), "--periods", "700"]).exit_code
        == 0
    )
    meta = json.loads((fx / "meta.json").read_text())
    meta["synthetic"] = False  # pretend this is real data, which must always be enforced
    (fx / "meta.json").write_text(json.dumps(meta))
    monkeypatch.chdir(tmp_path)
    cfg = str(ROOT / "configs" / "ci_fixture.yaml")
    r = runner.invoke(
        app, [command, "run", "--fixture", str(fx), "--out", str(tmp_path / "res"), "--config", cfg]
    )
    assert r.exit_code == 2
    assert "FEASIBILITY CHECK FAILED" in r.output
    assert not (tmp_path / "validation").exists() or not any((tmp_path / "validation").iterdir())


# ---- 4. infeasible events in every report header ------------------------------------------


def _stub_build(statuses: dict[str, list[str]]):  # type: ignore[no-untyped-def]
    def fake(panel: Panel, score, membership, run, method):  # type: ignore[no-untyped-def]
        rows = [
            {
                "date": panel.close.index[300 + 21 * k],
                "method": method,
                "n_universe": 30,
                "feasible": st == "optimal",
                "status": st,
                "shrinkage": 0.03,
                "gross": 0.4,
                "effective_n": 10.0,
                "one_way_turnover": 0.2,
                "max_abs_exposure": 0.1,
                "max_risk_share": 0.2,
                "names_at_cap": 3,
            }
            for k, st in enumerate(statuses[method])
        ]
        return BuildResult(
            pd.DataFrame(0.0, index=panel.close.index, columns=panel.close.columns),
            pd.DataFrame(rows),
        )

    return fake


def test_fallbacks_and_skips_are_counted_and_warned(market, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    panel, mem, _ = market
    plan = {
        "mv_lw": ["optimal"] * 4,
        "leg_risk_parity": ["optimal", "infeasible", "infeasible", "optimal", "universe_too_small"],
        "leg_hrp": ["infeasible", "optimal", "optimal", "optimal"],
    }
    monkeypatch.setattr(compare, "build_weights", _stub_build(plan))
    far = pd.Timestamp("2100-01-01")
    t = compare.run_portfolios(
        panel, mem, RUN, guard=HoldoutGuard(far, far, tmp_path / "l.csv"), registry=TrialRegistry()
    )
    diag = t["portfolio_diagnostics"].set_index("method")
    assert diag.loc["leg_risk_parity", "held_weight_fallbacks"] == 2
    assert diag.loc["leg_risk_parity", "skipped_rebalances"] == 1
    assert diag.loc["leg_risk_parity", "rebalances_attempted"] == 4
    assert (
        diag.loc["leg_hrp", "held_weight_fallbacks"] == 1
        and diag.loc["mv_lw", "held_weight_fallbacks"] == 0
    )
    comp = t["portfolio_comparison"].set_index("variant")
    assert comp.loc["leg_risk_parity", "held_weight_fallbacks"] == 2
    header = "\n".join(infeasible_header(t["portfolio_diagnostics"]))
    assert "WARNING: leg_risk_parity fell back to held weights at 2 of 4 attempted" in header
    assert "WARNING: leg_hrp fell back to held weights at 1 of 4" in header
    assert "WARNING: mv_lw" not in header and "NOTE: leg_risk_parity skipped 1 of 5" in header


def test_cli_report_header_always_has_the_required_lines(tmp_path: Path) -> None:
    fx, out = tmp_path / "fx", tmp_path / "res"
    assert (
        runner.invoke(app, ["data", "make-fixture", "--out", str(fx), "--periods", "700"]).exit_code
        == 0
    )
    cfg = str(ROOT / "configs" / "ci_fixture.yaml")
    r = runner.invoke(
        app, ["portfolio", "run", "--fixture", str(fx), "--out", str(out), "--config", cfg]
    )
    assert r.exit_code == 0, r.output
    text = (out / "synthetic" / "portfolio_report.md").read_text()
    head = text.split("## Headline table")[0]
    assert head.startswith("SYNTHETIC DATA")
    for needle in (
        "NOT true long-short risk parity",
        "leg-wise HRP",
        "IC source: assumed",
        "is an assumption, not an estimate",
        "Universe feasibility check",
        "Mechanism test only",
        "Infeasible constraint events",
        "held_weight_fallbacks",
        "pre_registered_default",
    ):
        assert needle in head, needle
    meta = json.loads((out / "synthetic" / "portfolio_report.meta.json").read_text())
    assert meta["synthetic"] is True
