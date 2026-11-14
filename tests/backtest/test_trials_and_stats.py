import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from sentinel.backtest import stats
from sentinel.backtest.engine import run_backtest, summarize
from sentinel.backtest.trials import TrialError, TrialRegistry, TrialSpec

from .conftest import make_panel, zero_cost_cfg


def test_counting_rules() -> None:
    reg = TrialRegistry()
    a = TrialSpec("signal", "max", {"w": 21})
    reg.register(a)
    reg.register(TrialSpec("signal", "max", {"w": 63}))  # new parameter set, new trial
    reg.register(TrialSpec("portfolio_variant", "max_long_only", {}))
    assert reg.n_trials == 3
    reg.register(a)  # identical rerun
    assert reg.n_trials == 3 and reg.n_reruns == 1 and reg.n_evaluations == 4
    reg.register(TrialSpec("sensitivity", "max|borrow=0", {}))
    assert reg.n_trials == 3 and reg.n_sensitivity_cells == 1 and reg.n_evaluations == 5
    with pytest.raises(TrialError):
        TrialSpec("bogus", "x", {})
    with pytest.raises(TrialError):
        reg.record_result("deadbeef", 0.1, 10, 0.0, 3.0)


def test_counter_persists_across_processes(tmp_path: Path) -> None:
    path = tmp_path / "trials.csv"
    TrialRegistry(path).register(TrialSpec("signal", "x", {"a": 1}))
    second = TrialRegistry(path)
    assert second.n_trials == 1
    second.register(TrialSpec("signal", "x", {"a": 1}))
    assert second.n_reruns == 1  # the rerun is visible even in a new process
    assert TrialRegistry(path).n_evaluations == 2


def test_variance_requires_results_when_multiple_trials() -> None:
    reg = TrialRegistry()
    reg.register(TrialSpec("signal", "a", {}))
    assert reg.sharpe_variance() == 0.0
    reg.register(TrialSpec("signal", "b", {}))
    with pytest.raises(TrialError):
        reg.sharpe_variance()


def test_engine_registers_every_run_and_requires_a_trial(guard, registry) -> None:  # type: ignore[no-untyped-def]
    panel = make_panel()
    w = pd.DataFrame({"A": 0.5, "B": -0.5}, index=panel.close.index)
    cfg = zero_cost_cfg()
    with pytest.raises(TypeError):
        run_backtest(w, panel, cfg, guard=guard, registry=registry)  # type: ignore[call-arg]
    run_backtest(w, panel, cfg, guard=guard, registry=registry, trial=TrialSpec("signal", "s", {}))
    assert registry.n_trials == 1 and registry.n_evaluations == 1
    run_backtest(w, panel, cfg, guard=guard, registry=registry, trial=TrialSpec("signal", "s", {}))
    assert registry.n_trials == 1 and registry.n_reruns == 1  # not silent
    run_backtest(
        w, panel, cfg, guard=guard, registry=registry, trial=TrialSpec("signal", "s", {"v": 2})
    )
    assert registry.n_trials == 2


def test_expected_max_sharpe_known_values() -> None:
    assert stats.expected_max_sharpe(1, 1.0) == 0.0
    assert stats.expected_max_sharpe(100, 1.0) == pytest.approx(2.5, abs=0.1)
    assert stats.expected_max_sharpe(100, 4.0) == pytest.approx(
        2 * stats.expected_max_sharpe(100, 1.0)
    )


def test_psr_matches_independent_formula() -> None:
    sr, n, skew, kurt = 0.1, 500, -0.3, 4.0
    z = (sr - 0.02) * math.sqrt(n - 1) / math.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2)
    assert stats.probabilistic_sharpe(sr, 0.02, n, skew, kurt) == pytest.approx(NormalDist().cdf(z))
    assert stats.probabilistic_sharpe(sr, sr, n) == pytest.approx(0.5)
    assert math.isnan(stats.probabilistic_sharpe(sr, 0, 2))


def test_dsr_equals_psr_for_single_trial_and_falls_with_more_trials() -> None:
    r = pd.Series(np.random.default_rng(2).normal(0.001, 0.01, 1000))
    one = TrialRegistry()
    one.register(TrialSpec("signal", "a", {}))
    d1 = stats.deflated_sharpe(r, one)
    assert d1["dsr_probability"] == pytest.approx(d1["psr_vs_zero"])

    def with_trials(n: int) -> float:
        reg = TrialRegistry()
        rng = np.random.default_rng(3)
        for k in range(n):
            d = reg.register(TrialSpec("signal", f"s{k}", {}))
            reg.record_result(d, float(rng.normal(0.0, 0.05)), 1000, 0.0, 3.0)
        return stats.deflated_sharpe(r, reg)["dsr_probability"]

    assert with_trials(50) < with_trials(5) < d1["dsr_probability"]


def test_multiple_testing_known_answers() -> None:
    p = [0.01, 0.04, 0.03, 0.005]
    assert stats.holm_adjust(p) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert stats.benjamini_hochberg(p) == pytest.approx([0.02, 0.04, 0.04, 0.02])


@given(st.lists(st.floats(min_value=1e-6, max_value=1.0), min_size=1, max_size=30))
def test_adjustments_are_ordered(p: list[float]) -> None:
    holm, bh = stats.holm_adjust(p), stats.benjamini_hochberg(p)
    for raw, h, b in zip(p, holm, bh, strict=True):
        assert raw - 1e-12 <= h <= 1.0 and b <= h + 1e-12 and b >= raw - 1e-12


def test_max_drawdown_and_sharpe_helpers() -> None:
    assert stats.max_drawdown(pd.Series([0.1, -0.5, 0.1])) == pytest.approx(-0.5)
    r = pd.Series([0.01, 0.03, 0.02, 0.0])
    assert stats.sharpe_per_period(r) == pytest.approx(r.mean() / r.std(ddof=1))
    assert 0 <= stats.sharpe_pvalue(pd.Series(np.random.default_rng(0).normal(0, 1, 200))) <= 1
    assert math.isnan(stats.sharpe_per_period(pd.Series([0.0, 0.0])))


def test_summarize_keys(guard, registry) -> None:  # type: ignore[no-untyped-def]
    panel = make_panel()
    w = pd.DataFrame({"A": 0.5, "B": -0.5}, index=panel.close.index)
    res = run_backtest(
        w,
        panel,
        zero_cost_cfg(),
        guard=guard,
        registry=registry,
        trial=TrialSpec("signal", "s", {}),
    )
    s = summarize(res)
    assert s["borrow_rate_assumed"] == 0.0 and "sharpe_net" in s and s["n_days"] > 50
