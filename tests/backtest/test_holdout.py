from pathlib import Path

import pandas as pd
import pytest

from sentinel.backtest.engine import run_backtest
from sentinel.backtest.holdout import (
    HoldoutAccessDeniedError,
    HoldoutAlreadyUsedError,
    HoldoutError,
    HoldoutGrant,
    HoldoutGuard,
)
from sentinel.backtest.trials import TrialRegistry, TrialSpec

from .conftest import make_panel, zero_cost_cfg


def _guard(
    tmp_path: Path, start: str = "2020-03-01", end: str | None = "2020-12-31"
) -> HoldoutGuard:
    return HoldoutGuard(
        pd.Timestamp(start), pd.Timestamp(end) if end else None, tmp_path / "log.csv"
    )


def _run(guard: HoldoutGuard, grant: HoldoutGrant | None = None):  # type: ignore[no-untyped-def]
    panel = make_panel(120)  # business days from 2020-01-01 reach into June, past the holdout
    w = pd.DataFrame({"A": 0.5, "B": -0.5}, index=panel.close.index)
    return run_backtest(
        w,
        panel,
        zero_cost_cfg(),
        guard=guard,
        registry=TrialRegistry(),
        trial=TrialSpec("signal", "t", {}),
        unlock=grant,
    )


def _unlock(guard: HoldoutGuard, reason: str = "single pre-registered final test") -> HoldoutGrant:
    return guard.unlock(pd.Timestamp("2020-06-30"), "test", reason)


def test_engine_fails_without_unlock_flag_and_logs_denial(tmp_path: Path) -> None:
    g = _guard(tmp_path)
    with pytest.raises(HoldoutAccessDeniedError):
        _run(g)
    log = g.access_log()
    assert list(log["action"]) == ["denied_no_unlock"]
    assert log["timestamp_utc"].iloc[0] and len(str(log["commit"].iloc[0])) >= 7
    assert str(log["holdout_start"].iloc[0]) == "2020-03-01"


def test_development_only_run_passes_and_logs_nothing(tmp_path: Path) -> None:
    g = _guard(tmp_path, start="2030-01-01", end="2030-12-31")
    _run(g)
    assert g.access_log().empty


def test_unlock_needs_end_date_reason_and_window(tmp_path: Path) -> None:
    with pytest.raises(HoldoutError, match="not locked"):
        _unlock(_guard(tmp_path, end=None))
    with pytest.raises(HoldoutError, match="reason"):
        _unlock(_guard(tmp_path / "b"), reason=" ")
    with pytest.raises(HoldoutError, match="past the fixed holdout end"):
        _unlock(_guard(tmp_path / "c", end="2020-04-01"))


def test_holdout_is_touched_once_but_one_grant_serves_a_whole_run(tmp_path: Path) -> None:
    g = _guard(tmp_path, end="2021-12-31")
    grant = _unlock(g)
    _run(g, grant)
    _run(g, grant)  # many reads under the same grant, each one logged
    assert list(g.access_log()["action"]) == ["granted", "access_under_grant", "access_under_grant"]
    with pytest.raises(HoldoutAlreadyUsedError):
        _unlock(g, "trying again")
    assert g.access_log()["action"].iloc[-1] == "denied_repeat"
    # A fresh guard object reading the same committed log is also refused.
    with pytest.raises(HoldoutAlreadyUsedError):
        _unlock(_guard(tmp_path, end="2021-12-31"), "new process")


def test_forged_or_expired_grants_are_refused(tmp_path: Path) -> None:
    g = _guard(tmp_path, end="2021-12-31")
    forged = HoldoutGrant("abc", g.holdout_start, pd.Timestamp("2021-12-31"))
    with pytest.raises(HoldoutAccessDeniedError):
        _run(g, forged)
    grant = g.unlock(pd.Timestamp("2020-03-15"), "test", "reason")
    with pytest.raises(HoldoutAccessDeniedError):  # run reaches June, grant covers mid March
        _run(g, grant)
    assert "denied_invalid_grant" in set(g.access_log()["action"])


def test_load_and_dev_only(tmp_path: Path) -> None:
    g = _guard(tmp_path)
    df = pd.DataFrame({"date": pd.to_datetime(["2020-02-28", "2020-03-01", "2020-03-02"])})
    assert len(g.dev_only(df)) == 1  # boundary: the start date itself is holdout
    with pytest.raises(HoldoutAccessDeniedError):
        g.load(df)
    assert g.load(df.iloc[:1]).equals(df.iloc[:1])
    grant = g.unlock(pd.Timestamp("2020-03-02"), "test", "load test")
    assert g.load(df, grant=grant).equals(df)
    assert g.load(df.iloc[:0]).empty


def test_synthetic_and_real_logs_are_separate(tmp_path: Path) -> None:
    t = pd.Timestamp("2020-01-01")
    s = HoldoutGuard.for_run(t, None, synthetic=True, results_root=tmp_path)
    r = HoldoutGuard.for_run(t, None, synthetic=False, results_root=tmp_path)
    assert s.log_path == tmp_path / "synthetic" / "holdout_access_log.csv"
    assert s.log_path != r.log_path
    assert r.log_path.as_posix().endswith("holdout/access_log.csv")


def test_committed_real_log_starts_empty() -> None:
    log = Path(__file__).resolve().parents[2] / "holdout" / "access_log.csv"
    assert log.read_text().count("\n") == 1  # header only, the real holdout is untouched
