import numpy as np
import pandas as pd
import pandera.errors as pae
import pytest

from sentinel.data import quality
from sentinel.data.schemas import PriceSchema
from sentinel.data.synthetic import make_market
from sentinel.data.universe import membership_matrix


def _toy() -> tuple[pd.DataFrame, pd.DataFrame]:
    cal = pd.bdate_range("2020-01-01", periods=4)
    mem = pd.DataFrame(True, index=cal, columns=["A", "B"])
    close = pd.DataFrame({"A": [1.0, 1.1, 1.2, 1.3], "B": [1.0, 1.0, np.nan, np.nan]}, index=cal)
    return mem, close


def test_coverage_known_answer() -> None:
    mem, close = _toy()
    cov = quality.coverage_by_year(mem, close)
    assert cov.loc[0, "member_days"] == 8
    assert cov.loc[0, "missing_days"] == 2
    assert cov.loc[0, "missing_frac"] == pytest.approx(0.25)
    assert cov.loc[0, "members_never_priced"] == 0


def test_never_priced_counted() -> None:
    mem, close = _toy()
    cov = quality.coverage_by_year(mem, close.drop(columns="B"))
    assert cov.loc[0, "missing_days"] == 4
    assert cov.loc[0, "members_never_priced"] == 1


def test_delisting_return_applied_after_last_price() -> None:
    mem, close = _toy()
    rets = close.pct_change(fill_method=None)
    adj = quality.apply_delisting_returns(rets, mem, -0.5)
    assert adj["B"].iloc[2] == -0.5
    assert np.isnan(adj["B"].iloc[3])
    assert adj["A"].equals(rets["A"])


def test_sensitivity_ordering_on_synthetic() -> None:
    m = make_market()
    mem = membership_matrix(m.initial_members, m.initial_date, m.changes, m.calendar)
    rep = quality.quality_report(m.prices, mem)
    s = rep["survivorship_sensitivity"].set_index("scenario")["ann_return"]  # type: ignore[union-attr]
    # Harsher assumptions must never raise the estimated return.
    assert s["delist_return_-1.00"] <= s["delist_return_-0.30"] <= s["baseline_missing_dropped"]
    assert s["missing_imputed_penalty_0.15"] <= s["missing_imputed_penalty_0.05"]
    assert s["missing_imputed_penalty_0.05"] <= s["baseline_missing_dropped"]
    assert all(v == 0 for v in rep["integrity"].values())  # type: ignore[union-attr]


def test_dropped_names_show_up_as_missing() -> None:
    m = make_market()
    mem = membership_matrix(m.initial_members, m.initial_date, m.changes, m.calendar)
    cov = quality.coverage_by_year(mem, quality.wide_close(m.prices, m.calendar))
    assert cov["missing_frac"].max() > 0
    assert cov["members_never_priced"].max() == len(m.dropped)


def test_integrity_flags_problems() -> None:
    m = make_market(periods=60)
    p = m.prices.copy()
    t = p["ticker"].iloc[0]
    idx = p.index[p["ticker"] == t][:8]
    p.loc[idx, "adj_close"] = 10.0  # stale run
    p.loc[idx[3], "adj_close"] = 10.0
    out = quality.integrity_checks(p, max_stale_days=5)
    assert out["stale_runs_over_limit"] >= 1
    p2 = pd.concat([p, p.iloc[:1]])
    assert quality.integrity_checks(p2)["duplicate_ticker_dates"] == 1


def test_schema_rejects_bad_rows() -> None:
    m = make_market(periods=30)
    bad = m.prices.copy()
    bad.loc[bad.index[0], "close"] = -1.0
    with pytest.raises(pae.SchemaError):
        PriceSchema.validate(bad)
    bad = m.prices.copy()
    bad["available_date"] = bad["date"] - pd.Timedelta(days=1)
    with pytest.raises(pae.SchemaError):
        PriceSchema.validate(bad)
    bad = m.prices.copy()
    bad.loc[bad.index[0], "high"] = 0.01
    with pytest.raises(pae.SchemaError):
        PriceSchema.validate(bad)


def test_synthetic_is_deterministic() -> None:
    a, b = make_market(seed=3), make_market(seed=3)
    assert a.prices.equals(b.prices) and a.changes.equals(b.changes)
    assert not a.prices.equals(make_market(seed=4).prices)


def test_ticker_reuse_flags_known_answer() -> None:
    cal = pd.bdate_range("2020-01-01", periods=60)
    mem = pd.DataFrame(False, index=cal, columns=["OK", "LATE", "NOPX", "BACK", "NEVER"])
    mem["OK"] = True
    mem["LATE"] = True
    mem["NOPX"] = True
    mem.loc[cal[0] : cal[9], "BACK"] = True
    mem.loc[cal[30] :, "BACK"] = True
    close = pd.DataFrame(1.0, index=cal, columns=["OK", "LATE", "BACK"])
    close.loc[: cal[24], "LATE"] = np.nan  # first price 35 calendar days after the add date
    flags = quality.ticker_reuse_flags(mem, close, tolerance_days=10).set_index("ticker")
    assert "NEVER" not in flags.index
    assert not flags.loc["OK", "late_first_price"] and not flags.loc["OK", "multiple_spells"]
    assert flags.loc["LATE", "late_first_price"] and flags.loc["LATE", "days_late"] > 10
    assert flags.loc["NOPX", "late_first_price"] and pd.isna(flags.loc["NOPX", "first_price_date"])
    assert flags.loc["BACK", "multiple_spells"] and flags.loc["BACK", "spells"] == 2


def test_report_contains_reuse_section_on_synthetic() -> None:
    from sentinel.data.report import render_report

    m = make_market()
    mem = membership_matrix(m.initial_members, m.initial_date, m.changes, m.calendar)
    text = render_report(quality.quality_report(m.prices, mem, expected_size=30), True)
    assert "Ticker reuse and history alignment flags" in text
