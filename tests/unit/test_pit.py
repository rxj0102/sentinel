import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from sentinel.data.pit import LookAheadError, asof_join, assert_no_lookahead, known_as_of, with_lag


def test_with_lag_and_assert() -> None:
    df = pd.DataFrame({"date": pd.to_datetime(["2020-01-01"])})
    out = with_lag(df, "date", 2)
    assert out["available_date"].iloc[0] == pd.Timestamp("2020-01-03")
    assert_no_lookahead(out)
    with pytest.raises(ValueError):
        with_lag(df, "date", -1)


def test_assert_detects_lookahead() -> None:
    df = pd.DataFrame(
        {"date": pd.to_datetime(["2020-01-02"]), "available_date": pd.to_datetime(["2020-01-01"])}
    )
    with pytest.raises(LookAheadError):
        assert_no_lookahead(df)


def test_known_as_of_is_strict() -> None:
    df = pd.DataFrame({"available_date": pd.to_datetime(["2020-01-01", "2020-01-02"])})
    assert len(known_as_of(df, pd.Timestamp("2020-01-02"))) == 1


def test_asof_join_never_uses_same_day_or_future() -> None:
    left = pd.DataFrame({"d": pd.to_datetime(["2020-01-02", "2020-01-05"])})
    right = pd.DataFrame(
        {
            "available_date": pd.to_datetime(["2020-01-02", "2020-01-04"]),
            "x": [1.0, 2.0],
        }
    )
    out = asof_join(left, right, "d", right_cols=["x"])
    assert out["x"].isna().iloc[0]  # same-day availability is excluded
    assert out["x"].iloc[1] == 2.0


@given(st.integers(min_value=0, max_value=400))
def test_lag_never_creates_lookahead(lag: int) -> None:
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=5)})
    assert_no_lookahead(with_lag(df, "date", lag))
