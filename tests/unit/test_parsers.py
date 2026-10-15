import json

import numpy as np
import pandas as pd
import pytest

from sentinel.data.sources import parsers

WIKI = """
<table><thead>
<tr><th rowspan=2>Effective Date</th><th colspan=2>Added</th>
<th colspan=2>Removed</th><th rowspan=2>Reason</th></tr>
<tr><th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th></tr></thead>
<tbody>
<tr><td>March 18, 2024</td><td>AAA</td><td>Aaa Inc</td><td>BBB</td><td>Bbb Corp</td><td>x</td></tr>
<tr><td>January 5, 2023</td><td>CCC</td><td>Ccc</td><td></td><td></td><td>y</td></tr>
</tbody></table>
"""

FRENCH = """This file was created by CMPT_ME_BEME_RETS using the 202401 CRSP database.

,Mkt-RF,SMB,HML,RF
20240102,  0.50, -0.20,  0.10, 0.01
20240103, -1.00,  0.30, -99.99, 0.01

Copyright 2024 Kenneth R. French
"""


def test_wikipedia_changes() -> None:
    out = parsers.parse_sp500_changes(WIKI)
    assert list(out["added"]) == ["CCC", "AAA"]
    assert out["removed"].isna().iloc[0] and out["removed"].iloc[1] == "BBB"
    assert out["date"].is_monotonic_increasing
    with pytest.raises(ValueError):
        parsers.parse_sp500_changes("<table><tr><td>1</td></tr></table>")


def test_french_daily_percent_and_missing() -> None:
    out = parsers.parse_french_daily(FRENCH, "FF3D")
    mkt = out[out["series_id"] == "FF3D_Mkt-RF"].sort_values("event_date")
    assert mkt["value"].tolist() == pytest.approx([0.005, -0.01])
    hml = out[out["series_id"] == "FF3D_HML"].sort_values("event_date")
    assert np.isnan(hml["value"].iloc[1])
    assert (out["available_date"] - out["event_date"]).dt.days.eq(60).all()


def test_fred_plain_and_vintage() -> None:
    plain = {
        "observations": [
            {"date": "2020-01-01", "value": "1.5"},
            {"date": "2020-01-02", "value": "."},
        ]
    }
    out = parsers.parse_fred_observations(plain, "DGS10")
    assert out["value"].iloc[0] == 1.5 and np.isnan(out["value"].iloc[1])
    assert out["available_date"].iloc[0] == pd.Timestamp("2020-01-02")
    vint = {"observations": [{"date": "2020-01-01", "value": "2", "realtime_start": "2020-02-15"}]}
    v = parsers.parse_fred_observations(vint, "GDP")
    assert v["available_date"].iloc[0] == pd.Timestamp("2020-02-15")
    json.dumps(plain)


def test_yfinance_normalizer() -> None:
    idx = pd.date_range("2020-01-01", periods=2, name="Date")
    fields = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    cols = pd.MultiIndex.from_product([fields, ["AAA", "BBB"]], names=["Price", "Ticker"])
    data = np.ones((2, 12))
    data[:, 5] = 100.0  # placeholder; fix highs and lows consistently below
    raw = pd.DataFrame(1.0, index=idx, columns=cols)
    raw.loc[:, ("High", slice(None))] = 1.1
    raw.loc[:, ("Low", slice(None))] = 0.9
    raw.loc[:, ("Volume", slice(None))] = 1000.0
    raw.loc[idx[1], ("Close", "BBB")] = np.nan
    raw.loc[idx[1], ("Adj Close", "BBB")] = np.nan
    out = parsers.normalize_yfinance(raw)
    assert len(out) == 3
    assert set(out["ticker"]) == {"AAA", "BBB"}
    assert (out["available_date"] > out["date"]).all()
