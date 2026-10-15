"""Network fetchers. Thin on purpose: all logic lives in parsers.py. Never called in CI."""

from __future__ import annotations

import os
import time

import pandas as pd
import requests

from sentinel.data.sources import parsers

EDGAR_UA_ENV = "SENTINEL_USER_AGENT"  # EDGAR fair access requires a descriptive User-Agent


def _get(url: str, **kw: object) -> requests.Response:
    ua = os.environ.get(EDGAR_UA_ENV, "sentinel-research contact-not-set")
    resp = requests.get(url, headers={"User-Agent": ua}, timeout=60, **kw)  # type: ignore[arg-type]
    resp.raise_for_status()
    time.sleep(0.2)  # stay far below 10 requests per second
    return resp


def fetch_sp500_changes() -> pd.DataFrame:
    return parsers.parse_sp500_changes(
        _get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies").text
    )


def fetch_yfinance(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)
    return parsers.normalize_yfinance(raw)


def fetch_fred(series_id: str) -> pd.DataFrame:
    key = os.environ["FRED_API_KEY"]  # free key, no card required
    url = (
        "https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}&api_key={key}&file_type=json"
    )
    return parsers.parse_fred_observations(_get(url).json(), series_id)
