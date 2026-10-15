"""Network fetchers. UNTESTED AGAINST LIVE ENDPOINTS.

Thin on purpose: all logic lives in parsers.py, which has offline contract tests. If a source
changes format, the fix belongs in parsers.py and should be reproducible from a saved payload.
Never called in CI.
"""

from __future__ import annotations

import io
import os
import time
import zipfile

import pandas as pd
import requests

from sentinel.data.sources import parsers
from sentinel.logging import get_logger

UA_ENV = "SENTINEL_USER_AGENT"  # EDGAR and Wikipedia ask for a descriptive User-Agent with contact
FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FRENCH_FILES = {
    "FF5D": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "MOMD": "F-F_Momentum_Factor_daily_CSV.zip",
}
log = get_logger("fetch")


def user_agent() -> str:
    ua = os.environ.get(UA_ENV)
    if not ua:
        raise RuntimeError(
            f"Set {UA_ENV} to something like 'sentinel-research your-email@example.com'. "
            "Wikipedia and SEC EDGAR reject anonymous clients."
        )
    return ua


def get(url: str, retries: int = 3, pause: float = 0.3) -> requests.Response:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers={"User-Agent": user_agent()}, timeout=60)
            resp.raise_for_status()
            time.sleep(pause)  # far below the EDGAR limit of 10 requests per second
            return resp
        except requests.RequestException as exc:
            last = exc
            log.warning("fetch_retry", url=url, attempt=attempt + 1, error=str(exc))
            time.sleep(2**attempt)
    raise RuntimeError(f"giving up on {url}") from last


def fetch_wikipedia_sp500() -> tuple[list[str], pd.DataFrame, str]:
    """Return (current symbols, change log, raw html). Raw html is kept for debugging."""
    html = get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies").text
    return parsers.parse_sp500_current(html), parsers.parse_sp500_changes(html), html


def fetch_yfinance_batch(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(
        tickers, start=start, end=end, auto_adjust=False, progress=False, group_by="column"
    )
    if raw is None or raw.empty:
        return pd.DataFrame()
    if not isinstance(raw.columns, pd.MultiIndex):  # single ticker comes back flat
        raw.columns = pd.MultiIndex.from_product([raw.columns, tickers[:1]])
    return parsers.normalize_yfinance(raw)


def fetch_fred(series_id: str) -> pd.DataFrame:
    key = os.environ["FRED_API_KEY"]  # free key, no card required
    url = (
        "https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}&api_key={key}&file_type=json"
    )
    return parsers.parse_fred_observations(get(url).json(), series_id)


def fetch_french(code: str) -> pd.DataFrame:
    """Download a Ken French daily factor zip and parse its single CSV."""
    resp = get(FRENCH_BASE + FRENCH_FILES[code])
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        text = zf.read(name).decode("latin-1")
    return parsers.parse_french_daily(text, code)


CHECK_URLS = {
    "wikipedia": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "yahoo": "https://query1.finance.yahoo.com/v8/finance/chart/AAPL?range=5d&interval=1d",
    "french": FRENCH_BASE + FRENCH_FILES["FF5D"],
    "fred": "https://api.stlouisfed.org/fred/series?series_id=DGS10",
    "edgar": "https://data.sec.gov/submissions/CIK0000320193.json",
    "gdelt": "http://data.gdeltproject.org/gdeltv2/lastupdate.txt",
}


def doctor() -> dict[str, str]:
    """Connectivity check for every host. Reports status without raising."""
    out: dict[str, str] = {}
    for name, url in CHECK_URLS.items():
        try:
            r = requests.get(url, headers={"User-Agent": user_agent()}, timeout=30)
            out[name] = f"HTTP {r.status_code}"
        except Exception as exc:  # noqa: BLE001 - report everything, never crash the doctor
            out[name] = f"ERROR {type(exc).__name__}: {exc}"
    return out
