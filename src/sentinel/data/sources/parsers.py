"""Pure parsing of source payloads into schema-conformant tables. No network here.

Keeping parsing separate from fetching lets contract tests run offline on small canned
payloads, so CI never touches the network.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd

from sentinel.data.pit import with_lag
from sentinel.data.schemas import MembershipChangeSchema, SeriesSchema


def parse_sp500_changes(html: str) -> pd.DataFrame:
    """Parse the Wikipedia S and P 500 changes table (Added and Removed ticker columns)."""
    tables = pd.read_html(io.StringIO(html))
    for t in tables:
        if isinstance(t.columns, pd.MultiIndex) and any(
            str(c[0]).lower().startswith("added") for c in t.columns
        ):
            cols = {tuple(str(x) for x in c): c for c in t.columns}
            date_col = next(c for k, c in cols.items() if "date" in k[0].lower())
            added = next(
                c for k, c in cols.items() if k[0].lower() == "added" and "ticker" in k[1].lower()
            )
            removed = next(
                c for k, c in cols.items() if k[0].lower() == "removed" and "ticker" in k[1].lower()
            )
            out = pd.DataFrame(
                {
                    "date": pd.to_datetime(t[date_col], errors="coerce"),
                    "added": t[added].astype("object"),
                    "removed": t[removed].astype("object"),
                }
            ).dropna(subset=["date"])
            out = out.astype({"added": "object", "removed": "object"})
            for c in ("added", "removed"):
                out[c] = out[c].where(out[c].notna(), None)
            out = out[out["added"].notna() | out["removed"].notna()]
            out = out.sort_values("date").reset_index(drop=True)
            return MembershipChangeSchema.validate(out)
    raise ValueError("no S and P 500 changes table found")


def parse_sp500_current(html: str) -> list[str]:
    """Parse the current constituents table (Symbol column) from the same Wikipedia page."""
    for t in pd.read_html(io.StringIO(html)):
        if not isinstance(t.columns, pd.MultiIndex) and "Symbol" in t.columns:
            return [str(x).strip() for x in t["Symbol"].dropna()]
    raise ValueError("no constituents table found")


def to_yahoo_symbol(symbol: str) -> str:
    """Wikipedia writes BRK.B, Yahoo expects BRK-B."""
    return symbol.strip().replace(".", "-")


def parse_french_daily(text: str, series_prefix: str, lag_days: int = 60) -> pd.DataFrame:
    """Parse a Ken French daily factor CSV into long SeriesSchema format.

    Values are published in percent and converted to decimals. The library is revised and
    updated with a delay, so a conservative availability lag is applied.
    """
    lines = text.splitlines()
    header_idx = next(i for i, ln in enumerate(lines) if ln.strip().startswith(","))
    names = [x.strip() for x in lines[header_idx].split(",")[1:]]
    rows: list[list[str]] = []
    for ln in lines[header_idx + 1 :]:
        parts = [p.strip() for p in ln.split(",")]
        if not parts or not parts[0].isdigit() or len(parts[0]) != 8:
            if rows:
                break
            continue
        rows.append(parts)
    df = pd.DataFrame(rows, columns=["date", *names])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    long = df.melt(id_vars="date", var_name="name", value_name="value")
    long["value"] = pd.to_numeric(long["value"], errors="coerce") / 100.0
    long.loc[long["value"] <= -0.99, "value"] = np.nan  # French sentinel for missing
    long["series_id"] = series_prefix + "_" + long["name"]
    long = long.rename(columns={"date": "event_date"})
    long = with_lag(long, "event_date", lag_days)
    return SeriesSchema.validate(long[["series_id", "event_date", "available_date", "value"]])


def parse_fred_observations(
    payload: dict[str, Any], series_id: str, lag_days: int = 1
) -> pd.DataFrame:
    """Parse a FRED or ALFRED observations response.

    If realtime_start is present (ALFRED vintages), it is the availability date. Otherwise a
    fixed lag after the observation date is used.
    """
    obs = pd.DataFrame(payload["observations"])
    obs["event_date"] = pd.to_datetime(obs["date"])
    obs["value"] = pd.to_numeric(obs["value"].replace(".", np.nan), errors="coerce").astype(float)
    if "realtime_start" in obs:
        obs["available_date"] = pd.to_datetime(obs["realtime_start"])
        obs["available_date"] = obs[["available_date", "event_date"]].max(axis=1)
    else:
        obs = with_lag(obs, "event_date", lag_days)
    obs["series_id"] = series_id
    return SeriesSchema.validate(obs[["series_id", "event_date", "available_date", "value"]])


def normalize_yfinance(raw: pd.DataFrame, lag_days: int = 1) -> pd.DataFrame:
    """Convert yfinance download output (auto_adjust=False, column MultiIndex of field and
    ticker) into long format with the PriceSchema columns and dtypes. Rows without a close are
    dropped. Strict validation is deferred to after quarantine in the rebuild, so a few bad
    real rows are set aside and counted instead of aborting the whole download."""
    stacked = raw.stack(level=1, future_stack=True).reset_index()
    stacked.columns = [str(c).lower().replace(" ", "_") for c in stacked.columns]
    stacked = stacked.rename(columns={"level_1": "ticker", "price": "ticker"})
    if "ticker" not in stacked:
        stacked = stacked.rename(columns={stacked.columns[1]: "ticker"})
    keep = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
    out = stacked[keep].dropna(subset=["close", "adj_close"]).copy()
    num = ["open", "high", "low", "close", "adj_close", "volume"]
    out[num] = out[num].astype(float)  # yfinance may return integer volume
    out["volume"] = out["volume"].fillna(0.0)
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).astype("datetime64[ns]")
    out = with_lag(out, "date", lag_days).sort_values(["ticker", "date"]).reset_index(drop=True)
    out["available_date"] = out["available_date"].astype("datetime64[ns]")
    # Quarantine runs before strict validation in the rebuild, so bad real rows do not abort it.
    return out
