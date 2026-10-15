"""Quarantine of invalid price rows instead of aborting a whole rebuild.

Real Yahoo data contains duplicate rows, zero prices and high below low. The strict schema
would reject the entire table. Fatal rows are set aside with a reason and counted, optional
fields that are invalid are blanked and counted. Nothing is silently fixed.
"""

from __future__ import annotations

import pandas as pd


def quarantine_prices(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Return (clean, rejected, stats). Rejected carries a reason column."""
    work = df.copy()
    rejected: list[pd.DataFrame] = []

    def reject(mask: pd.Series, reason: str) -> None:
        nonlocal work
        if bool(mask.any()):
            bad = work[mask].copy()
            bad["reason"] = reason
            rejected.append(bad)
            work = work[~mask]

    dup = work.duplicated(["ticker", "date"], keep="last")
    reject(dup, "duplicate_ticker_date")
    reject(~(work["close"] > 0) | ~(work["adj_close"] > 0), "non_positive_or_missing_close")
    reject(work["volume"] < 0, "negative_volume")
    reject(work["available_date"] < work["date"], "available_before_event")

    sanitized = 0
    for col in ("open", "high", "low"):
        bad = work[col].notna() & ~(work[col] > 0)
        sanitized += int(bad.sum())
        work.loc[bad, col] = float("nan")
    inverted = work["high"].notna() & work["low"].notna() & (work["high"] < work["low"])
    sanitized += int(2 * inverted.sum())
    work.loc[inverted, ["high", "low"]] = float("nan")

    rej = (
        pd.concat(rejected, ignore_index=True)
        if rejected
        else pd.DataFrame(columns=[*df.columns, "reason"])
    )
    stats = {
        "rows_in": len(df),
        "rows_rejected": len(rej),
        "duplicates_dropped": int(dup.sum()),
        "cells_blanked": sanitized,
        "rows_out": len(work),
    }
    return work.reset_index(drop=True), rej, stats
