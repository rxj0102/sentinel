"""Point-in-time helpers. All joins in the platform use available_date."""

from __future__ import annotations

import pandas as pd


class LookAheadError(ValueError):
    """Raised when a table would let a trade use information before it was known."""


def with_lag(df: pd.DataFrame, event_col: str, lag_days: int) -> pd.DataFrame:
    """Add available_date = event date plus a fixed calendar lag."""
    if lag_days < 0:
        raise ValueError("lag_days must be non-negative")
    out = df.copy()
    out["available_date"] = out[event_col] + pd.Timedelta(days=lag_days)
    return out


def assert_no_lookahead(df: pd.DataFrame, event_col: str = "date") -> None:
    bad = df["available_date"] < df[event_col]
    if bool(bad.any()):
        raise LookAheadError(f"{int(bad.sum())} rows have available_date before {event_col}")


def known_as_of(df: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Rows a trader could see at the start of as_of (strictly earlier availability)."""
    return df[df["available_date"] < as_of]


def asof_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_date: str,
    by: str | None = None,
    right_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Attach the latest right row with available_date strictly before left_date.

    Strictness (allow_exact_matches=False) means same-day information is never used.
    """
    cols = right_cols or [c for c in right.columns if c not in ("available_date", by)]
    r = right.sort_values("available_date")
    keep = ["available_date", *cols] + ([by] if by else [])
    merged = pd.merge_asof(
        left.sort_values(left_date),
        r[keep],
        left_on=left_date,
        right_on="available_date",
        by=by,
        allow_exact_matches=False,
        suffixes=("", "_right"),
    )
    return merged
