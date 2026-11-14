"""Pandera schemas for every table that crosses a module boundary.

Every table carries available_date, the first date a trader could have used the row.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series


class PriceSchema(pa.DataFrameModel):
    date: Series[pd.Timestamp]
    ticker: Series[str]
    open: Series[float] = pa.Field(gt=0, nullable=True)
    high: Series[float] = pa.Field(gt=0, nullable=True)
    low: Series[float] = pa.Field(gt=0, nullable=True)
    close: Series[float] = pa.Field(gt=0)
    adj_close: Series[float] = pa.Field(gt=0)
    volume: Series[float] = pa.Field(ge=0)
    available_date: Series[pd.Timestamp]

    class Config:
        strict = True
        unique = ["date", "ticker"]

    @pa.dataframe_check
    @classmethod
    def high_low_consistent(cls, df: pd.DataFrame) -> Series[bool]:
        ok = (df["high"] >= df["low"]) | df["high"].isna() | df["low"].isna()
        return ok  # type: ignore[return-value]

    @pa.dataframe_check
    @classmethod
    def available_not_before_event(cls, df: pd.DataFrame) -> Series[bool]:
        return df["available_date"] >= df["date"]  # type: ignore[return-value]


class MembershipChangeSchema(pa.DataFrameModel):
    """One row per index change event. added or removed may be empty, not both."""

    date: Series[pd.Timestamp]
    added: Series[str] = pa.Field(nullable=True)
    removed: Series[str] = pa.Field(nullable=True)

    class Config:
        strict = True

    @pa.dataframe_check
    @classmethod
    def something_changed(cls, df: pd.DataFrame) -> Series[bool]:
        return df["added"].notna() | df["removed"].notna()  # type: ignore[return-value]


class SeriesSchema(pa.DataFrameModel):
    """Long format macro or factor series with vintage awareness."""

    series_id: Series[str]
    event_date: Series[pd.Timestamp]
    available_date: Series[pd.Timestamp]
    value: Series[float] = pa.Field(nullable=True)

    class Config:
        strict = True

    @pa.dataframe_check
    @classmethod
    def available_not_before_event(cls, df: pd.DataFrame) -> Series[bool]:
        return df["available_date"] >= df["event_date"]  # type: ignore[return-value]
