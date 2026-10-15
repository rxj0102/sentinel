import numpy as np
import pandas as pd
import pytest

from sentinel.data.prices_download import chunked, download_prices, exclusive_end
from sentinel.data.quarantine import quarantine_prices
from sentinel.data.schemas import PriceSchema
from sentinel.data.synthetic import make_market


def test_exclusive_end_includes_the_stated_day() -> None:
    assert exclusive_end("2022-12-30") == "2022-12-31"
    assert exclusive_end("2022-12-31") == "2023-01-01"


def test_quarantine_catches_the_real_world_problems_and_output_validates() -> None:
    p = make_market(periods=40).prices.copy()
    t = p["ticker"].iloc[0]
    rows = p.index[p["ticker"] == t]
    p = pd.concat([p, p.loc[[rows[0]]]], ignore_index=True)  # duplicate ticker and date
    p.loc[p.index[1], "close"] = 0.0
    p.loc[p.index[2], "adj_close"] = np.nan
    p.loc[p.index[3], "volume"] = -5.0
    p.loc[p.index[4], "open"] = 0.0  # optional field, blanked not dropped
    p.loc[p.index[5], ["high", "low"]] = [1.0, 2.0]  # high below low
    clean, rejected, st = quarantine_prices(p)
    assert st["duplicates_dropped"] == 1 and st["rows_rejected"] == 4
    assert st["cells_blanked"] == 1 + 2
    assert set(rejected["reason"]) == {
        "duplicate_ticker_date",
        "non_positive_or_missing_close",
        "negative_volume",
    }
    assert st["rows_in"] == len(p) and st["rows_out"] == len(clean) == len(p) - 4
    PriceSchema.validate(clean)  # the strict schema now accepts it
    with pytest.raises(Exception):  # noqa: B017 - the raw frame is rejected, as intended
        PriceSchema.validate(p)


def test_quarantine_clean_input_is_untouched() -> None:
    p = make_market(periods=40).prices
    clean, rejected, st = quarantine_prices(p)
    assert rejected.empty and st["rows_rejected"] == 0 and st["cells_blanked"] == 0
    pd.testing.assert_frame_equal(clean, p.reset_index(drop=True))


def _frame(tickers: list[str]) -> pd.DataFrame:
    return make_market(periods=30).prices.assign(ticker=tickers[0]).iloc[:5]


def test_download_retries_counts_failures_and_lists_missing() -> None:
    calls: list[list[str]] = []
    sleeps: list[float] = []

    def fetch(batch: list[str]) -> pd.DataFrame:
        calls.append(batch)
        if "BAD" in batch[0]:
            raise ConnectionError("Too Many Requests")
        if batch[0] == "EMPTY":
            return pd.DataFrame()
        return _frame(batch)  # returns only the first ticker of the batch

    res = download_prices(
        ["AAA", "BBB", "BAD1", "BAD2", "EMPTY", "ZZZ"],
        fetch,
        batch_size=2,
        retries=3,
        sleep=sleeps.append,
    )
    assert res.batches_total == 3 and res.batches_failed == 1
    assert len([c for c in calls if "BAD1" in c]) == 3 and sleeps == [2.0, 4.0, 8.0]
    assert sorted(res.without_prices) == ["BAD1", "BAD2", "BBB", "EMPTY", "ZZZ"]
    assert len(res.frames) == 1 and set(res.frames[0]["ticker"]) == {"AAA"}


def test_download_maps_yahoo_symbols_back() -> None:
    def fetch(batch: list[str]) -> pd.DataFrame:
        assert batch == ["BRK-B"]
        return _frame(["BRK-B"])

    res = download_prices(["BRK.B"], fetch, batch_size=5)
    assert set(res.frames[0]["ticker"]) == {"BRK.B"} and res.without_prices == []
    assert chunked(list("abcde"), 2) == [["a", "b"], ["c", "d"], ["e"]]
