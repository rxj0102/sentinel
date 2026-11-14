"""Batch download orchestration with retries. The fetch function is injected, so the logic
is testable offline. Failures are counted and surfaced, never swallowed."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from sentinel.data.sources.parsers import to_yahoo_symbol
from sentinel.logging import get_logger

log = get_logger("download")


def exclusive_end(end: str) -> str:
    """Yahoo treats the end date as exclusive. Add a day so the stated end date is included."""
    return str((pd.Timestamp(end) + pd.Timedelta(days=1)).date())


def chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


@dataclass
class DownloadResult:
    frames: list[pd.DataFrame] = field(default_factory=list)
    without_prices: list[str] = field(default_factory=list)
    batches_total: int = 0
    batches_failed: int = 0


def download_prices(
    tickers: list[str],
    fetch: Callable[[list[str]], pd.DataFrame],
    batch_size: int = 40,
    retries: int = 3,
    sleep: Callable[[float], None] = time.sleep,
) -> DownloadResult:
    """Download in batches. A batch is retried with backoff, and a batch that still fails
    is counted and its tickers are listed as without prices."""
    res = DownloadResult()
    for batch in chunked(tickers, batch_size):
        res.batches_total += 1
        ysyms = {to_yahoo_symbol(t): t for t in batch}
        got: pd.DataFrame | None = None
        for attempt in range(1, retries + 1):
            try:
                got = fetch(list(ysyms))
                break
            except Exception as exc:  # noqa: BLE001 - report every failure type
                log.warning(
                    "batch_retry",
                    batch_head=batch[:3],
                    attempt=attempt,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                sleep(2.0**attempt)
        if got is None:
            res.batches_failed += 1
            res.without_prices += batch
            log.error("batch_failed", batch_head=batch[:3], requested=len(batch))
            continue
        if got.empty:
            res.without_prices += batch
            log.warning("batch_empty", batch_head=batch[:3], requested=len(batch))
            continue
        got = got.copy()
        got["ticker"] = got["ticker"].map(ysyms).fillna(got["ticker"])
        res.frames.append(got)
        missing = [t for t in batch if t not in set(got["ticker"])]
        res.without_prices += missing
        log.info(
            "batch_ok",
            rows=len(got),
            requested=len(batch),
            returned=int(got["ticker"].nunique()),
            missing=len(missing),
        )
    return res
