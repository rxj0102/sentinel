"""Local rebuild of real data from free sources. UNTESTED AGAINST LIVE ENDPOINTS.

Run on a machine with network access. Writes raw and curated parquet plus meta.json so that
sentinel data quality can run on the result. Nothing Yahoo derived is ever committed: data/raw
and data/curated are gitignored.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from sentinel.data.schemas import PriceSchema
from sentinel.data.sources import fetch, parsers
from sentinel.data.storage import write_table
from sentinel.data.universe import reconstruct_members_at
from sentinel.logging import get_logger

log = get_logger("rebuild")


def chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def build_universe(
    current: list[str], changes: pd.DataFrame, start: pd.Timestamp
) -> tuple[set[str], list[str]]:
    """Initial members at start and the full ticker list ever needed since start."""
    initial = reconstruct_members_at(set(current), changes, start)
    since = changes[changes["date"] > start]
    ever = set(current) | initial
    for col in ("added", "removed"):
        ever |= {t for t in since[col].dropna() if isinstance(t, str)}
    return initial, sorted(ever)


def rebuild(
    start: str,
    end: str,
    raw_dir: Path = Path("data/raw"),
    curated_dir: Path = Path("data/curated"),
    batch_size: int = 40,
    limit: int | None = None,
    dry_run: bool = False,
    with_factors: bool = True,
) -> None:
    ts = pd.Timestamp(start)
    current, changes, html = fetch.fetch_wikipedia_sp500()
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "wikipedia_sp500.html").write_text(html)
    initial, tickers = build_universe(current, changes, ts)
    log.info(
        "universe",
        current=len(current),
        initial=len(initial),
        tickers=len(tickers),
        changes=len(changes),
    )
    if limit:
        tickers = tickers[:limit]
    if dry_run:
        log.info("dry_run_done", tickers=len(tickers))
        return

    frames: list[pd.DataFrame] = []
    empty: list[str] = []
    for batch in chunked(tickers, batch_size):
        ysyms = {parsers.to_yahoo_symbol(t): t for t in batch}
        try:
            got = fetch.fetch_yfinance_batch(list(ysyms), start, end)
        except Exception as exc:  # noqa: BLE001 - one bad batch must not kill the run
            log.error("batch_failed", batch=batch[:3], error=str(exc))
            empty += batch
            continue
        if got.empty:
            empty += batch
            continue
        got["ticker"] = got["ticker"].map(ysyms).fillna(got["ticker"])
        frames.append(got)
        empty += [t for t in batch if t not in set(got["ticker"])]
        log.info("batch_ok", rows=len(got), tickers=got["ticker"].nunique())
    prices = PriceSchema.validate(pd.concat(frames, ignore_index=True))

    manifest = curated_dir / "manifest.json"
    write_table(prices, raw_dir / "prices_yahoo.parquet", raw_dir / "manifest.json")
    write_table(prices, curated_dir / "prices.parquet", manifest)
    write_table(changes, curated_dir / "changes.parquet", manifest)
    pd.DataFrame({"ticker": sorted(empty)}).to_csv(
        curated_dir / "no_price_history.csv", index=False
    )
    (curated_dir / "meta.json").write_text(
        json.dumps(
            {
                "initial_members": sorted(initial),
                "initial_date": str(ts.date()),
                "synthetic": False,
                "expected_size": 500,
                "size_tolerance": 10,
                "tickers_without_prices": len(empty),
            },
            indent=2,
        )
    )
    log.info("prices_done", rows=len(prices), tickers_without_prices=len(empty))

    if with_factors:
        for code in fetch.FRENCH_FILES:
            try:
                write_table(fetch.fetch_french(code), curated_dir / f"{code}.parquet", manifest)
            except Exception as exc:  # noqa: BLE001
                log.error("french_failed", code=code, error=str(exc))
