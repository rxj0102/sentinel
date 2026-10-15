"""Local rebuild of real data from free sources. UNTESTED AGAINST LIVE ENDPOINTS.

Run on a machine with network access. Writes raw and curated parquet plus meta.json so that
sentinel data quality can run on the result. Nothing Yahoo derived is ever committed: data/raw
and data/curated are gitignored.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from sentinel.data.prices_download import download_prices, exclusive_end
from sentinel.data.quarantine import quarantine_prices
from sentinel.data.schemas import PriceSchema
from sentinel.data.sources import fetch
from sentinel.data.storage import write_table
from sentinel.data.universe import reconstruct_members_at
from sentinel.logging import get_logger

log = get_logger("rebuild")


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

    result = download_prices(
        tickers,
        lambda batch: fetch.fetch_yfinance_batch(batch, start, exclusive_end(end)),
        batch_size=batch_size,
    )
    if not result.frames or result.batches_failed == result.batches_total:
        raise RuntimeError(
            f"all {result.batches_total} price batches failed or returned nothing. "
            "Check connectivity with sentinel data doctor and the log above."
        )
    raw = pd.concat(result.frames, ignore_index=True)
    clean, rejected, qstats = quarantine_prices(raw)
    log.info("prices_quarantined", **qstats)
    prices = PriceSchema.validate(clean)
    empty = result.without_prices

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
    log.info(
        "rebuild_summary",
        rows=len(prices),
        tickers_requested=len(tickers),
        tickers_without_prices=len(empty),
        batches_total=result.batches_total,
        batches_failed=result.batches_failed,
        rows_rejected=qstats["rows_rejected"],
        cells_blanked=qstats["cells_blanked"],
    )

    if with_factors:
        for code in fetch.FRENCH_FILES:
            try:
                write_table(fetch.fetch_french(code), curated_dir / f"{code}.parquet", manifest)
            except Exception as exc:  # noqa: BLE001
                log.error("french_failed", code=code, error=str(exc), error_type=type(exc).__name__)
