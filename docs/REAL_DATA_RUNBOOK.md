# Real data runbook: the five most likely failure points

Status: written without access to the live endpoints. The rebuild code is UNTESTED AGAINST LIVE ENDPOINTS. This list comes from what I know of how these services behave, ranked by how likely each is to bite on the first run.

How to read the lines below. Lines marked exact come from this repo and are written as the code emits them: one JSON object per line, with the fields first and then event, level and timestamp. Lines marked expected are messages from other libraries or services that I am predicting from memory, so treat the wording as unverified. Everything goes to the terminal and to logs/rebuild.log, including a full traceback if the run dies.

Send back the whole log file, not an excerpt. The order of lines matters.

## Before the first run

    sentinel data doctor

Expected output is one line per host. A line like fred: HTTP 400 is fine, it only means no API key was sent. The hosts that matter for the rebuild are wikipedia, yahoo and french.

## 1. User agent not set or not quoted (most likely)

Where: sentinel.data.sources.fetch.user_agent, called by doctor, the Wikipedia fetch and the Ken French fetch.

Exact, as a traceback ending:

    RuntimeError: Set SENTINEL_USER_AGENT to something like 'sentinel-research your-email@example.com'. Wikipedia and SEC EDGAR reject anonymous clients.

Why likely: the value has a space, so on a Mac it must be quoted. Unquoted, zsh exports only the first word and prints an error such as the following (expected, unverified).

    export: not valid in this context: your-email@example.com

Fix: export SENTINEL_USER_AGENT with the whole value in quotes, as in the README.

## 2. Wikipedia refuses the request or the tables moved

Where: fetch_wikipedia_sp500 and the two parsers. The universe, and so every later step, depends on it.

Exact when the request is refused, repeated three times and then fatal:

    {"url": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "attempt": 1, "error": "403 Client Error: Forbidden for url: https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "error_type": "HTTPError", "event": "fetch_retry", "level": "warning", "timestamp": "..."}

followed by this error with a traceback, and then a rebuild_failed line:

    RuntimeError: giving up on https://en.wikipedia.org/wiki/List_of_S%26P_500_companies

Exact when the page loads but a table is not recognized (the page layout changed):

    ValueError: no S and P 500 changes table found

or

    ValueError: no constituents table found

The 403 text above is expected wording, the two ValueError lines are exact. If it is a layout change, send me the saved page at data/raw/wikipedia_sp500.html. The fix lands in the offline parser and can be tested from that file.

## 3. Yahoo throttles, returns nothing, or changes its column layout

Where: fetch_yfinance_batch through download_prices. Most likely after several batches, because of rate limits.

Exact per batch when a call raises. The error text is whatever the library said, and the error type names the exception class:

    {"batch_head": ["A", "AAL", "AAP"], "attempt": 1, "error": "Too Many Requests. Rate limited. Try after a while.", "error_type": "YFRateLimitError", "event": "batch_retry", "level": "warning", "timestamp": "..."}

After the third attempt (exact):

    {"batch_head": ["A", "AAL", "AAP"], "requested": 40, "event": "batch_failed", "level": "error", "timestamp": "..."}

Exact when a call returns an empty table, and when it succeeds but returns fewer names than asked:

    {"batch_head": ["A", "AAL", "AAP"], "requested": 40, "event": "batch_empty", "level": "warning", "timestamp": "..."}
    {"rows": 9120, "requested": 40, "returned": 31, "missing": 9, "event": "batch_ok", "level": "info", "timestamp": "..."}

Fatal only if every batch fails (exact):

    RuntimeError: all N price batches failed or returned nothing. Check connectivity with sentinel data doctor and the log above.

A column layout change would show up as a batch_retry line with error_type KeyError and a message naming a column such as adj_close (expected, unverified). The fix goes in normalize_yfinance in the parsers module.

Always read the final summary line (exact fields):

    {"rows": ..., "tickers_requested": ..., "tickers_without_prices": ..., "batches_total": ..., "batches_failed": ..., "rows_rejected": ..., "cells_blanked": ..., "event": "rebuild_summary", ...}

If batches_failed is more than zero, rerun with a smaller batch size or after a wait, and send both logs. Names without prices are listed in data/curated/no_price_history.csv.

## 4. Real prices break the strict schema

Where: after the download, in quarantine_prices and then PriceSchema. Real Yahoo data is known to contain duplicate rows, zero prices, a high below a low, and sometimes integer volume.

Exact, always emitted once per rebuild:

    {"rows_in": ..., "rows_rejected": ..., "duplicates_dropped": ..., "cells_blanked": ..., "rows_out": ..., "event": "prices_quarantined", "level": "info", "timestamp": "..."}

A small share rejected is normal and the rows are in data/curated/rejected_rows.csv with a reason. A large share is a data quality flag worth sending to me. If the strict schema still fails after quarantine, the traceback ends with a pandera error (expected wording, unverified), for example a dtype complaint such as this:

    pandera.errors.SchemaError: expected series 'date' to have type datetime64[ns], got datetime64[us]

Types are coerced in the parser, but this is the exact thing I cannot rule out without the live data.

## 5. Ken French factor files do not parse

Where: fetch_french and parse_french_daily. Not needed by Phases 1 to 4, so this should not block anything.

Exact when the download or parse fails (the run continues):

    {"code": "FF5D", "error": "", "error_type": "StopIteration", "event": "french_failed", "level": "error", "timestamp": "..."}

An empty error with StopIteration means the header line starting with a comma was not found, so the file layout changed. A refused download shows fetch_retry lines first. If it fails, send me the downloaded zip, and the fix goes in the offline parser with a test built from a saved file.

## Expected surprises that are not failures

- tickers_without_prices in the rebuild_summary will be large, likely in the hundreds, because Yahoo has no delisted tickers. That is the survivorship problem, not a bug, and it is exactly what the survivorship report measures.
- After the rebuild, run the quality report. The monthly member count reconciliation may show many months outside tolerance because the public change log has gaps. That is gate condition 2 itself, so send the whole report.
- The ticker reuse section of the quality report may flag a number of names. Send it as it is.
- The portfolio and backtest commands will refuse a real run if the universe is too small for the position cap, and print the required and observed sizes. That is the feasibility check working.
