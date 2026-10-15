"""Structured logging setup. Lines are JSON, one per event, written to stdout and optionally
to a log file, so a run can be sent back for debugging."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TextIO

import structlog


class _Stdout:
    """Resolve sys.stdout at write time, so a stream swapped later (tests, CLI runners) is
    never written after it was closed."""

    def write(self, data: str) -> int:
        return sys.stdout.write(data)

    def flush(self) -> None:
        sys.stdout.flush()


class _Tee:
    """Write to several text streams."""

    def __init__(self, *streams: TextIO | _Stdout) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for s in self.streams:
            s.write(data)
        return len(data)

    def flush(self) -> None:
        for s in self.streams:
            s.flush()


def configure_logging(level: int = logging.INFO, log_file: Path | None = None) -> None:
    """Configure structlog. With log_file, structlog lines and standard library log records
    (for example from yfinance) are also appended to that file."""
    logging.basicConfig(level=level, format="%(message)s")
    stream: TextIO | _Tee | _Stdout = _Stdout()
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handle = log_file.open("a", buffering=1)
        stream = _Tee(_Stdout(), handle)
        logging.getLogger().addHandler(logging.StreamHandler(handle))
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.WriteLoggerFactory(file=stream),  # type: ignore[arg-type]
        cache_logger_on_first_use=False,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)  # type: ignore[no-any-return]
