"""Command line interface."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer

from sentinel.data import quality
from sentinel.data.report import render_report
from sentinel.data.storage import write_table
from sentinel.data.synthetic import make_market
from sentinel.data.universe import membership_matrix
from sentinel.logging import configure_logging, get_logger
from sentinel.results import write_result, write_text_result

app = typer.Typer(help="Sentinel: behavioral alpha and risk platform", no_args_is_help=True)
data_app = typer.Typer(help="Data layer commands", no_args_is_help=True)
app.add_typer(data_app, name="data")


@app.command("docs-check")
def docs_check_cmd(root: Path = Path(".")) -> None:
    """Fail if docs break the REAL-DATA GATE or style rules."""
    from sentinel.docs_check import check_docs

    errors = check_docs(root)
    for e in errors:
        typer.echo(e)
    if errors:
        raise typer.Exit(code=1)
    typer.echo("docs ok")


@data_app.command("make-fixture")
def make_fixture(out: Path = Path("data/fixtures"), seed: int = 0, periods: int = 750) -> None:
    """Write the deterministic synthetic market used by CI. Contains no real data."""
    configure_logging()
    log = get_logger("fixture")
    m = make_market(seed=seed, periods=periods)
    manifest = out / "manifest.json"
    write_table(m.prices, out / "prices.parquet", manifest)
    write_table(m.changes, out / "changes.parquet", manifest)
    (out / "meta.json").write_text(
        json.dumps(
            {
                "initial_members": sorted(m.initial_members),
                "initial_date": str(m.initial_date.date()),
                "synthetic": True,
                "expected_size": len(m.initial_members),
                "size_tolerance": 10,
                "seed": seed,
                "periods": periods,
            },
            indent=2,
        )
    )
    log.info("fixture_written", rows=len(m.prices), out=str(out))


@data_app.command("quality")
def quality_cmd(fixture: Path = Path("data/fixtures"), out: Path = Path("results")) -> None:
    """Data quality and survivorship report. Reads the meta file to know if data is synthetic."""
    configure_logging()
    log = get_logger("quality")
    prices = pd.read_parquet(fixture / "prices.parquet")
    changes = pd.read_parquet(fixture / "changes.parquet")
    meta = json.loads((fixture / "meta.json").read_text())
    synthetic = bool(meta.get("synthetic", True))
    cal = pd.DatetimeIndex(sorted(prices["date"].unique()))
    mem = membership_matrix(
        set(meta["initial_members"]), pd.Timestamp(meta["initial_date"]), changes, cal
    )
    rep = quality.quality_report(
        prices,
        mem,
        expected_size=int(meta.get("expected_size", 500)),
        size_tolerance=int(meta.get("size_tolerance", 10)),
    )
    base = Path("data_quality")
    for key in ("coverage_by_year", "survivorship_sensitivity", "member_count_reconciliation"):
        frame = rep[key]
        assert isinstance(frame, pd.DataFrame)
        write_result(out, f"{base}_{key}", frame, synthetic)
    write_result(out, f"{base}_integrity", dict(rep["integrity"]), synthetic)  # type: ignore[call-overload]
    write_text_result(out, f"{base}_report", render_report(rep, synthetic), synthetic)
    log.info("quality_written", out=str(out), synthetic=synthetic)


@data_app.command("doctor")
def doctor_cmd() -> None:
    """Check connectivity to every data host. UNTESTED AGAINST LIVE ENDPOINTS."""
    from sentinel.data.sources.fetch import doctor

    for name, status in doctor().items():
        typer.echo(f"{name}: {status}")


@data_app.command("rebuild")
def rebuild_cmd(
    start: str = "2005-01-03",
    end: str = "2022-12-30",
    limit: int | None = None,
    dry_run: bool = False,
    log_file: Path = Path("logs/rebuild.log"),
) -> None:
    """Rebuild real data locally. UNTESTED AGAINST LIVE ENDPOINTS. Send the log file back.

    Needs SENTINEL_USER_AGENT in the environment. The end date defaults to the end of the
    development window, so the holdout period is never downloaded by this command.
    """
    import logging

    from sentinel.data.rebuild import rebuild

    log_file.parent.mkdir(parents=True, exist_ok=True)
    configure_logging()
    logging.getLogger().addHandler(logging.FileHandler(log_file))
    rebuild(start, end, limit=limit, dry_run=dry_run)
