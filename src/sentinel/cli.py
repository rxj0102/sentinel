"""Command line interface."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer

from sentinel.data import quality
from sentinel.data.storage import write_table
from sentinel.data.synthetic import make_market
from sentinel.data.universe import membership_matrix
from sentinel.logging import configure_logging, get_logger

app = typer.Typer(help="Sentinel: behavioral alpha and risk platform", no_args_is_help=True)
data_app = typer.Typer(help="Data layer commands", no_args_is_help=True)
app.add_typer(data_app, name="data")


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
                "seed": seed,
                "periods": periods,
            },
            indent=2,
        )
    )
    log.info("fixture_written", rows=len(m.prices), out=str(out))


@data_app.command("quality")
def quality_cmd(
    fixture: Path = Path("data/fixtures"), out: Path = Path("results/data_quality")
) -> None:
    """Data quality and survivorship report. Writes CSV and JSON under out."""
    configure_logging()
    log = get_logger("quality")
    prices = pd.read_parquet(fixture / "prices.parquet")
    changes = pd.read_parquet(fixture / "changes.parquet")
    meta = json.loads((fixture / "meta.json").read_text())
    cal = pd.DatetimeIndex(sorted(prices["date"].unique()))
    mem = membership_matrix(
        set(meta["initial_members"]), pd.Timestamp(meta["initial_date"]), changes, cal
    )
    rep = quality.quality_report(prices, mem)
    out.mkdir(parents=True, exist_ok=True)
    cov = rep["coverage_by_year"]
    sens = rep["survivorship_sensitivity"]
    assert isinstance(cov, pd.DataFrame) and isinstance(sens, pd.DataFrame)
    cov.to_csv(out / "coverage_by_year.csv", index=False)
    sens.to_csv(out / "survivorship_sensitivity.csv", index=False)
    (out / "integrity.json").write_text(json.dumps(rep["integrity"], indent=2))
    log.info("quality_written", out=str(out))
