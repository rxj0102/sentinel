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
from sentinel.data.universe import membership_from_meta, membership_matrix
from sentinel.logging import configure_logging, get_logger
from sentinel.results import write_result, write_text_result

app = typer.Typer(help="Sentinel: behavioral alpha and risk platform", no_args_is_help=True)
data_app = typer.Typer(help="Data layer commands", no_args_is_help=True)
app.add_typer(data_app, name="data")


@app.command("audit-holdout")
def audit_holdout_cmd(root: Path = Path(".")) -> None:
    """Fail if code outside the engine can read data at or after the holdout start."""
    from sentinel.audit import scan_tree

    problems = scan_tree(root)
    for p in problems:
        typer.echo(p)
    if problems:
        raise typer.Exit(code=1)
    typer.echo("holdout audit ok")


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
def quality_cmd(
    fixture: Path = Path("data/fixtures"),
    out: Path = Path("results"),
    config: Path = Path("configs/ci_fixture.yaml"),
) -> None:
    """Data quality and survivorship report. Reads the meta file to know if data is synthetic.

    Only the development window is read, the holdout is dropped through the guard.
    """
    from sentinel.backtest.holdout import HoldoutGuard
    from sentinel.config import load_run_config

    configure_logging()
    log = get_logger("quality")
    prices = pd.read_parquet(fixture / "prices.parquet")
    changes = pd.read_parquet(fixture / "changes.parquet")
    meta = json.loads((fixture / "meta.json").read_text())
    synthetic = bool(meta.get("synthetic", True))
    d = load_run_config(config).data
    guard = HoldoutGuard.for_run(
        pd.Timestamp(d.holdout_start),
        pd.Timestamp(d.holdout_end) if d.holdout_end else None,
        synthetic,
        results_root=out,
    )
    prices = guard.dev_only(prices)
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
    for key in (
        "coverage_by_year",
        "survivorship_sensitivity",
        "member_count_reconciliation",
        "ticker_reuse_flags",
    ):
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
    import traceback

    from sentinel.data.rebuild import rebuild

    configure_logging(log_file=log_file)
    log = get_logger("rebuild")
    try:
        rebuild(start, end, limit=limit, dry_run=dry_run)
    except Exception as exc:  # noqa: BLE001 - the full traceback must reach the log file
        log.error(
            "rebuild_failed",
            error=str(exc),
            error_type=type(exc).__name__,
            traceback=traceback.format_exc(),
        )
        raise typer.Exit(code=1) from exc


signals_app = typer.Typer(help="Signal research commands", no_args_is_help=True)
app.add_typer(signals_app, name="signals")


@signals_app.command("research")
def signals_research(
    fixture: Path = Path("data/fixtures"),
    out: Path = Path("results"),
    config: Path = Path("configs/ci_fixture.yaml"),
) -> None:
    """IC, decay, turnover, capacity and orthogonalization for every signal."""
    from sentinel.signals.base import panel_from_prices
    from sentinel.signals.run import research

    configure_logging()
    log = get_logger("signals")
    from sentinel.backtest.holdout import HoldoutGuard
    from sentinel.config import load_run_config

    prices = pd.read_parquet(fixture / "prices.parquet")
    meta = json.loads((fixture / "meta.json").read_text())
    synthetic = bool(meta.get("synthetic", True))
    d = load_run_config(config).data
    guard = HoldoutGuard.for_run(
        pd.Timestamp(d.holdout_start),
        pd.Timestamp(d.holdout_end) if d.holdout_end else None,
        synthetic,
        results_root=out,
    )
    prices = guard.dev_only(prices)  # explicit development window, the holdout is never read
    cal = pd.DatetimeIndex(sorted(prices["date"].unique()))
    changes = guard.dev_only(pd.read_parquet(fixture / "changes.parquet"))
    membership = membership_from_meta(meta, changes, cal)
    for name, table in research(panel_from_prices(prices, cal), membership).items():
        write_result(out, name, table, synthetic)
    log.info("signals_written", out=str(out), synthetic=synthetic)


@signals_app.command("model-cards")
def signals_model_cards(out: Path = Path("docs/model_cards")) -> None:
    """Write one MODEL_CARD per registered signal. Results sections stay gated."""
    from sentinel.signals.library import REGISTRY
    from sentinel.signals.run import render_model_card

    out.mkdir(parents=True, exist_ok=True)
    for name, (_, spec) in REGISTRY.items():
        (out / f"{name}.md").write_text(render_model_card(spec))


backtest_app = typer.Typer(help="Backtest and validation commands", no_args_is_help=True)
app.add_typer(backtest_app, name="backtest")


@backtest_app.command("run")
def backtest_run(
    fixture: Path = Path("data/fixtures"),
    out: Path = Path("results"),
    config: Path = Path("configs/ci_fixture.yaml"),
    unlock_holdout: bool = False,
    unlock_reason: str = "",
) -> None:
    """Run every pre-registered signal through the engine. No parameter is tuned.

    The holdout is refused unless --unlock-holdout is given, a reason is stated, and the
    holdout end date is written in the config. Every access is logged.
    """
    from sentinel.backtest.holdout import HoldoutGuard
    from sentinel.backtest.run import run_all
    from sentinel.backtest.trials import TrialRegistry
    from sentinel.config import load_run_config
    from sentinel.portfolio.feasibility import (
        FeasibilityError,
        check_feasibility,
        enforcement,
        membership_counts,
    )
    from sentinel.signals.base import panel_from_prices

    configure_logging()
    log = get_logger("backtest")
    run = load_run_config(config)
    prices = pd.read_parquet(fixture / "prices.parquet")
    synthetic = bool(json.loads((fixture / "meta.json").read_text()).get("synthetic", True))
    d = run.data
    guard = HoldoutGuard.for_run(
        pd.Timestamp(d.holdout_start),
        pd.Timestamp(d.holdout_end) if d.holdout_end else None,
        synthetic,
        results_root=out,
    )
    grant = None
    if unlock_holdout:
        if d.holdout_end is not None:  # the locked end date bounds what can be read
            prices = prices[prices["date"] <= pd.Timestamp(d.holdout_end)]
        grant = guard.unlock(pd.Timestamp(prices["date"].max()), "cli backtest run", unlock_reason)
        prices = guard.load(prices, grant=grant, caller="cli backtest run")
    else:
        prices = guard.dev_only(prices)
    registry = TrialRegistry(
        (out / "synthetic" / "trial_log.csv") if synthetic else Path("validation/trial_log.csv")
    )
    cal = pd.DatetimeIndex(sorted(prices["date"].unique()))
    membership = membership_from_meta(
        json.loads((fixture / "meta.json").read_text()),
        guard.dev_only(pd.read_parquet(fixture / "changes.parquet")),
        cal,
    )
    enforce = enforcement(synthetic, run.portfolio.require_full_gross_feasibility)
    try:
        feas = check_feasibility(membership_counts(membership, run), run, enforce)
    except FeasibilityError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    log.info("feasibility", synthetic=synthetic, enforced=enforce, summary=feas.summary())
    tables = run_all(
        panel_from_prices(prices, cal),
        membership,
        run.backtest,
        guard=guard,
        registry=registry,
        unlock=grant,
    )
    for name, table in tables.items():
        write_result(out, name, table, synthetic)
    log.info(
        "backtest_written",
        synthetic=synthetic,
        trials=registry.n_trials,
        evaluations=registry.n_evaluations,
        reruns=registry.n_reruns,
    )


portfolio_app = typer.Typer(help="Portfolio construction commands", no_args_is_help=True)
app.add_typer(portfolio_app, name="portfolio")


@portfolio_app.command("run")
def portfolio_run(
    fixture: Path = Path("data/fixtures"),
    out: Path = Path("results"),
    config: Path = Path("configs/ci_fixture.yaml"),
) -> None:
    """Compare mean-variance, leg-wise risk parity and leg-wise HRP against the baseline.

    Settings come from the pre-registered portfolio config. The holdout is never read.
    Every portfolio variant is registered as a trial.
    """
    from sentinel.backtest.holdout import HoldoutGuard
    from sentinel.backtest.trials import TrialRegistry
    from sentinel.config import load_run_config
    from sentinel.portfolio.compare import run_portfolios
    from sentinel.portfolio.feasibility import (
        FeasibilityError,
        check_feasibility,
        eligible_counts,
        enforcement,
    )
    from sentinel.portfolio.report import render_portfolio_report
    from sentinel.signals.base import panel_from_prices
    from sentinel.signals.composite import composite_score

    configure_logging()
    log = get_logger("portfolio")
    run = load_run_config(config)
    prices = pd.read_parquet(fixture / "prices.parquet")
    meta = json.loads((fixture / "meta.json").read_text())
    synthetic = bool(meta.get("synthetic", True))
    d = run.data
    guard = HoldoutGuard.for_run(
        pd.Timestamp(d.holdout_start),
        pd.Timestamp(d.holdout_end) if d.holdout_end else None,
        synthetic,
        results_root=out,
    )
    prices = guard.dev_only(prices)
    changes = guard.dev_only(pd.read_parquet(fixture / "changes.parquet"))
    cal = pd.DatetimeIndex(sorted(prices["date"].unique()))
    registry = TrialRegistry(
        (out / "synthetic" / "trial_log.csv") if synthetic else Path("validation/trial_log.csv")
    )
    panel = panel_from_prices(prices, cal)
    membership = membership_from_meta(meta, changes, cal)
    score = composite_score(panel, membership, run.portfolio)
    enforce = enforcement(synthetic, run.portfolio.require_full_gross_feasibility)
    try:
        feas = check_feasibility(eligible_counts(panel, score, membership, run), run, enforce)
    except FeasibilityError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    log.info("feasibility", synthetic=synthetic, enforced=enforce, summary=feas.summary())
    tables = run_portfolios(panel, membership, run, guard=guard, registry=registry)
    for name, table in tables.items():
        write_result(out, name, table, synthetic)
    write_text_result(
        out, "portfolio_report", render_portfolio_report(tables, synthetic, feas, run), synthetic
    )
    log.info("portfolio_written", synthetic=synthetic, trials=registry.n_trials)
