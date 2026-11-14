import json
from pathlib import Path

from typer.testing import CliRunner

from sentinel.cli import app
from sentinel.logging import configure_logging, get_logger

runner = CliRunner()


def test_fixture_then_quality_end_to_end(tmp_path: Path) -> None:
    fx, out = tmp_path / "fx", tmp_path / "q"
    r = runner.invoke(app, ["data", "make-fixture", "--out", str(fx), "--periods", "400"])
    assert r.exit_code == 0, r.output
    r = runner.invoke(app, ["data", "quality", "--fixture", str(fx), "--out", str(out)])
    assert r.exit_code == 0, r.output
    syn = out / "synthetic"
    assert (syn / "data_quality_coverage_by_year.csv").exists()
    assert (syn / "data_quality_survivorship_sensitivity.csv").exists()
    assert (syn / "data_quality_member_count_reconciliation.csv").exists()
    assert all(
        v == 0 for v in json.loads((syn / "data_quality_integrity.json").read_text()).values()
    )
    report = (syn / "data_quality_report.md").read_text()
    assert report.startswith("SYNTHETIC DATA")
    assert "Neither scenario set below is an estimate" in report
    assert "upper_bound_survivorship_damage" in report and "milder_case" in report
    meta = json.loads((syn / "data_quality_report.meta.json").read_text())
    assert meta["synthetic"] is True


def test_fixture_is_reproducible(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        runner.invoke(app, ["data", "make-fixture", "--out", str(d), "--periods", "300"])
    assert (a / "manifest.json").read_text() == (b / "manifest.json").read_text()


def test_logging_smoke() -> None:
    configure_logging()
    get_logger("t").info("hello", k=1)
