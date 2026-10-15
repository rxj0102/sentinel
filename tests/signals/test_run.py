import json
from pathlib import Path

from typer.testing import CliRunner

from sentinel.cli import app
from sentinel.data.synthetic import make_market
from sentinel.signals.base import panel_from_prices
from sentinel.signals.library import REGISTRY
from sentinel.signals.run import render_model_card, research

runner = CliRunner()


def test_research_tables_shape() -> None:
    m = make_market(periods=400)
    tables = research(panel_from_prices(m.prices, m.calendar), horizons=(1, 5))
    dec = tables["signals_ic_decay"]
    assert set(dec["signal"]) == set(REGISTRY) and set(dec["horizon"]) == {1, 5}
    assert len(tables["signals_portfolio"]) == len(REGISTRY)
    assert len(tables["signals_ortho_ic"]) == len(REGISTRY)


def test_model_card_has_gated_results() -> None:
    card = render_model_card(REGISTRY["max_lottery"][1])
    assert "PENDING REAL-DATA GATE" in card and "negative" in card


def test_cli_research_labels_synthetic(tmp_path: Path) -> None:
    fx, out = tmp_path / "fx", tmp_path / "res"
    runner.invoke(app, ["data", "make-fixture", "--out", str(fx), "--periods", "400"])
    r = runner.invoke(app, ["signals", "research", "--fixture", str(fx), "--out", str(out)])
    assert r.exit_code == 0, r.output
    meta = json.loads((out / "synthetic" / "signals_ic_decay.meta.json").read_text())
    assert meta["synthetic"] is True
    cards = tmp_path / "cards"
    assert runner.invoke(app, ["signals", "model-cards", "--out", str(cards)]).exit_code == 0
    assert len(list(cards.glob("*.md"))) == len(REGISTRY)
