import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from sentinel.cli import app
from sentinel.data.synthetic import make_market
from sentinel.signals.base import panel_from_prices
from sentinel.signals.library import REGISTRY
from sentinel.signals.run import render_model_card, research

runner = CliRunner()


def test_research_tables_shape() -> None:
    m = make_market(periods=400)
    panel = panel_from_prices(m.prices, m.calendar)
    all_in = pd.DataFrame(True, index=panel.close.index, columns=panel.close.columns)
    tables = research(panel, all_in, horizons=(1, 5))
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


def test_universe_mask_removes_non_members() -> None:
    from sentinel.data.universe import mask_to_members

    m = make_market(periods=300)
    panel = panel_from_prices(m.prices, m.calendar)
    sig = panel.close.copy()
    member = pd.DataFrame(True, index=sig.index, columns=sig.columns)
    member.iloc[:, 0] = False
    out = mask_to_members(sig, member)
    assert out.iloc[:, 0].isna().all() and out.iloc[:, 1:].notna().any().any()
    # Names added to the index later are excluded before they join.
    mem = pd.DataFrame(False, index=sig.index, columns=sig.columns)
    mem.iloc[100:, 1] = True
    assert mask_to_members(sig, mem).iloc[:100, 1].isna().all()
