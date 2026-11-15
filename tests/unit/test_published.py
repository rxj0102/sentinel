"""Published portfolios track (Amendment 3): data cut, mapping, statistics and determinism."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sentinel import published, readme
from sentinel.config import load_run_config

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / published.EVIDENCE / "long_short_returns.csv"


@pytest.fixture(scope="module")
def frozen() -> pd.DataFrame:
    return pd.read_csv(SNAPSHOT, dtype={"month": str})


def test_snapshot_holds_no_holdout_month(frozen: pd.DataFrame) -> None:
    start = load_run_config(ROOT / "configs" / "default.yaml").data.holdout_start
    assert frozen["month"].max() < f"{start.year}-{start.month:02d}"
    assert frozen["month"].max() == "2022-12"


def test_snapshot_matches_its_manifest(frozen: pd.DataFrame) -> None:
    manifest = json.loads((ROOT / published.EVIDENCE / "manifest.json").read_text())
    assert manifest["sha256"] == hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest()
    assert manifest["rows"] == len(frozen) and manifest["release"] == published.RELEASE
    assert set(frozen["construction"]) == set(published.FILES)
    assert set(frozen["signal"]) == {a for a, _ in published.MAPPING.values()}


def test_mapping_signs_equal_the_locked_table() -> None:
    """The sign used here must be the 21 day sign locked in docs/hypotheses.md."""
    locked = {}
    for line in (ROOT / "docs" / "hypotheses.md").read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 7 and cells[5] in ("+", "-") and cells[0] != "signal":
            locked[cells[0]] = 1 if cells[5] == "+" else -1
    assert len(locked) == 6
    for name, (_, sign) in published.MAPPING.items():
        assert locked[name] == sign
    assert set(published.MAPPING) | set(published.UNMAPPED) == set(locked)


def test_cut_drops_the_whole_cutoff_month() -> None:
    df = pd.DataFrame({"date": pd.to_datetime(["2022-12-30", "2023-01-02", "2023-01-31"]), "x": 1})
    kept = published.keep_development_rows(df, "2023-01-03")
    assert list(kept["date"].dt.strftime("%Y-%m-%d")) == ["2022-12-30"]


def test_build_snapshot_keeps_only_mapped_long_short_rows(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        {
            "signalname": ["MaxRet", "MaxRet", "Other", "High52", "High52"],
            "port": ["LS", "01", "LS", "LS", "LS"],
            "date": ["2010-01-29", "2010-01-29", "2010-01-29", "2010-02-26", "2024-02-29"],
            "ret": [1.5, 9.0, 9.0, -0.5, 9.0],
        }
    )
    manifest = published.build_snapshot(tmp_path, {"quintiles_vw": raw}, "2023-01-03")
    out = pd.read_csv(tmp_path / published.EVIDENCE / "long_short_returns.csv")
    assert manifest["rows"] == 2 and manifest["last_month"] == "2010-02"
    assert sorted(out["ret_pct"]) == [-0.5, 1.5]


def test_newey_west_and_verdict_rules() -> None:
    x = np.random.default_rng(0).normal(0.01, 0.05, 300)
    plain = x.mean() / (x.std() / np.sqrt(len(x)))
    assert published.newey_west_t(x, 0) == pytest.approx(plain, rel=1e-9)
    assert published.verdict(0.01, 0.04, 0.08) == "confirmed"
    assert published.verdict(0.01, 0.06, 0.12) == "not confirmed"
    assert published.verdict(-0.01, 0.99, 0.04) == "contradicted"
    assert published.verdict(-0.01, 0.99, 0.06) == "not confirmed"
    d = published.describe(pd.Series(x))
    assert d["mean_annual"] == pytest.approx(12 * x.mean())
    assert d["p_two_sided"] == pytest.approx(2 * d["p_one_sided"], rel=1e-9)


def test_run_reproduces_the_committed_metrics(frozen: pd.DataFrame, tmp_path: Path) -> None:
    committed = json.loads((ROOT / published.EVIDENCE / "metrics.json").read_text())
    got = published.run(tmp_path, frozen, "2022-12-30", registry=None)
    again = json.loads((tmp_path / published.EVIDENCE / "metrics.json").read_text())
    assert again == committed
    assert got["window"] == {"start": "2005-01", "end": "2022-12", "nw_lags": 6, "alpha": 0.05}
    for row in got["signals"].values():
        assert row["n_months"] == 216
        assert row["p_one_sided_holm"] >= row["p_one_sided"]


def test_readme_matches_the_evidence() -> None:
    tables = {
        t: pd.read_csv(ROOT / readme.QUALITY / f"data_quality_{t}.csv")
        for t in readme.QUALITY_TABLES
    }
    assert readme.render(ROOT, tables, check=True)
