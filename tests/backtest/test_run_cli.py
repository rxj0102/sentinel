import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from sentinel.cli import app

runner = CliRunner()
ROOT = Path(__file__).resolve().parents[2]


def _fixture(tmp_path: Path) -> Path:
    fx = tmp_path / "fx"
    r = runner.invoke(app, ["data", "make-fixture", "--out", str(fx), "--periods", "900"])
    assert r.exit_code == 0, r.output
    return fx


def test_backtest_cli_is_synthetic_labeled_counts_trials_and_flags_reruns(tmp_path: Path) -> None:
    fx, out = _fixture(tmp_path), tmp_path / "res"
    cfg = str(ROOT / "configs" / "ci_fixture.yaml")
    args = ["backtest", "run", "--fixture", str(fx), "--out", str(out), "--config", cfg]
    r = runner.invoke(app, args)
    assert r.exit_code == 0, r.output
    syn = out / "synthetic"
    meta = json.loads((syn / "backtest_summary.meta.json").read_text())
    assert meta["synthetic"] is True
    counts = pd.read_csv(syn / "backtest_trial_counts.csv").iloc[0]
    assert counts["n_trials_counted"] == 12  # six signals, long short and long only each
    assert counts["n_sensitivity_cells"] == 6 * 20 and counts["n_reruns"] == 0
    dsr = pd.read_csv(syn / "backtest_deflated_sharpe.csv")
    assert (dsr["n_trials"] == 12).all() and {"p_holm", "p_bh"} <= set(dsr.columns)
    sens = pd.read_csv(syn / "backtest_sensitivity.csv")
    assert set(sens["borrow_rate_assumed"]) == {0.0, 0.005, 0.01, 0.02, 0.05}
    # Second identical run: visible reruns, no new independent trials.
    assert runner.invoke(app, args).exit_code == 0
    again = pd.read_csv(syn / "backtest_trial_counts.csv").iloc[0]
    assert again["n_trials_counted"] == 12 and again["n_reruns"] > 0
    # The synthetic holdout was never read, so its log was never written.
    assert not (syn / "holdout_access_log.csv").exists()


def test_backtest_cli_refuses_holdout_without_reason_and_unlocks_once(tmp_path: Path) -> None:
    fx, out = _fixture(tmp_path), tmp_path / "res"
    cfg = str(ROOT / "configs" / "ci_fixture.yaml")
    base = ["backtest", "run", "--fixture", str(fx), "--out", str(out), "--config", cfg]
    no_reason = runner.invoke(app, [*base, "--unlock-holdout"])
    assert no_reason.exit_code != 0
    ok = runner.invoke(app, [*base, "--unlock-holdout", "--unlock-reason", "engineering test"])
    assert ok.exit_code == 0, ok.output
    again = runner.invoke(app, [*base, "--unlock-holdout", "--unlock-reason", "second try"])
    assert again.exit_code != 0
    log = pd.read_csv(out / "synthetic" / "holdout_access_log.csv")
    assert list(log["action"]).count("granted") == 1
    assert "denied_no_reason" in set(log["action"]) and "denied_repeat" in set(log["action"])


def test_committed_real_trial_log_is_empty() -> None:
    assert (ROOT / "validation" / "trial_log.csv").read_text().count("\n") == 1
