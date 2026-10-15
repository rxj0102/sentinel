"""CI scan: no module outside the engine path may read data at or after the holdout start."""

from pathlib import Path

from typer.testing import CliRunner

from sentinel.audit import scan_source, scan_tree
from sentinel.cli import app

ROOT = Path(__file__).resolve().parents[2]


def test_codebase_is_clean() -> None:
    assert scan_tree(ROOT) == []


def test_cli_exit_code() -> None:
    assert CliRunner().invoke(app, ["audit-holdout", "--root", str(ROOT)]).exit_code == 0


def test_detects_reader_outside_allowlist() -> None:
    bad = "import pandas as pd\ndef f():\n    return pd.read_parquet('prices.parquet')\n"
    assert scan_source(bad, "signals/sneaky.py")
    assert not scan_source(bad, "data/storage.py")


def test_detects_unguarded_cli_function() -> None:
    bad = "import pandas as pd\ndef cmd():\n    return pd.read_parquet('x')\n"
    assert scan_source(bad, "cli.py")
    ok = (
        "import pandas as pd\ndef cmd(guard):\n    df = pd.read_parquet('x')\n"
        "    return guard.dev_only(df)\n"
    )
    assert scan_source(ok, "cli.py") == []


def test_detects_holdout_dates_and_unlock_elsewhere() -> None:
    assert scan_source("def f(cfg):\n    return cfg.holdout_start\n", "signals/peek.py")
    assert scan_source("def f(g):\n    return g.unlock(1)\n", "backtest/run.py")
    assert scan_source("def f(g):\n    return g.unlock(1)\n", "cli.py") == []
    assert scan_source("def f(holdout_end=None):\n    pass\n", "portfolio/x.py")


def test_scripts_folder_is_scanned(tmp_path: Path) -> None:
    (tmp_path / "src" / "sentinel").mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "peek.py").write_text("import pandas as pd\npd.read_csv('x')\n")
    assert scan_tree(tmp_path)
