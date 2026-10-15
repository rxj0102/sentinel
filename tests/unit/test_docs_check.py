from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from sentinel.cli import app
from sentinel.docs_check import check_docs
from sentinel.results import is_synthetic, write_result


def _root(tmp_path: Path, readme: str, closed: bool = False) -> Path:
    (tmp_path / "docs").mkdir(exist_ok=True)
    (tmp_path / "README.md").write_text(readme)
    (tmp_path / "docs" / "REAL_DATA_GATE.md").write_text(
        f"Status: {'CLOSED' if closed else 'OPEN'}\n"
    )
    return tmp_path


def test_clean_docs_pass(tmp_path: Path) -> None:
    assert check_docs(_root(tmp_path, "Plain text.\n")) == []


def test_style_rules_and_fences(tmp_path: Path) -> None:
    bad = _root(tmp_path, 'A "quoted" word and a dash — here\n')
    assert len(check_docs(bad)) == 1
    ok = _root(tmp_path, '```\nprint("fine in code")\n```\n')
    assert check_docs(ok) == []


def test_finding_blocked_while_gate_open(tmp_path: Path) -> None:
    root = _root(tmp_path, "Sharpe is high [FINDING src=results/x.csv]\n")
    write_result(root / "results", "x", pd.DataFrame({"a": [1]}), synthetic=False)
    errs = check_docs(root)
    assert any("gate is open" in e.lower() for e in errs)


def test_synthetic_cannot_be_a_finding_even_when_gate_closed(tmp_path: Path) -> None:
    root = _root(tmp_path, "Result [FINDING src=results/synthetic/x.csv]\n", closed=True)
    write_result(root / "results", "x", pd.DataFrame({"a": [1]}), synthetic=True)
    errs = check_docs(root)
    assert any("synthetic result cited as a finding" in e for e in errs)


def test_synthetic_mention_needs_tag(tmp_path: Path) -> None:
    assert check_docs(_root(tmp_path, "See results/synthetic/x.csv\n"))
    ok = _root(tmp_path, "See results/synthetic/x.csv [SYNTHETIC-ENGINEERING-ONLY]\n")
    assert check_docs(ok) == []


def test_real_finding_allowed_only_when_closed_and_marked_real(tmp_path: Path) -> None:
    root = _root(tmp_path, "Result [FINDING src=results/x.csv]\n", closed=True)
    assert any("missing or not marked real" in e for e in check_docs(root))
    write_result(root / "results", "x", pd.DataFrame({"a": [1]}), synthetic=False)
    assert check_docs(root) == []
    assert is_synthetic(root / "results" / "x.csv") is False


def test_cli_exit_codes(tmp_path: Path) -> None:
    runner = CliRunner()
    good, bad = tmp_path / "g", tmp_path / "b"
    good.mkdir()
    bad.mkdir()
    _root(good, "Plain.\n")
    _root(bad, "[FINDING src=results/x.csv]\n")
    assert runner.invoke(app, ["docs-check", "--root", str(good)]).exit_code == 0
    assert runner.invoke(app, ["docs-check", "--root", str(bad)]).exit_code == 1


def test_indented_code_is_exempt(tmp_path: Path) -> None:
    assert check_docs(_root(tmp_path, '    pip install ".[dev]"\n')) == []
