"""Static audit of holdout access. Fails CI if code outside the engine reads holdout data.

This hardens the audit trail beyond the access log. It cannot prove a negative, so it is a
set of structural rules that any new data reading path must satisfy:

R1. Only allowlisted modules may call data readers (read_parquet, read_csv, read_table and
    similar). The allowlist is the CLI (which must route reads through the holdout guard)
    and the storage helper.
R2. In the CLI, every function that calls a data reader must also call a guard method
    (dev_only, load, authorize or unlock) in the same function.
R3. The names holdout_start and holdout_end may appear only in config, the holdout guard and
    the CLI that builds the guard.
R4. Calls to .unlock(...) may appear only in the holdout guard and the CLI.
R5. The same rules apply to scripts outside src.
"""

from __future__ import annotations

import ast
from pathlib import Path

READERS = {"read_parquet", "read_csv", "read_table", "read_feather", "read_json", "read_pickle"}
GUARD_CALLS = {"dev_only", "load", "authorize", "unlock"}
HOLDOUT_NAMES = {"holdout_start", "holdout_end"}

# holdout.py reads only its own access log, not market data.
READER_ALLOWED = {"cli.py", "data/storage.py", "backtest/holdout.py"}
HOLDOUT_NAME_ALLOWED = {"config.py", "backtest/holdout.py", "cli.py"}
UNLOCK_ALLOWED = {"backtest/holdout.py", "cli.py"}


def _call_name(node: ast.Call) -> str | None:
    f = node.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return None


def scan_source(source: str, rel: str) -> list[str]:
    """Violations in one module. rel is the path relative to src/sentinel (or scripts/...)."""
    tree = ast.parse(source)
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if name in READERS and rel not in READER_ALLOWED:
                out.append(f"{rel}:{node.lineno}: data reader {name} outside the allowlist")
            if name == "unlock" and rel not in UNLOCK_ALLOWED:
                out.append(f"{rel}:{node.lineno}: unlock called outside the holdout guard and CLI")
        names = set()
        if isinstance(node, ast.Attribute):
            names.add(node.attr)
        if isinstance(node, ast.Name):
            names.add(node.id)
        if isinstance(node, ast.keyword) and node.arg:
            names.add(node.arg)
        if isinstance(node, ast.arg):
            names.add(node.arg)
        if names & HOLDOUT_NAMES and rel not in HOLDOUT_NAME_ALLOWED:
            line = getattr(node, "lineno", 0)
            out.append(f"{rel}:{line}: holdout dates referenced outside allowed modules")
    if rel == "cli.py":
        for fn in ast.walk(tree):
            if isinstance(fn, ast.FunctionDef):
                calls = {
                    c for n in ast.walk(fn) if isinstance(n, ast.Call) and (c := _call_name(n))
                }
                if calls & READERS and not calls & GUARD_CALLS:
                    out.append(
                        f"cli.py:{fn.lineno}: {fn.name} reads data without the holdout guard"
                    )
    return out


def scan_tree(root: Path) -> list[str]:
    """Scan src/sentinel and scripts under root."""
    violations: list[str] = []
    src = root / "src" / "sentinel"
    for f in sorted(src.rglob("*.py")):
        violations += scan_source(f.read_text(), f.relative_to(src).as_posix())
    scripts = root / "scripts"
    if scripts.exists():
        for f in sorted(scripts.rglob("*.py")):
            violations += scan_source(f.read_text(), f"scripts/{f.relative_to(scripts).as_posix()}")
    return violations
