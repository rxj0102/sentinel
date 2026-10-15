"""Render the data quality and survivorship report as Markdown."""

from __future__ import annotations

import pandas as pd

HEADER = """# Data quality and survivorship report

Neither scenario set below is an estimate. The milder case and the upper bound on survivorship
damage are assumptions chosen to bracket the problem, because prices of vanished index members
are not available from free sources. The delisting scenarios are conservative: they apply a
terminal return to every member whose history ends early, even when the last price already
contains the terminal move. Read the baseline row as biased upward.
"""

SYNTHETIC_BANNER = (
    "SYNTHETIC DATA. Engineering output only. Nothing here is a finding and it must not be "
    "cited as one.\n"
)


def _table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        cells = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_report(rep: dict[str, object], synthetic: bool) -> str:
    cov, sens, rec = (
        rep["coverage_by_year"],
        rep["survivorship_sensitivity"],
        rep["member_count_reconciliation"],
    )
    assert isinstance(cov, pd.DataFrame) and isinstance(sens, pd.DataFrame)
    assert isinstance(rec, pd.DataFrame)
    parts = [SYNTHETIC_BANNER if synthetic else "", HEADER]
    parts += ["## Integrity checks (zero is healthy)", ""]
    integ = rep["integrity"]
    assert isinstance(integ, dict)
    parts += [f"- {k}: {v}" for k, v in integ.items()] + [""]
    parts += ["## Missing price share of index member days, by year", "", _table(cov), ""]
    parts += ["## Survivorship sensitivity (annualised equal weight return)", ""]
    parts += [_table(sens), ""]
    bad = rec[~rec["ok"]]
    parts += ["## Monthly member count reconciliation", ""]
    parts += [f"Months checked: {len(rec)}. Months outside tolerance: {len(bad)}.", ""]
    if len(bad):
        parts += [_table(bad), ""]
    return "\n".join(parts)
