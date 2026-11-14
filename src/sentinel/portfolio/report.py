"""Portfolio report. The header always carries the model description, the feasibility check,
the IC source and the infeasible-constraint event counts, so a result built on many held
weight fallbacks cannot look clean."""

from __future__ import annotations

import pandas as pd

from sentinel.config import RunConfig
from sentinel.data.report import SYNTHETIC_BANNER, _table
from sentinel.portfolio.compare import HEADLINE_CELL
from sentinel.portfolio.feasibility import FeasibilityReport

MODEL_DESCRIPTION = (
    "Constructions compared: mean-variance with Ledoit-Wolf shrinkage, leg-wise risk parity, "
    "leg-wise HRP, and an equal weight quantile baseline. Leg-wise risk parity and leg-wise HRP "
    "are NOT true long-short risk parity. Names are first chosen by the composite score, a long "
    "only rule is applied inside each leg separately, the legs are scaled to plus one and minus "
    "one, and the result passes through the common constraints. A true long-short version would "
    "equalize the risk contributions of one combined signed book."
)


def infeasible_header(diag: pd.DataFrame) -> list[str]:
    """Per construction counts of attempted, fallback and skipped rebalances, and warnings."""
    lines = [
        "## Infeasible constraint events",
        "",
        "A fallback means the constraints could not be met at a rebalance, so the previous "
        "weights were held. A skip means the inputs were missing. Counts are never hidden.",
        "",
    ]
    rows: list[dict[str, object]] = []
    for _, rec in diag.iterrows():
        rows.append(
            {
                "construction": rec["method"],
                "rebalances": int(rec["rebalances"]),
                "attempted": int(rec["rebalances_attempted"]),
                "held_weight_fallbacks": int(rec["held_weight_fallbacks"]),
                "skipped": int(rec["skipped_rebalances"]),
            }
        )
    lines += [_table(pd.DataFrame(rows)), ""]
    for r in rows:
        if r["held_weight_fallbacks"]:
            lines.append(
                f"WARNING: {r['construction']} fell back to held weights at "
                f"{r['held_weight_fallbacks']} of {r['attempted']} attempted rebalances. "
                "Read its results with that in mind."
            )
        if r["skipped"]:
            lines.append(
                f"NOTE: {r['construction']} skipped {r['skipped']} of {r['rebalances']} "
                "rebalances for lack of inputs."
            )
    return lines + [""]


def render_portfolio_report(
    tables: dict[str, pd.DataFrame], synthetic: bool, feas: FeasibilityReport, run: RunConfig
) -> str:
    pc = run.portfolio
    ic_line = (
        f"IC source: {pc.ic_source}. Assumed value {pc.ic_assumed} is an assumption, not an "
        "estimate."
    )
    diag = tables["portfolio_diagnostics"]
    rebal = tables["portfolio_rebalances"]
    mv = rebal[rebal["method"] == "mv_lw"]
    if pc.ic_source == "expanding_estimate" and len(mv):
        counts = mv["ic_source_used"].value_counts().to_dict()
        ic_line += f" Rebalance counts by IC outcome: {counts}."
    parts = [
        SYNTHETIC_BANNER if synthetic else "",
        "# Portfolio comparison report",
        "",
        MODEL_DESCRIPTION,
        "",
        "Nothing here is a finding while the REAL-DATA GATE is open.",
        "",
        ic_line,
        "",
        "## Universe feasibility check",
        "",
        feas.summary()
        + (
            " Mechanism test only, the synthetic fixture is too small for the cap."
            if synthetic and not feas.passes
            else ""
        ),
        f"Enforced: {feas.enforced}.",
        "",
        *infeasible_header(diag),
        "## Headline rule",
        "",
        f"Headline rows use the {HEADLINE_CELL} cost cell only. Every sensitivity cell is in "
        "portfolio_sensitivity. No cell is chosen as the headline.",
        "",
        "## Headline table (default cost cell)",
        "",
        _table(tables["portfolio_comparison"]),
        "",
        "## Construction diagnostics",
        "",
        _table(diag),
        "",
        "## Deflated Sharpe and multiple testing",
        "",
        _table(tables["portfolio_deflated_sharpe"]),
        "",
    ]
    return "\n".join(parts)
