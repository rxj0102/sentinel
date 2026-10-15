"""Honest comparison of the pre-registered constructions.

Headline rule (docs/hypotheses.md, Amendment 1): the headline result uses the pre-registered
default cost cell only, which is the config borrow rate and a cost multiplier of one. Every
sensitivity cell is reported next to it. No function here selects a cell by performance.
"""

from __future__ import annotations

import pandas as pd

from sentinel.backtest import stats
from sentinel.backtest.engine import run_backtest, sensitivity_grid, summarize
from sentinel.backtest.holdout import HoldoutGrant, HoldoutGuard
from sentinel.backtest.trials import TrialRegistry, TrialSpec
from sentinel.config import RunConfig
from sentinel.portfolio.build import METHODS, SKIPPED, build_weights
from sentinel.signals.base import Panel
from sentinel.signals.composite import composite_score
from sentinel.signals.portfolio_stats import quantile_weights

HEADLINE_CELL = "pre_registered_default"


def adopt_cell_as_configuration(
    registry: TrialRegistry, name: str, borrow_rate: float, cost_multiplier: float
) -> str:
    """Record that a sensitivity cell was used to choose a configuration.

    The rule: if any cell is ever used to choose a configuration, the choice is a trial. Any
    code path that adopts a non default cell must call this, which registers a counted
    parameter set and increases the trial count used by the deflated Sharpe.
    """
    return registry.register(
        TrialSpec(
            "parameter_set",
            f"{name}|adopted_cell",
            {"borrow_rate": borrow_rate, "cost_multiplier": cost_multiplier, "selected": True},
        )
    )


def run_portfolios(
    panel: Panel,
    membership: pd.DataFrame,
    run: RunConfig,
    *,
    guard: HoldoutGuard,
    registry: TrialRegistry,
    unlock: HoldoutGrant | None = None,
) -> dict[str, pd.DataFrame]:
    score = composite_score(panel, membership, run.portfolio)
    variants: dict[str, tuple[str, pd.DataFrame]] = {
        "equal_weight_quantile": (
            "signal",
            quantile_weights(score, run.portfolio.leg_quantile, sign=1),
        )
    }
    diag = []
    for method in METHODS:
        built = build_weights(panel, score, membership, run, method)
        variants[method] = ("portfolio_variant", built.weights)
        diag.append(built.diagnostics)

    summary, sens, returns = [], [], {}
    for name, (kind, w) in variants.items():
        key = f"composite|{name}"
        params = {"construction": name, "portfolio": run.portfolio.model_dump()}
        res = run_backtest(
            w,
            panel,
            run.backtest,
            guard=guard,
            registry=registry,
            trial=TrialSpec(kind, key, params),
            unlock=unlock,
        )
        summary.append({"variant": name, "cell": HEADLINE_CELL, **summarize(res)})
        returns[key] = res.net
        sens.append(
            sensitivity_grid(
                w, panel, run.backtest, guard=guard, registry=registry, name=key, unlock=unlock
            )
        )
    dsr = pd.DataFrame(
        [
            {"trial": k, "p_value": stats.sharpe_pvalue(r), **stats.deflated_sharpe(r, registry)}
            for k, r in returns.items()
        ]
    )
    dsr["p_holm"] = stats.holm_adjust(dsr["p_value"].tolist())
    dsr["p_bh"] = stats.benjamini_hochberg(dsr["p_value"].tolist())

    all_diag = pd.concat(diag, ignore_index=True)
    all_diag["skipped"] = all_diag["status"].isin(SKIPPED)
    all_diag["attempted"] = ~all_diag["skipped"]
    all_diag["fallback"] = all_diag["attempted"] & ~all_diag["feasible"].astype(bool)
    agg = (
        all_diag.groupby("method")
        .agg(
            rebalances=("date", "count"),
            rebalances_attempted=("attempted", "sum"),
            held_weight_fallbacks=("fallback", "sum"),
            skipped_rebalances=("skipped", "sum"),
            mean_shrinkage=("shrinkage", "mean"),
            mean_gross=("gross", "mean"),
            mean_effective_n=("effective_n", "mean"),
            mean_one_way_turnover=("one_way_turnover", "mean"),
            mean_max_abs_exposure=("max_abs_exposure", "mean"),
            mean_max_risk_share=("max_risk_share", "mean"),
            mean_names_at_cap=("names_at_cap", "mean"),
        )
        .reset_index()
    )
    agg["fallback_share_of_attempted"] = agg["held_weight_fallbacks"] / agg[
        "rebalances_attempted"
    ].replace(0, float("nan"))
    comparison = (
        pd.DataFrame(summary)
        .merge(
            agg[["method", "rebalances_attempted", "held_weight_fallbacks", "skipped_rebalances"]],
            left_on="variant",
            right_on="method",
            how="left",
        )
        .drop(columns="method")
    )
    return {
        "portfolio_comparison": comparison,
        "portfolio_sensitivity": pd.concat(sens, ignore_index=True),
        "portfolio_diagnostics": agg,
        "portfolio_rebalances": all_diag,
        "portfolio_deflated_sharpe": dsr,
        "portfolio_trial_counts": pd.DataFrame(
            [
                {
                    "n_trials_counted": registry.n_trials,
                    "n_evaluations_including_reruns_and_sensitivity": registry.n_evaluations,
                    "n_reruns": registry.n_reruns,
                    "n_sensitivity_cells": registry.n_sensitivity_cells,
                }
            ]
        ),
    }
