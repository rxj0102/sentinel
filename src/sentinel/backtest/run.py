"""Backtest orchestration over the pre-registered signal set. No parameter is tuned here.

Every signal runs with the parameters in docs/hypotheses.md, quantile and rebalance frequency
from the config, and the sign locked for the 21 day primary horizon.
"""

from __future__ import annotations

import inspect

import pandas as pd

from sentinel.backtest import stats
from sentinel.backtest.engine import run_backtest, sensitivity_grid, summarize
from sentinel.backtest.holdout import HoldoutGrant, HoldoutGuard
from sentinel.backtest.trials import TrialRegistry, TrialSpec
from sentinel.backtest.validation import fold_sharpes, purged_kfold_splits, walk_forward_splits
from sentinel.config import BacktestConfig
from sentinel.data.universe import mask_to_members
from sentinel.signals.base import Panel
from sentinel.signals.library import REGISTRY
from sentinel.signals.portfolio_stats import quantile_weights, quantile_weights_long_only

PRIMARY_HORIZON = 21
WF_TRAIN, WF_TEST, KFOLDS, EMBARGO = 126, 63, 5, 5  # fixed, reporting only, never tuned


def run_all(
    panel: Panel,
    membership: pd.DataFrame,
    cfg: BacktestConfig,
    *,
    guard: HoldoutGuard,
    registry: TrialRegistry,
    unlock: HoldoutGrant | None = None,
) -> dict[str, pd.DataFrame]:
    summary_rows, sens, folds = [], [], []
    returns: dict[str, pd.Series] = {}
    for name, (fn, spec) in REGISTRY.items():
        sign = spec.sign_at(PRIMARY_HORIZON)
        params = {
            "signal": name,
            "sign": sign,
            "primary_horizon": PRIMARY_HORIZON,
            "signal_params": {
                k: v.default for k, v in inspect.signature(fn).parameters.items() if k != "panel"
            },
        }
        sig = mask_to_members(fn(panel), membership)
        variants = {
            "long_short": ("signal", quantile_weights(sig, cfg.quantile, sign)),
            "long_only": ("portfolio_variant", quantile_weights_long_only(sig, cfg.quantile, sign)),
        }
        for variant, (kind, w) in variants.items():
            res = run_backtest(
                w,
                panel,
                cfg,
                guard=guard,
                registry=registry,
                trial=TrialSpec(kind, f"{name}|{variant}", params),
                unlock=unlock,
            )
            summary_rows.append({"signal": name, "variant": variant, **summarize(res)})
            returns[f"{name}|{variant}"] = res.net
            if variant == "long_short":
                sens.append(
                    sensitivity_grid(
                        w,
                        panel,
                        cfg,
                        guard=guard,
                        registry=registry,
                        name=f"{name}|{variant}",
                        unlock=unlock,
                    )
                )
                span = cfg.delay + PRIMARY_HORIZON
                n = len(res.net)
                if n >= WF_TRAIN + WF_TEST + span + EMBARGO:
                    for scheme, it in (
                        ("walk_forward", walk_forward_splits(n, WF_TRAIN, WF_TEST, span, EMBARGO)),
                        ("purged_kfold", purged_kfold_splits(n, KFOLDS, span, EMBARGO)),
                    ):
                        f = fold_sharpes(res.net, it)
                        f.insert(0, "scheme", scheme)
                        f.insert(0, "signal", name)
                        folds.append(f)
    dsr_rows = []
    for key, net in returns.items():
        d = stats.deflated_sharpe(net, registry)
        dsr_rows.append({"trial": key, "p_value": stats.sharpe_pvalue(net), **d})
    dsr = pd.DataFrame(dsr_rows)
    dsr["p_holm"] = stats.holm_adjust(dsr["p_value"].tolist())
    dsr["p_bh"] = stats.benjamini_hochberg(dsr["p_value"].tolist())
    counts = pd.DataFrame(
        [
            {
                "n_trials_counted": registry.n_trials,
                "n_evaluations_including_reruns_and_sensitivity": registry.n_evaluations,
                "n_reruns": registry.n_reruns,
                "n_sensitivity_cells": registry.n_sensitivity_cells,
            }
        ]
    )
    return {
        "backtest_summary": pd.DataFrame(summary_rows),
        "backtest_sensitivity": pd.concat(sens, ignore_index=True),
        "backtest_fold_sharpes": pd.concat(folds, ignore_index=True) if folds else pd.DataFrame(),
        "backtest_deflated_sharpe": dsr,
        "backtest_trial_counts": counts,
    }
