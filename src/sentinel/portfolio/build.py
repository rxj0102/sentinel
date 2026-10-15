"""Build rebalance-date weights for the three pre-registered constructions.

All three share the composite alpha, the point in time universe, the Ledoit-Wolf covariance
and the same constraints, so the comparison isolates the weighting rule. Settings come from
PortfolioConfig (docs/hypotheses.md, Amendment 1). Nothing is tuned here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sentinel.config import RunConfig
from sentinel.portfolio.constraints import ConstraintSet, project
from sentinel.portfolio.covariance import ledoit_wolf, risk_contributions
from sentinel.portfolio.hrp import hrp_weights
from sentinel.portfolio.mean_variance import mean_variance
from sentinel.portfolio.risk_parity import erc_weights
from sentinel.signals.base import Panel

METHODS = ("mv_lw", "risk_parity", "hrp")
MONTH = 21


@dataclass(frozen=True)
class BuildResult:
    weights: pd.DataFrame  # zeros except on rebalance rows
    diagnostics: pd.DataFrame


def _z(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return np.asarray((x - x.mean()) / sd) if sd > 0 else np.zeros_like(x)


def constraint_set(run: RunConfig) -> ConstraintSet:
    return ConstraintSet(
        max_gross=run.backtest.max_gross,
        max_position=run.backtest.max_position,
        exposure_tol=run.portfolio.exposure_tol,
        turnover_cap_one_way=run.portfolio.turnover_cap_one_way,
    )


def exposures_matrix(
    panel: Panel, i: int, names: list[str], window_returns: np.ndarray
) -> np.ndarray | None:
    """Standardized exposures at row i: log dollar volume, market beta, 12 minus 1 momentum."""
    adv = panel.dollar_volume.iloc[max(0, i - 62) : i + 1][names].mean().to_numpy()
    mom = (
        panel.close.iloc[i - 21][names].to_numpy() / panel.close.iloc[i - 252][names].to_numpy() - 1
    )
    mkt = window_returns.mean(axis=1)
    var_m = mkt.var(ddof=1)
    if not (np.isfinite(adv).all() and np.isfinite(mom).all()) or (adv <= 0).any() or var_m <= 0:
        return None
    centered = window_returns - window_returns.mean(axis=0)
    beta = (centered * (mkt - mkt.mean())[:, None]).sum(axis=0) / ((len(mkt) - 1) * var_m)
    return np.column_stack([_z(np.log(adv)), _z(beta), _z(mom)])


def build_weights(
    panel: Panel, score: pd.DataFrame, membership: pd.DataFrame, run: RunConfig, method: str
) -> BuildResult:
    if method not in METHODS:
        raise ValueError(f"unknown method {method}, expected one of {METHODS}")
    pc, every = run.portfolio, run.backtest.rebalance_every
    cons = constraint_set(run)
    cols = list(panel.close.columns)
    n = len(panel.close)
    ret = panel.ret
    member = membership.reindex(index=panel.close.index, columns=cols).fillna(False)
    weights = np.zeros((n, len(cols)))
    prev = np.zeros(len(cols))
    rows: list[dict[str, object]] = []
    for i in range(n):
        if i % every != 0 or i < max(pc.cov_window, 253):
            continue
        window = ret.iloc[i - pc.cov_window + 1 : i + 1]
        ok = (
            score.iloc[i].notna()
            & member.iloc[i]
            & window.notna().all()
            & panel.close.iloc[i].notna()
        ).to_numpy()
        idx = np.flatnonzero(ok)
        names = [cols[j] for j in idx]
        row: dict[str, object] = {
            "date": panel.close.index[i],
            "method": method,
            "n_universe": len(idx),
            "feasible": False,
            "status": "universe_too_small",
        }
        if len(idx) >= pc.min_names:
            rets = window.to_numpy()[:, idx]
            expo = exposures_matrix(panel, i, names, rets)
            leg = int(pc.leg_quantile * len(idx))
            if expo is not None and leg >= 2:
                cov_d, shrink = ledoit_wolf(rets)
                cov_m = MONTH * cov_d
                z = _z(score.iloc[i].to_numpy()[idx])
                w_prev = prev[idx]
                closing = float(np.abs(prev).sum() - np.abs(w_prev).sum())
                if method == "mv_lw":
                    mu = pc.ic_assumed * np.sqrt(np.diag(cov_m)) * z
                    res = mean_variance(mu, cov_m, cons, expo, w_prev, pc.risk_aversion, closing)
                else:
                    order = np.argsort(z)
                    short, long = order[:leg], order[-leg:]
                    rule = erc_weights if method == "risk_parity" else hrp_weights
                    target = np.zeros(len(idx))
                    target[long] = rule(cov_m[np.ix_(long, long)])
                    target[short] = -rule(cov_m[np.ix_(short, short)])
                    res = project(target, cons, expo, w_prev, closing)
                w = res.weights
                new = np.zeros(len(cols))
                new[idx] = w
                weights[i] = new
                rc = risk_contributions(w, cov_m)
                gross = float(np.abs(w).sum())
                row.update(
                    feasible=res.feasible,
                    status=res.status,
                    shrinkage=shrink,
                    gross=gross,
                    net=float(w.sum()),
                    one_way_turnover=0.5 * float(np.abs(new - prev).sum()),
                    effective_n=gross**2 / float((w**2).sum()) if gross > 0 else 0.0,
                    max_abs_exposure=float(np.abs(expo.T @ w).max()),
                    max_risk_share=float(np.nanmax(np.abs(rc))) if gross > 0 else 0.0,
                    names_at_cap=int((np.abs(w) >= cons.max_position - 1e-7).sum()),
                )
                prev = new
        if row["status"] == "universe_too_small":
            prev = np.zeros(len(cols))
        rows.append(row)
    out = pd.DataFrame(weights, index=panel.close.index, columns=cols)
    return BuildResult(out, pd.DataFrame(rows))
