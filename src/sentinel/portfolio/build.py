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
from sentinel.portfolio.erc import erc_weights
from sentinel.portfolio.hierarchical import hierarchical_weights
from sentinel.portfolio.ic_estimate import estimate_at, per_date_ic, resolve_ic
from sentinel.portfolio.mean_variance import mean_variance
from sentinel.signals.base import Panel

METHODS = ("mv_lw", "leg_risk_parity", "leg_hrp")
SKIPPED = ("universe_too_small", "exposures_unavailable", "leg_too_small")
MONTH = 21


@dataclass(frozen=True)
class BuildResult:
    weights: pd.DataFrame  # zeros except on rebalance rows
    diagnostics: pd.DataFrame


def leg_size(n_names: int, quantile: float) -> int:
    """Names per leg. The small epsilon avoids float error such as 0.2 times 170."""
    return int(quantile * n_names + 1e-9)


def eligible_indices(
    panel: Panel, score: pd.DataFrame, member: pd.DataFrame, i: int, window: int
) -> np.ndarray:
    """Positions of names usable at row i: member, scored, priced, full return window."""
    win = panel.ret.iloc[i - window + 1 : i + 1]
    ok = score.iloc[i].notna() & member.iloc[i] & win.notna().all() & panel.close.iloc[i].notna()
    return np.flatnonzero(ok.to_numpy())


def rebalance_rows(n_rows: int, every: int, cov_window: int) -> list[int]:
    return [i for i in range(n_rows) if i % every == 0 and i >= max(cov_window, 253)]


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
    ic_by_pos = (
        per_date_ic(score, panel.close, pc.ic_horizon, run.backtest.delay)
        if pc.ic_source == "expanding_estimate"
        else None
    )
    for i in rebalance_rows(n, every, pc.cov_window):
        idx = eligible_indices(panel, score, member, i, pc.cov_window)
        names = [cols[j] for j in idx]
        row: dict[str, object] = {
            "date": panel.close.index[i],
            "method": method,
            "n_universe": len(idx),
            "feasible": False,
            "status": "universe_too_small",
            "ic_used": np.nan,
            "ic_source_used": "n/a",
            "ic_estimate": np.nan,
            "ic_std_error": np.nan,
            "ic_n_dates": 0,
        }
        if len(idx) >= pc.min_names:
            window = ret.iloc[i - pc.cov_window + 1 : i + 1]
            rets = window.to_numpy()[:, idx]
            expo = exposures_matrix(panel, i, names, rets)
            leg = leg_size(len(idx), pc.leg_quantile)
            if expo is None:
                row["status"] = "exposures_unavailable"
            elif leg < 2:
                row["status"] = "leg_too_small"
            else:
                cov_d, shrink = ledoit_wolf(rets)
                cov_m = MONTH * cov_d
                z = _z(score.iloc[i].to_numpy()[idx])
                w_prev = prev[idx]
                closing = float(np.abs(prev).sum() - np.abs(w_prev).sum())
                if method == "mv_lw":
                    ic, why = pc.ic_assumed, "assumed"
                    if ic_by_pos is not None:
                        est = estimate_at(ic_by_pos, i, pc.ic_horizon, run.backtest.delay)
                        ic, why = resolve_ic(est, pc.ic_assumed, pc.ic_min_dates)
                        row.update(
                            ic_estimate=est.estimate,
                            ic_std_error=est.std_error,
                            ic_n_dates=est.n_dates,
                        )
                    row.update(ic_used=ic, ic_source_used=why)
                    mu = ic * np.sqrt(np.diag(cov_m)) * z
                    res = mean_variance(mu, cov_m, cons, expo, w_prev, pc.risk_aversion, closing)
                else:
                    order = np.argsort(z)
                    short, long = order[:leg], order[-leg:]
                    rule = erc_weights if method == "leg_risk_parity" else hierarchical_weights
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
        if row["status"] in SKIPPED:
            prev = np.zeros(len(cols))
        rows.append(row)
    out = pd.DataFrame(weights, index=panel.close.index, columns=cols)
    return BuildResult(out, pd.DataFrame(rows))
