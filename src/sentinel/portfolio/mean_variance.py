"""Constrained mean-variance optimization with a shrunk covariance."""

from __future__ import annotations

import cvxpy as cp
import numpy as np

from sentinel.portfolio.constraints import (
    ConstraintSet,
    SolveResult,
    constraint_list,
    hold,
)


def mean_variance(
    mu: np.ndarray,
    cov: np.ndarray,
    cons: ConstraintSet,
    exposures: np.ndarray | None,
    w_prev: np.ndarray,
    risk_aversion: float,
    forced_close_abs: float = 0.0,
) -> SolveResult:
    """Maximize mu'w - (risk_aversion / 2) w'cov w under the common constraints."""
    w = cp.Variable(len(mu))
    objective = cp.Maximize(mu @ w - 0.5 * risk_aversion * cp.quad_form(w, cp.psd_wrap(cov)))
    prob = cp.Problem(objective, constraint_list(w, cons, exposures, w_prev, forced_close_abs))
    try:
        prob.solve(solver=cp.CLARABEL)
    except cp.error.SolverError:
        return hold(w_prev, "solver_error")
    if prob.status not in ("optimal", "optimal_inaccurate") or w.value is None:
        return hold(w_prev, str(prob.status))
    return SolveResult(np.asarray(w.value, dtype=float), True, str(prob.status))


def unconstrained_solution(mu: np.ndarray, cov: np.ndarray, risk_aversion: float) -> np.ndarray:
    """Closed form optimum without constraints, used as a numerical reference."""
    return np.asarray(np.linalg.solve(cov, mu) / risk_aversion)
