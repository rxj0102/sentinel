"""Common portfolio constraints, shared by mean-variance and the projection step."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cvxpy as cp
import numpy as np


@dataclass(frozen=True)
class ConstraintSet:
    max_gross: float
    max_position: float
    exposure_tol: float
    turnover_cap_one_way: float
    dollar_neutral: bool = True


@dataclass(frozen=True)
class SolveResult:
    weights: np.ndarray
    feasible: bool
    status: str


def constraint_list(
    w: Any,
    cons: ConstraintSet,
    exposures: np.ndarray | None,
    w_prev: np.ndarray,
    forced_close_abs: float,
) -> list[Any]:
    """cvxpy constraints. w_prev is the previous weight on the same names as w, and
    forced_close_abs is the absolute previous weight in names that are no longer tradable and
    must be closed, which also consumes turnover."""
    cs: list[Any] = [cp.norm1(w) <= cons.max_gross, cp.abs(w) <= cons.max_position]
    if cons.dollar_neutral:
        cs.append(cp.sum(w) == 0)
    if exposures is not None and exposures.shape[1] > 0:
        cs.append(cp.abs(exposures.T @ w) <= cons.exposure_tol)
    cs.append(0.5 * (cp.norm1(w - w_prev) + forced_close_abs) <= cons.turnover_cap_one_way)
    return cs


def hold(w_prev: np.ndarray, status: str) -> SolveResult:
    return SolveResult(w_prev.copy(), False, status)


def project(
    target: np.ndarray,
    cons: ConstraintSet,
    exposures: np.ndarray | None,
    w_prev: np.ndarray,
    forced_close_abs: float = 0.0,
) -> SolveResult:
    """Least change (L2) feasible weights nearest to the target.

    If the constraints are infeasible, the previous weights are held and the result is
    flagged, never silently relaxed.
    """
    w = cp.Variable(len(target))
    prob = cp.Problem(
        cp.Minimize(cp.sum_squares(w - target)),
        constraint_list(w, cons, exposures, w_prev, forced_close_abs),
    )
    try:
        prob.solve(solver=cp.CLARABEL)
    except cp.error.SolverError:
        return hold(w_prev, "solver_error")
    if prob.status not in ("optimal", "optimal_inaccurate") or w.value is None:
        return hold(w_prev, str(prob.status))
    return SolveResult(np.asarray(w.value, dtype=float), True, str(prob.status))
