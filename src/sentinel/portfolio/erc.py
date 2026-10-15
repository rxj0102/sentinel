"""Equal risk contribution weights for one long only leg, used as leg-wise risk parity."""

from __future__ import annotations

import numpy as np


def erc_weights(cov: np.ndarray, tol: float = 1e-10, max_iter: int = 10_000) -> np.ndarray:
    """Weights summing to one with equal risk contributions.

    Cyclical coordinate descent on the strictly convex problem
    min 0.5 y' S y - (1/n) sum log y_i, whose solution normalized to sum one is the ERC
    portfolio (Griveau-Billard, Richard, Roncalli 2013). Requires a positive definite matrix.
    """
    n = cov.shape[0]
    if n == 1:
        return np.ones(1)
    c = 1.0 / n
    y = 1.0 / np.sqrt(np.diag(cov))
    y = y / y.sum()
    diag = np.diag(cov)
    for _ in range(max_iter):
        y_old = y.copy()
        for i in range(n):
            b = float(cov[i] @ y - diag[i] * y[i])
            y[i] = (-b + np.sqrt(b * b + 4.0 * diag[i] * c)) / (2.0 * diag[i])
        if float(np.max(np.abs(y - y_old))) < tol * max(1.0, float(np.max(np.abs(y)))):
            break
    return np.asarray(y / y.sum())
