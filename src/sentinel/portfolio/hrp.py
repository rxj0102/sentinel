"""Hierarchical risk parity (Lopez de Prado 2016)."""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform


def correlation(cov: np.ndarray) -> np.ndarray:
    sd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sd, sd)
    return np.asarray(np.clip(corr, -1.0, 1.0))


def quasi_diagonal_order(cov: np.ndarray) -> list[int]:
    """Leaf order of single linkage clustering on the distance sqrt((1 - rho) / 2)."""
    n = cov.shape[0]
    if n == 1:
        return [0]
    dist = np.sqrt(np.clip((1.0 - correlation(cov)) / 2.0, 0.0, None))
    np.fill_diagonal(dist, 0.0)
    link = linkage(squareform(dist, checks=False), method="single")
    return [int(i) for i in leaves_list(link)]


def _cluster_variance(cov: np.ndarray, idx: list[int]) -> float:
    sub = cov[np.ix_(idx, idx)]
    ivp = 1.0 / np.diag(sub)
    ivp = ivp / ivp.sum()
    return float(ivp @ sub @ ivp)


def hrp_weights(cov: np.ndarray) -> np.ndarray:
    """Long only weights summing to one by recursive bisection on cluster variance."""
    n = cov.shape[0]
    order = quasi_diagonal_order(cov)
    w = np.ones(n)
    clusters = [order]
    while clusters:
        nxt: list[list[int]] = []
        for c in clusters:
            if len(c) < 2:
                continue
            half = len(c) // 2
            left, right = c[:half], c[half:]
            v_l, v_r = _cluster_variance(cov, left), _cluster_variance(cov, right)
            alpha = 1.0 - v_l / (v_l + v_r)
            w[left] *= alpha
            w[right] *= 1.0 - alpha
            nxt += [left, right]
        clusters = nxt
    return np.asarray(w / w.sum())
