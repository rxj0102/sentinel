import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sentinel.portfolio.constraints import ConstraintSet, project
from sentinel.portfolio.covariance import ledoit_wolf, risk_contributions
from sentinel.portfolio.erc import erc_weights
from sentinel.portfolio.hierarchical import hierarchical_weights, quasi_diagonal_order
from sentinel.portfolio.mean_variance import mean_variance, unconstrained_solution


def _psd(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n + 3))
    return a @ a.T / (n + 3) + 0.05 * np.eye(n)


def _lw_reference(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Independent implementation of the Ledoit-Wolf (2004) formulas, shrinking to mu * I."""
    t, p = x.shape
    xc = x - x.mean(axis=0)
    s = xc.T @ xc / t
    mu = np.trace(s) / p
    d2 = np.sum((s - mu * np.eye(p)) ** 2) / p
    b2 = sum(np.sum((np.outer(r, r) - s) ** 2) for r in xc) / (t**2 * p)
    b2 = min(b2, d2)
    shrink = b2 / d2
    return shrink * mu * np.eye(p) + (1 - shrink) * s, shrink


def test_ledoit_wolf_matches_independent_formula() -> None:
    x = np.random.default_rng(0).standard_t(5, size=(120, 8)) @ _psd(8, 1)
    cov, shrink = ledoit_wolf(x)
    ref_cov, ref_shrink = _lw_reference(x)
    assert shrink == pytest.approx(ref_shrink, rel=1e-6)
    assert cov == pytest.approx(ref_cov, rel=1e-6)


def test_ledoit_wolf_properties() -> None:
    wide = np.random.default_rng(2).normal(size=(40, 120))  # more assets than observations
    sample = np.cov(wide.T)
    assert np.linalg.matrix_rank(sample) < 120  # the sample covariance is singular
    cov, shrink = ledoit_wolf(wide)
    assert np.linalg.eigvalsh(cov).min() > 0 and 0 <= shrink <= 1
    assert np.trace(cov) == pytest.approx(np.trace(np.cov(wide.T, bias=True)), rel=1e-9)
    assert np.linalg.cond(cov) < np.linalg.cond(sample + 1e-12 * np.eye(120))
    chol = np.linalg.cholesky(
        np.array([[1, 0.8, 0, 0], [0.8, 4, 0, 0], [0, 0, 0.25, 0.1], [0, 0, 0.1, 9]])
    )
    big = np.random.default_rng(3).normal(size=(20_000, 4)) @ chol.T
    assert ledoit_wolf(big)[1] < 0.01  # structured truth and plenty of data: almost no shrinkage
    iid = np.random.default_rng(3).normal(size=(500, 4))
    assert ledoit_wolf(iid)[1] > 0.5  # truth is the target itself, so shrink hard
    with pytest.raises(ValueError):
        ledoit_wolf(np.array([[1.0, np.nan], [2.0, 3.0], [3.0, 4.0]]))
    with pytest.raises(ValueError):
        ledoit_wolf(np.ones(5))


def test_erc_closed_forms() -> None:
    sd = np.array([0.1, 0.2, 0.4])
    w = erc_weights(np.diag(sd**2))
    assert w == pytest.approx((1 / sd) / (1 / sd).sum(), rel=1e-8)  # diagonal: inverse vol
    for rho in (-0.3, 0.0, 0.6):  # two assets: w1 / w2 = sd2 / sd1 whatever the correlation
        cov = np.array([[0.04, rho * 0.2 * 0.3], [rho * 0.2 * 0.3, 0.09]])
        w2 = erc_weights(cov)
        assert w2[0] / w2[1] == pytest.approx(0.3 / 0.2, rel=1e-7)
    eq = 0.04 * (0.5 * np.eye(5) + 0.5 * np.ones((5, 5)))
    assert erc_weights(eq) == pytest.approx(np.full(5, 0.2))
    assert erc_weights(np.array([[2.0]])) == pytest.approx([1.0])


@given(n=st.integers(2, 12), seed=st.integers(0, 10_000))
@settings(max_examples=40, deadline=None)
def test_erc_equalizes_risk_contributions(n: int, seed: int) -> None:
    cov = _psd(n, seed)
    w = erc_weights(cov)
    assert (w > 0).all() and w.sum() == pytest.approx(1.0)
    rc = risk_contributions(w, cov)
    assert rc.max() - rc.min() < 1e-6


def test_hierarchical_hand_computed_example() -> None:
    # A and B are highly correlated and form the first cluster, C is on its own.
    sd = np.ones(3)
    corr = np.array([[1, 0.9, 0.1], [0.9, 1, 0.1], [0.1, 0.1, 1.0]])
    cov = corr * np.outer(sd, sd)
    v_ab = 0.25 + 0.25 + 2 * 0.25 * 0.9  # variance of the inverse variance mix of A and B
    alpha = 1 - v_ab / (v_ab + 1.0)  # weight of the AB cluster
    w = hierarchical_weights(cov)
    assert w[0] == pytest.approx(alpha / 2) and w[1] == pytest.approx(alpha / 2)
    assert w[2] == pytest.approx(1 - alpha)
    assert w.sum() == pytest.approx(1.0)


def test_hierarchical_properties() -> None:
    cov = _psd(9, 4)
    w = hierarchical_weights(cov)
    assert (w > 0).all() and w.sum() == pytest.approx(1.0)
    assert sorted(quasi_diagonal_order(cov)) == list(range(9))
    perm = np.random.default_rng(5).permutation(9)
    assert hierarchical_weights(cov[np.ix_(perm, perm)]) == pytest.approx(w[perm], rel=1e-9)
    assert hierarchical_weights(np.array([[3.0]])) == pytest.approx([1.0])
    iid = 0.04 * np.eye(6)
    assert hierarchical_weights(iid) == pytest.approx(np.full(6, 1 / 6))


CONS = ConstraintSet(max_gross=2.0, max_position=0.1, exposure_tol=0.2, turnover_cap_one_way=0.5)


def _check(
    w: np.ndarray,
    cons: ConstraintSet,
    x: np.ndarray | None,
    w_prev: np.ndarray,
    closed: float = 0.0,
) -> None:
    tol = 1e-5
    assert abs(w.sum()) < tol
    assert np.abs(w).sum() <= cons.max_gross + tol
    assert np.abs(w).max() <= cons.max_position + tol
    if x is not None:
        assert np.abs(x.T @ w).max() <= cons.exposure_tol + tol
    assert 0.5 * (np.abs(w - w_prev).sum() + closed) <= cons.turnover_cap_one_way + tol


@given(seed=st.integers(0, 1000))
@settings(max_examples=15, deadline=None)
def test_projection_satisfies_every_constraint(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 30
    target = rng.normal(0, 0.2, n)
    x = rng.normal(size=(n, 3))
    w_prev = np.zeros(n)
    res = project(target, CONS, x, w_prev)
    assert res.feasible
    _check(res.weights, CONS, x, w_prev)


def test_projection_leaves_feasible_targets_unchanged_and_flags_infeasible() -> None:
    n = 20
    target = np.zeros(n)
    target[:3], target[3:6] = 0.05, -0.05
    same = project(target, CONS, None, target.copy())
    assert same.weights == pytest.approx(target, abs=1e-6)
    tight = ConstraintSet(2.0, 0.1, 0.2, turnover_cap_one_way=0.01)
    prev = np.zeros(n)
    res = project(target, tight, None, prev, forced_close_abs=5.0)  # closing alone breaks the cap
    assert not res.feasible and res.weights == pytest.approx(prev) and res.status != "optimal"


def test_mean_variance_matches_closed_form_without_constraints() -> None:
    n = 6
    cov = _psd(n, 7)
    mu = np.random.default_rng(8).normal(0, 0.02, n)
    loose = ConstraintSet(
        max_gross=1e6,
        max_position=1e6,
        exposure_tol=1e6,
        turnover_cap_one_way=1e6,
        dollar_neutral=False,
    )
    res = mean_variance(mu, cov, loose, None, np.zeros(n), risk_aversion=10.0)
    assert res.feasible
    assert res.weights == pytest.approx(unconstrained_solution(mu, cov, 10.0), rel=1e-3, abs=1e-5)


@given(seed=st.integers(0, 1000))
@settings(max_examples=15, deadline=None)
def test_mean_variance_satisfies_constraints_and_risk_aversion_lowers_variance(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 25
    cov = _psd(n, seed)
    mu = rng.normal(0, 0.03, n)
    x = rng.normal(size=(n, 3))
    w_prev = np.zeros(n)
    low = mean_variance(mu, cov, CONS, x, w_prev, risk_aversion=1.0)
    high = mean_variance(mu, cov, CONS, x, w_prev, risk_aversion=100.0)
    assert low.feasible and high.feasible
    _check(low.weights, CONS, x, w_prev)
    _check(high.weights, CONS, x, w_prev)
    var = lambda w: float(w @ cov @ w)  # noqa: E731
    assert var(high.weights) <= var(low.weights) + 1e-9


def test_mean_variance_with_zero_alpha_holds_nothing() -> None:
    n = 10
    res = mean_variance(np.zeros(n), _psd(n, 1), CONS, None, np.zeros(n), 10.0)
    assert np.abs(res.weights).max() < 1e-6
