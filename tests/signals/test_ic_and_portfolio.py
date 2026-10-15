import numpy as np
import pandas as pd
import pytest

from sentinel.signals import ic as icmod
from sentinel.signals import orthogonalize as orth
from sentinel.signals import portfolio_stats as ps

RNG = np.random.default_rng(7)


def _plant(
    n_dates: int = 500, n_names: int = 120, effect: float = 0.004, seed: int = 1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Signal z_t. Return on day t+2 loads on z_t, which matches delay 1 horizon 1."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n_dates)
    cols = [f"T{i}" for i in range(n_names)]
    z = pd.DataFrame(rng.normal(size=(n_dates, n_names)), index=idx, columns=cols)
    ret = pd.DataFrame(rng.normal(0, 0.02, (n_dates, n_names)), index=idx, columns=cols)
    ret = ret + effect * z.shift(2).fillna(0.0)
    close = 100 * (1 + ret).cumprod()
    return z, close


def test_rank_ic_perfect_and_reversed() -> None:
    sig = pd.DataFrame(np.tile(np.arange(20.0), (3, 1)))
    fwd = sig * 0.01
    assert icmod.rank_ic(sig, fwd).tolist() == pytest.approx([1.0] * 3)
    assert icmod.rank_ic(sig, -fwd).tolist() == pytest.approx([-1.0] * 3)


def test_planted_signal_is_recovered_and_placebo_is_not() -> None:
    z, close = _plant()
    tab = icmod.decay_table(z, close, expected_sign=1, horizons=(1,), delay=1)
    row = tab.iloc[0]
    assert row["mean_ic"] > 0.05 and row["t_stat_nw"] > 5
    placebo = pd.DataFrame(
        np.random.default_rng(99).normal(size=z.shape), index=z.index, columns=z.columns
    )
    prow = icmod.decay_table(placebo, close, expected_sign=1, horizons=(1,), delay=1).iloc[0]
    assert abs(prow["t_stat_nw"]) < 3  # placebo shows no signal


def test_signal_that_peeks_one_day_early_is_visible_in_decay() -> None:
    z, close = _plant()
    early = z.shift(-1)  # uses information one day too early
    on_time = icmod.decay_table(z, close, 1, (1,), delay=1).iloc[0]["mean_ic"]
    peek = icmod.decay_table(early, close, 1, (1,), delay=1).iloc[0]["mean_ic"]
    assert peek < on_time  # the planted effect sits at the proper lag, an early shift misses it


def test_newey_west_matches_classical_without_lags_and_shrinks_with_autocorr() -> None:
    x = pd.Series(np.random.default_rng(3).normal(0.1, 1, 400))
    classical = x.mean() / (x.std(ddof=0) / np.sqrt(len(x)))
    assert icmod.newey_west_tstat(x, 0) == pytest.approx(classical)
    ar = pd.Series(np.zeros(2000))
    e = np.random.default_rng(4).normal(size=2000)
    for t in range(1, 2000):
        ar.iloc[t] = 0.7 * ar.iloc[t - 1] + e[t] + 0.05
    assert abs(icmod.newey_west_tstat(ar, 20)) < abs(icmod.newey_west_tstat(ar, 0))
    assert np.isnan(icmod.newey_west_tstat(pd.Series([1.0]), 3))


def test_residualize_is_orthogonal_to_exposures() -> None:
    idx = pd.bdate_range("2020-01-01", periods=5)
    cols = [f"T{i}" for i in range(60)]
    mk = lambda: pd.DataFrame(RNG.normal(size=(5, 60)), index=idx, columns=cols)  # noqa: E731
    beta, size = mk(), mk()
    sig = 3 * beta - 2 * size + 0.5 * mk()
    res = orth.residualize(sig, {"beta": beta, "size": size})
    for t in idx:
        assert abs(res.loc[t].mean()) < 1e-9
        assert abs((res.loc[t] * beta.loc[t]).sum()) < 1e-8
        assert abs((res.loc[t] * size.loc[t]).sum()) < 1e-8
    pure = orth.residualize(2 * beta + 1, {"beta": beta})
    assert pure.abs().max().max() < 1e-9


def test_residualize_skips_sparse_dates() -> None:
    idx = pd.bdate_range("2020-01-01", periods=2)
    sig = pd.DataFrame(np.nan, index=idx, columns=list("abcdefgh"))
    exp = {"x": pd.DataFrame(1.0, index=idx, columns=sig.columns)}
    assert orth.residualize(sig, exp).isna().all().all()


def test_factor_regression_closed_form() -> None:
    rng = np.random.default_rng(5)
    f = pd.DataFrame(rng.normal(size=(300, 2)), columns=["mkt", "smb"])
    exact = 0.001 + 0.5 * f["mkt"] - 0.3 * f["smb"]
    out = orth.factor_regression(exact, f).set_index("term")
    assert out.loc["alpha", "coef"] == pytest.approx(0.001)
    assert out.loc["mkt", "coef"] == pytest.approx(0.5)
    assert out.loc["smb", "coef"] == pytest.approx(-0.3)
    noisy = exact + rng.normal(0, 0.01, 300)
    got = orth.factor_regression(noisy, f).set_index("term")
    x = np.column_stack([np.ones(300), f.to_numpy()])
    beta = np.linalg.solve(x.T @ x, x.T @ noisy.to_numpy())
    resid = noisy.to_numpy() - x @ beta
    se = np.sqrt(np.diag(resid @ resid / (300 - 3) * np.linalg.inv(x.T @ x)))
    assert got["se"].to_numpy() == pytest.approx(se)
    assert got["t_stat"].to_numpy() == pytest.approx(beta / se)


def _sig() -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-01", periods=4)
    return pd.DataFrame(np.tile(np.arange(20.0), (4, 1)), index=idx)


def test_quantile_weights_neutral_and_sign() -> None:
    w = ps.quantile_weights(_sig(), 0.2)
    assert w.sum(axis=1).abs().max() < 1e-12
    assert (w.clip(lower=0).sum(axis=1) - 1).abs().max() < 1e-12
    assert w.iloc[0, 19] > 0 and w.iloc[0, 0] < 0
    flipped = ps.quantile_weights(_sig(), 0.2, sign=-1)
    assert flipped.iloc[0, 19] < 0 and flipped.iloc[0, 0] > 0
    few = ps.quantile_weights(_sig().iloc[:, :5], 0.2)
    assert (few == 0).all().all()


def test_turnover_known_answers() -> None:
    w = ps.quantile_weights(_sig(), 0.2)
    assert ps.turnover(w).abs().max() < 1e-12  # static weights, no trading
    a = pd.DataFrame({"x": [1.0, -1.0], "y": [-1.0, 1.0]})
    assert ps.turnover(a).iloc[0] == pytest.approx(2.0)  # full flip, 0.5 * (2 + 2)


def test_gross_return_uses_lagged_weights() -> None:
    idx = pd.bdate_range("2020-01-01", periods=3)
    w = pd.DataFrame({"x": [1.0, 0.0, 0.0], "y": [-1.0, 0.0, 0.0]}, index=idx)
    r = pd.DataFrame({"x": [0.0, 0.10, 0.5], "y": [0.0, 0.02, 0.5]}, index=idx)
    out = ps.gross_ls_return(w, r, delay=1)
    assert out.iloc[0] == pytest.approx(0.08)  # day-1 returns earned by day-0 weights
    assert out.iloc[1] == pytest.approx(0.0)


def test_capacity_known_answer() -> None:
    idx = pd.bdate_range("2020-01-01", periods=40)
    dv = pd.DataFrame(1_000_000.0, index=idx, columns=["x", "y"])
    w = pd.DataFrame({"x": 0.5, "y": -0.25}, index=idx)
    cap = ps.capacity(w, dv, participation=0.1, pct=0.0)
    assert cap == pytest.approx(0.1 * 1_000_000 / 0.5)  # binding name is x
