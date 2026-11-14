"""Orthogonalization of signals and of strategy returns against standard factors."""

from __future__ import annotations

import numpy as np
import pandas as pd


def residualize(signal: pd.DataFrame, exposures: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Cross-sectional OLS residual of the signal on exposures, date by date.

    Residuals are orthogonal to each exposure and to a constant within every date.
    Dates with too few complete rows are left as NaN.
    """
    names = list(exposures)
    out = pd.DataFrame(np.nan, index=signal.index, columns=signal.columns)
    for t in signal.index:
        y = signal.loc[t]
        x = pd.concat([exposures[k].loc[t].rename(k) for k in names], axis=1)
        ok = y.notna() & x.notna().all(axis=1)
        if ok.sum() < len(names) + 3:
            continue
        xm = np.column_stack([np.ones(int(ok.sum())), x[ok].to_numpy()])
        beta, *_ = np.linalg.lstsq(xm, y[ok].to_numpy(), rcond=None)
        out.loc[t, ok[ok].index] = y[ok].to_numpy() - xm @ beta
    return out


def factor_regression(ret: pd.Series, factors: pd.DataFrame) -> pd.DataFrame:
    """OLS of a return series on factor returns with classical standard errors.

    Returns a table with alpha (the intercept) and one row per factor: coefficient,
    standard error, t-stat. Used to ask whether a signal is just a repackaged factor.
    """
    df = pd.concat([ret.rename("y"), factors], axis=1).dropna()
    y = df["y"].to_numpy()
    x = np.column_stack([np.ones(len(df)), df[factors.columns].to_numpy()])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    dof = len(y) - x.shape[1]
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(x.T @ x)
    se = np.sqrt(np.diag(cov))
    names = ["alpha", *factors.columns]
    return pd.DataFrame({"term": names, "coef": beta, "se": se, "t_stat": beta / se})
