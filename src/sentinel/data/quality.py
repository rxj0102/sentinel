"""Data quality report, including the required survivorship measurements.

The survivorship measurement answers one question: of all (member, day) pairs the index
actually contained, what fraction have no price row in our data, by year. Missing pairs are
concentrated in delisted names, so a high fraction means returns are biased upward.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def wide_close(prices: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Adjusted close, calendar x ticker, NaN where no row exists."""
    wide = prices.pivot(index="date", columns="ticker", values="adj_close")
    return wide.reindex(calendar)


def coverage_by_year(membership: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """Per-year missing price share of point-in-time member days.

    Columns: year, member_days, missing_days, missing_frac, members_never_priced.
    """
    priced = close.reindex(columns=membership.columns).notna().reindex(membership.index)
    missing = membership & ~priced.fillna(False)
    never_priced = [c for c in membership.columns if not bool(priced[c].any())]
    mem = membership
    year = pd.DatetimeIndex(mem.index).year
    out = pd.DataFrame(
        {
            "member_days": mem.groupby(year).sum().sum(axis=1),
            "missing_days": missing.groupby(year).sum().sum(axis=1),
        }
    )
    out["missing_frac"] = out["missing_days"] / out["member_days"].replace(0, np.nan)
    out["members_never_priced"] = [
        sum(1 for c in never_priced if bool(mem[c][year == y].any())) for y in out.index
    ]
    out.index.name = "year"
    return out.reset_index()


def apply_delisting_returns(
    returns: pd.DataFrame, membership: pd.DataFrame, delist_return: float
) -> pd.DataFrame:
    """Insert a terminal return for names that vanish while still index members.

    For a ticker whose last valid price is on day L, which is before the end of the sample
    and on which the ticker was an index member, set the next day return to delist_return.
    This is conservative: it applies even when the last print already embeds the terminal
    move. Names with no price at all cannot be repaired (the last price is unknown) and
    remain missing.
    """
    out = returns.copy()
    idx = returns.index
    for t in returns.columns:
        valid = returns[t].dropna()
        if valid.empty:
            continue
        last = valid.index[-1]
        pos = int(idx.get_loc(last))  # type: ignore[arg-type]
        if pos + 1 < len(idx) and bool(membership[t].iloc[pos]):
            out.loc[idx[pos + 1], t] = delist_return
    return out


def ew_member_return(returns: pd.DataFrame, membership: pd.DataFrame) -> pd.Series:
    """Equal weight daily return of the universe held at the previous close.

    Membership is lagged one day because a position is chosen before the return accrues.
    Missing prices are ignored.
    """
    held = membership.shift(1, fill_value=False).reindex_like(returns)
    r = returns.where(held)
    return r.mean(axis=1)


@dataclass(frozen=True)
class SensitivityRow:
    scenario: str
    ann_return: float
    n_days: int
    scenario_set: str


def _daily_missing_frac(close: pd.DataFrame, membership: pd.DataFrame) -> pd.Series:
    priced = close.reindex(columns=membership.columns).notna().reindex(membership.index)
    missing = (membership & ~priced.fillna(False)).sum(axis=1)
    return missing / membership.sum(axis=1).replace(0, np.nan)


MILDER = "milder_case"
UPPER = "upper_bound_survivorship_damage"


def _set_name(value: float, values: tuple[float, ...], higher_is_milder: bool) -> str:
    """Name the harshest scenario of a family the upper bound, the gentlest the milder case."""
    milder = max(values) if higher_is_milder else min(values)
    return MILDER if value == milder else UPPER


def survivorship_sensitivity(
    close: pd.DataFrame,
    membership: pd.DataFrame,
    coverage: pd.DataFrame,
    delist_returns: tuple[float, ...] = (-0.3, -1.0),
    max_missing_frac: float = 0.05,
    imputation_penalties: tuple[float, ...] = (0.05, 0.15),
) -> pd.DataFrame:
    """Annualised equal weight member return under alternative survivorship treatments.

    Two scenario sets, both assumptions and neither an estimate: the milder case and the
    upper bound on survivorship damage, within each family below.
    - baseline: member days without a price are dropped, which biases returns upward.
    - delist_return: vanishing members receive a terminal return.
    - missing_imputed_penalty: missing members are assumed to earn the observed member
      return minus an annual penalty, applied to the missing share of the universe.
    - years_missing_le: only years whose missing share is small.
    """
    rets = close.pct_change(fill_method=None)
    rows: list[SensitivityRow] = []

    def ann(s: pd.Series) -> float:
        s = s.dropna()
        return float(np.prod(1 + s.to_numpy()) ** (252 / len(s)) - 1) if len(s) else float("nan")

    base = ew_member_return(rets, membership)
    rows.append(
        SensitivityRow("baseline_missing_dropped", ann(base), int(base.notna().sum()), "baseline")
    )
    for d in delist_returns:
        adj = ew_member_return(apply_delisting_returns(rets, membership, d), membership)
        tag = _set_name(d, delist_returns, higher_is_milder=True)
        rows.append(
            SensitivityRow(f"delist_return_{d:+.2f}", ann(adj), int(adj.notna().sum()), tag)
        )
    frac_missing = _daily_missing_frac(close, membership)
    obs = base
    for pen in imputation_penalties:
        imp = obs - pen / 252 * frac_missing
        tag = _set_name(pen, imputation_penalties, higher_is_milder=False)
        rows.append(
            SensitivityRow(
                f"missing_imputed_penalty_{pen:.2f}", ann(imp), int(imp.notna().sum()), tag
            )
        )
    ok_years = set(coverage.loc[coverage["missing_frac"] <= max_missing_frac, "year"])
    sub = base[pd.DatetimeIndex(base.index).year.isin(list(ok_years))]
    rows.append(
        SensitivityRow(
            f"years_missing_le_{max_missing_frac:.2f}", ann(sub), len(sub), "restricted_sample"
        )
    )
    return pd.DataFrame([r.__dict__ for r in rows])


def integrity_checks(
    prices: pd.DataFrame, max_stale_days: int = 5, extreme_return: float = 0.5
) -> dict[str, int]:
    """Counts of suspicious rows. Zero is the healthy value for every key."""
    p = prices.sort_values(["ticker", "date"])
    g = p.groupby("ticker")["adj_close"]
    ret = g.diff() / g.shift()
    unchanged = (g.diff() == 0).astype(int)
    run = unchanged.groupby(p["ticker"]).transform(lambda s: s.groupby((s == 0).cumsum()).cumsum())
    return {
        "duplicate_ticker_dates": int(p.duplicated(["ticker", "date"]).sum()),
        "extreme_returns": int((ret.abs() > extreme_return).sum()),
        "stale_runs_over_limit": int((run == max_stale_days).sum()),
        "non_positive_prices": int((p["adj_close"] <= 0).sum()),
        "lookahead_rows": int((p["available_date"] < p["date"]).sum()),
    }


def quality_report(
    prices: pd.DataFrame,
    membership: pd.DataFrame,
    expected_size: int = 500,
    size_tolerance: int = 10,
    max_stale_days: int = 5,
    extreme_return: float = 0.5,
    max_missing_frac: float = 0.05,
) -> dict[str, object]:
    close = wide_close(prices, pd.DatetimeIndex(membership.index))
    cov = coverage_by_year(membership, close)
    return {
        "integrity": integrity_checks(prices, max_stale_days, extreme_return),
        "coverage_by_year": cov,
        "member_count_reconciliation": reconcile_member_counts(
            monthly_member_counts(membership), expected_size, size_tolerance
        ),
        "survivorship_sensitivity": survivorship_sensitivity(
            close, membership, cov, max_missing_frac=max_missing_frac
        ),
    }


def monthly_member_counts(membership: pd.DataFrame) -> pd.DataFrame:
    """Index member count at each month end."""
    counts = membership.sum(axis=1)
    month_end = counts.groupby(pd.DatetimeIndex(counts.index).to_period("M")).last()
    out = month_end.rename("members").reset_index()
    out["month"] = out["index"].astype(str)
    return out[["month", "members"]]


def reconcile_member_counts(
    counts: pd.DataFrame, expected: int = 500, tolerance: int = 10
) -> pd.DataFrame:
    """Flag months whose member count is further than tolerance from expected."""
    out = counts.copy()
    out["expected"] = expected
    out["deviation"] = out["members"] - expected
    out["ok"] = out["deviation"].abs() <= tolerance
    return out
