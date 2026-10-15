"""Universe feasibility check, Amendment 2 section C.

With a per name cap and a gross target, each leg needs at least (max_gross / 2) / max_position
names for the cap to allow full gross. The universe must hold that many names per leg divided
by the leg quantile. Real data runs always enforce this and fail loudly. Synthetic runs only
report it, and the report says the fixture is a mechanism test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from sentinel.config import RunConfig
from sentinel.portfolio.build import eligible_indices, leg_size, rebalance_rows
from sentinel.signals.base import Panel


class FeasibilityError(RuntimeError):
    """Raised when the universe is too small for the pre-registered constraints."""


def required_leg_names(max_position: float, max_gross: float) -> int:
    return math.ceil((max_gross / 2.0) / max_position - 1e-12)


def required_universe(max_position: float, max_gross: float, leg_quantile: float) -> int:
    n_leg = required_leg_names(max_position, max_gross)
    n = math.ceil(n_leg / leg_quantile - 1e-12)
    while leg_size(n, leg_quantile) < n_leg:  # guard against float rounding
        n += 1
    return n


@dataclass(frozen=True)
class FeasibilityReport:
    required_leg_names: int
    required_universe: int
    n_rebalances: int
    min_observed: int
    min_observed_date: str
    n_rebalances_short: int
    enforced: bool

    @property
    def passes(self) -> bool:
        return self.n_rebalances > 0 and self.n_rebalances_short == 0

    def summary(self) -> str:
        verdict = "PASS" if self.passes else "FAIL"
        return (
            f"{verdict}: full gross under the position cap needs {self.required_leg_names} names "
            f"per leg and a universe of {self.required_universe}. Minimum eligible universe "
            f"observed {self.min_observed} on {self.min_observed_date}. Rebalances below the "
            f"requirement: {self.n_rebalances_short} of {self.n_rebalances}."
        )


def enforcement(synthetic: bool, require_flag: bool) -> bool:
    """Real runs always enforce. Synthetic runs enforce only if the config switch is on."""
    return (not synthetic) or require_flag


def eligible_counts(
    panel: Panel, score: pd.DataFrame, membership: pd.DataFrame, run: RunConfig
) -> pd.Series:
    """Eligible universe size at each rebalance row, using the same rule as the optimizer."""
    cols = list(panel.close.columns)
    member = membership.reindex(index=panel.close.index, columns=cols).fillna(False)
    rows = rebalance_rows(len(panel.close), run.backtest.rebalance_every, run.portfolio.cov_window)
    return pd.Series(
        [len(eligible_indices(panel, score, member, i, run.portfolio.cov_window)) for i in rows],
        index=panel.close.index[rows],
        dtype=int,
    )


def membership_counts(membership: pd.DataFrame, run: RunConfig) -> pd.Series:
    """Index member count at each rebalance row. An upper bound on the eligible universe."""
    rows = rebalance_rows(len(membership), run.backtest.rebalance_every, run.portfolio.cov_window)
    counts = membership.sum(axis=1).astype(int)
    return pd.Series(counts.to_numpy()[rows], index=membership.index[rows], dtype=int)


def check_feasibility(counts: pd.Series, run: RunConfig, enforce: bool) -> FeasibilityReport:
    """Compare eligible universe sizes with the requirement. Raises when enforced and failing."""
    need = required_universe(
        run.backtest.max_position, run.backtest.max_gross, run.portfolio.leg_quantile
    )
    report = FeasibilityReport(
        required_leg_names=required_leg_names(run.backtest.max_position, run.backtest.max_gross),
        required_universe=need,
        n_rebalances=len(counts),
        min_observed=int(counts.min()) if len(counts) else 0,
        min_observed_date=(
            str(counts.index[int(counts.to_numpy().argmin())].date()) if len(counts) else "n/a"
        ),
        n_rebalances_short=int((counts < need).sum()),
        enforced=enforce,
    )
    if enforce and not report.passes:
        raise FeasibilityError(
            "FEASIBILITY CHECK FAILED. "
            + report.summary()
            + " Widen the universe or change the pre-registered cap through an amendment."
            if len(counts)
            else "FEASIBILITY CHECK FAILED. No rebalance dates are available after the warm-up."
        )
    return report
