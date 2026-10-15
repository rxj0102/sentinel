"""Point-in-time index membership built from an index change log."""

from __future__ import annotations

import pandas as pd


def reconstruct_members_at(
    current: set[str], changes: pd.DataFrame, as_of: pd.Timestamp
) -> set[str]:
    """Roll the change log backwards from today to recover membership on as_of.

    A change dated d takes effect on d. Undoing every change with date > as_of leaves the
    set of members on as_of. Gaps in the public change log make this an approximation, so
    callers should compare the result size against the expected index size.
    """
    members = set(current)
    later = changes[changes["date"] > as_of].sort_values("date", ascending=False)
    for row in later.itertuples(index=False):
        if isinstance(row.added, str):
            members.discard(row.added)
        if isinstance(row.removed, str):
            members.add(row.removed)
    return members


def membership_matrix(
    initial: set[str],
    initial_date: pd.Timestamp,
    changes: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Boolean frame (calendar x tickers), True when a ticker is an index member that day."""
    events = changes[changes["date"] > initial_date].sort_values("date")
    universe = set(initial)
    for col in ("added", "removed"):
        universe |= {t for t in events[col].dropna() if isinstance(t, str)}
    cols = sorted(universe) if tickers is None else tickers
    state = pd.DataFrame(False, index=calendar, columns=cols)
    for t in initial & set(cols):
        state[t] = True
    for row in events.itertuples(index=False):
        mask = state.index >= row.date
        if isinstance(row.added, str) and row.added in state.columns:
            state.loc[mask, row.added] = True
        if isinstance(row.removed, str) and row.removed in state.columns:
            state.loc[mask, row.removed] = False
    return state


def membership_from_meta(
    meta: dict[str, object], changes: pd.DataFrame, calendar: pd.DatetimeIndex
) -> pd.DataFrame:
    """Membership matrix from the meta file and an already guarded change log.

    The caller must clip changes to the development window first, a change dated inside the
    holdout reveals holdout period information.
    """
    initial = meta["initial_members"]
    assert isinstance(initial, list)
    return membership_matrix(
        set(initial), pd.Timestamp(str(meta["initial_date"])), changes, calendar
    )


def mask_to_members(signal: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    """NaN out signal values for names that are not index members that day.

    Without this, ranking would include names that had not yet joined the index, which is a
    look-ahead in the universe.
    """
    member = membership.reindex(index=signal.index, columns=signal.columns).fillna(False)
    return signal.where(member.astype(bool))
