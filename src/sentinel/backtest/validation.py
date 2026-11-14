"""Walk-forward and purged k-fold splits with embargo, plus an overlap proof.

Sample i has a label that spans positions [i, i + span], where span = delay + horizon. Two
samples are independent only if their label windows are separated by more than embargo.
Every generator verifies this before yielding, and tests re-verify it independently.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sentinel.backtest.stats import annualize_sharpe, sharpe_per_period


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    test: np.ndarray


class OverlapError(ValueError):
    """Raised when a train sample label window overlaps or touches the embargo of a test one."""


def assert_no_overlap(split: Split, span: int, embargo: int) -> None:
    """Pairwise guarantee: for every train i and test j the gap between label windows
    [i, i+span] and [j, j+span] exceeds embargo."""
    if len(split.train) == 0 or len(split.test) == 0:
        raise OverlapError("empty train or test window")
    i = split.train[:, None]
    j = split.test[None, :]
    gap = np.where(i < j, j - (i + span), i - (j + span))
    if bool((gap <= embargo).any()):
        raise OverlapError("train and test label windows overlap within the embargo")


def walk_forward_splits(
    n: int,
    train_size: int,
    test_size: int,
    span: int = 0,
    embargo: int = 0,
    step: int | None = None,
    expanding: bool = True,
) -> Iterator[Split]:
    """Train strictly before test. The last train sample is purged back by span + embargo."""
    step = step or test_size
    gap = span + embargo + 1
    start = train_size + gap - 1
    if start + test_size > n:
        raise ValueError("series too short for one train window, the purge gap and one test window")
    for test_start in range(start, n - test_size + 1, step):
        last_train = test_start - gap
        first_train = 0 if expanding else max(0, last_train - train_size + 1)
        split = Split(
            np.arange(first_train, last_train + 1), np.arange(test_start, test_start + test_size)
        )
        assert_no_overlap(split, span, embargo)
        yield split


def purged_kfold_splits(n: int, k: int, span: int = 0, embargo: int = 0) -> Iterator[Split]:
    """k contiguous test folds. Train samples whose label window is within span plus embargo
    of the test fold, on either side, are purged."""
    if k < 2 or k > n:
        raise ValueError("k must be between 2 and n")
    for fold in np.array_split(np.arange(n), k):
        a, b = int(fold[0]), int(fold[-1])
        before = np.arange(0, max(0, a - span - embargo))
        after = np.arange(min(n, b + span + embargo + 1), n)
        train = np.concatenate([before, after]).astype(int)
        if len(train) == 0:
            continue
        split = Split(train, fold)
        assert_no_overlap(split, span, embargo)
        yield split


def cross_validate(
    fit_predict: Callable[[np.ndarray, np.ndarray], float], splits: Iterator[Split]
) -> list[float]:
    """Run fit_predict(train_idx, test_idx) on every split and collect the scores."""
    return [fit_predict(s.train, s.test) for s in splits]


def fold_sharpes(returns: pd.Series, splits: Iterator[Split]) -> pd.DataFrame:
    """Annualized Sharpe of a return series on each test window. For rule based strategies
    with no fitted parameters this shows stability across time, not out of sample skill."""
    rows = []
    for k, s in enumerate(splits):
        window = returns.iloc[s.test]
        rows.append(
            {
                "fold": k,
                "test_start": window.index[0],
                "test_end": window.index[-1],
                "sharpe": annualize_sharpe(sharpe_per_period(window)),
                "n": len(window),
            }
        )
    return pd.DataFrame(rows)
