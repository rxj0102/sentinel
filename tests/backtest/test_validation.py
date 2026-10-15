import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from sentinel.backtest.validation import (
    OverlapError,
    Split,
    assert_no_overlap,
    cross_validate,
    fold_sharpes,
    purged_kfold_splits,
    walk_forward_splits,
)


def _brute_force_clean(split: Split, span: int, embargo: int) -> bool:
    """Independent check: windows expanded by the embargo must not intersect."""
    for i in split.train:
        a = set(range(int(i), int(i) + span + embargo + 1))
        a_core = set(range(int(i), int(i) + span + 1))
        for j in split.test:
            b = set(range(int(j), int(j) + span + embargo + 1))
            b_core = set(range(int(j), int(j) + span + 1))
            if a & b_core or b & a_core:
                return False
    return True


@given(
    n=st.integers(60, 200),
    train=st.integers(10, 40),
    test=st.integers(3, 15),
    span=st.integers(0, 8),
    embargo=st.integers(0, 6),
    expanding=st.booleans(),
)
@settings(max_examples=60, deadline=None)
def test_walk_forward_has_no_overlap_including_embargo(
    n: int, train: int, test: int, span: int, embargo: int, expanding: bool
) -> None:
    assume(n >= train + span + embargo + test)
    splits = list(walk_forward_splits(n, train, test, span, embargo, expanding=expanding))
    assert splits
    prev_end = -1
    for s in splits:
        assert s.train.max() < s.test.min()
        assert s.test.min() - s.train.max() > span + embargo  # purge gap
        assert _brute_force_clean(s, span, embargo)
        assert s.test.min() > prev_end  # test windows never overlap each other
        prev_end = s.test.max()
        assert s.test.max() < n


@given(
    n=st.integers(40, 150), k=st.integers(2, 8), span=st.integers(0, 6), embargo=st.integers(0, 5)
)
@settings(max_examples=60, deadline=None)
def test_purged_kfold_has_no_overlap_including_embargo(
    n: int, k: int, span: int, embargo: int
) -> None:
    splits = list(purged_kfold_splits(n, k, span, embargo))
    tested = np.concatenate([s.test for s in splits])
    assert sorted(tested.tolist()) == list(range(n))  # every sample tested exactly once
    for s in splits:
        assert not set(s.train.tolist()) & set(s.test.tolist())
        assert _brute_force_clean(s, span, embargo)


def test_purged_kfold_without_purge_is_the_complement() -> None:
    for s in purged_kfold_splits(20, 4, span=0, embargo=0):
        assert sorted(s.train.tolist() + s.test.tolist()) == list(range(20))


def test_known_purge_sizes() -> None:
    s = list(purged_kfold_splits(30, 3, span=2, embargo=1))[1]  # test fold is 10..19
    assert s.test.min() == 10 and s.test.max() == 19
    assert s.train.max() == 29 and s.train.min() == 0
    assert 7 not in s.train and 6 in s.train  # before: i + span + embargo < 10 -> i <= 6
    assert 22 not in s.train and 23 in s.train  # after: i > 19 + 2 + 1 = 22


def test_overlap_detector_catches_leaks() -> None:
    leaky = Split(train=np.arange(0, 10), test=np.arange(10, 15))
    with pytest.raises(OverlapError):
        assert_no_overlap(leaky, span=2, embargo=0)  # sample 9 label reaches 11
    touching = Split(train=np.arange(0, 7), test=np.arange(10, 15))
    assert_no_overlap(touching, span=2, embargo=0)
    assert_no_overlap(touching, span=2, embargo=1)  # one free position between the windows
    with pytest.raises(OverlapError):
        assert_no_overlap(touching, span=2, embargo=2)  # embargo needs two free positions
    with pytest.raises(OverlapError):
        assert_no_overlap(Split(np.array([], dtype=int), np.arange(3)), 0, 0)


def test_bad_arguments_and_helpers() -> None:
    with pytest.raises(ValueError, match="too short"):
        list(walk_forward_splits(30, 20, 10, span=5, embargo=5))
    with pytest.raises(ValueError):
        list(purged_kfold_splits(10, 1))
    scores = cross_validate(lambda tr, te: float(len(te)), purged_kfold_splits(20, 4))
    assert scores == [5.0] * 4
    r = pd.Series(
        np.random.default_rng(0).normal(0, 0.01, 100),
        index=pd.bdate_range("2020-01-01", periods=100),
    )
    tab = fold_sharpes(r, walk_forward_splits(100, 40, 20, span=1, embargo=1))
    assert len(tab) >= 2 and tab["n"].eq(20).all()
