"""Perturbing information that was not yet available must not change what a trader sees."""

import numpy as np
import pandas as pd

from sentinel.data.pit import asof_join, known_as_of
from sentinel.data.synthetic import make_market


def test_future_rows_do_not_change_past_view() -> None:
    m = make_market(periods=200)
    cut = m.calendar[100]
    base = known_as_of(m.prices, cut)
    perturbed = m.prices.copy()
    future = perturbed["available_date"] >= cut
    perturbed.loc[future, "adj_close"] = np.random.default_rng(1).uniform(1, 2, int(future.sum()))
    again = known_as_of(perturbed, cut)
    pd.testing.assert_frame_equal(base.reset_index(drop=True), again.reset_index(drop=True))


def test_asof_join_invariant_to_future_edits() -> None:
    m = make_market(periods=100)
    one = m.prices[m.prices["ticker"] == "S000"][["available_date", "adj_close"]]
    left = pd.DataFrame({"d": m.calendar[:50]})
    ref = asof_join(left, one, "d", right_cols=["adj_close"])
    edited = one.copy()
    edited.loc[edited["available_date"] >= m.calendar[50], "adj_close"] = -1.0
    out = asof_join(left, edited, "d", right_cols=["adj_close"])
    pd.testing.assert_frame_equal(ref, out)
