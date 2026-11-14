"""Every registered signal must be unchanged at date t when data after t is altered."""

import numpy as np
import pytest

from sentinel.data.synthetic import make_market
from sentinel.signals.base import Panel, panel_from_prices
from sentinel.signals.library import REGISTRY


def _perturb(p: Panel, cut: int, seed: int = 0) -> Panel:
    rng = np.random.default_rng(seed)
    frames = []
    for f in (p.close, p.high, p.low, p.volume):
        g = f.copy()
        g.iloc[cut + 1 :] = rng.uniform(1, 1000, g.iloc[cut + 1 :].shape)
        frames.append(g)
    return Panel(*frames)


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_signal_ignores_future(name: str) -> None:
    m = make_market(periods=420)
    panel = panel_from_prices(m.prices, m.calendar)
    fn, spec = REGISTRY[name]
    cut = 330
    base = fn(panel).iloc[: cut + 1]
    changed = fn(_perturb(panel, cut)).iloc[: cut + 1]
    assert base.equals(changed) or np.allclose(base.to_numpy(), changed.to_numpy(), equal_nan=True)
    assert base.iloc[-1].notna().sum() > 0  # signal is actually populated at the cut
    assert spec.min_history < cut


def test_leakage_test_detects_a_leaky_signal() -> None:
    """Negative control: a signal that uses tomorrow must fail the invariance check."""
    m = make_market(periods=420)
    panel = panel_from_prices(m.prices, m.calendar)

    def leaky(p: Panel):  # type: ignore[no-untyped-def]
        return p.close.shift(-1) / p.close

    cut = 330
    assert not np.allclose(
        leaky(panel).iloc[: cut + 1].to_numpy(),
        leaky(_perturb(panel, cut)).iloc[: cut + 1].to_numpy(),
        equal_nan=True,
    )
