import numpy as np
import pandas as pd
import pytest

from sentinel.signals import library as lib
from sentinel.signals.base import Panel, forward_returns, winsorize, zscore


def _panel(close: list[float] | np.ndarray, volume: float | list[float] = 1000.0) -> Panel:
    idx = pd.bdate_range("2020-01-01", periods=len(close))
    c = pd.DataFrame({"A": np.asarray(close, dtype=float)}, index=idx)
    v = pd.DataFrame({"A": np.broadcast_to(np.asarray(volume, dtype=float), len(close))}, index=idx)
    return Panel(close=c, high=c * 1.01, low=c * 0.99, volume=v)


def test_forward_returns_timing() -> None:
    p = _panel([100, 110, 121, 133.1, 146.41])
    f = forward_returns(p.close, horizon=1, delay=1)
    assert f["A"].iloc[0] == pytest.approx(0.10)  # close[2]/close[1] - 1
    assert np.isnan(f["A"].iloc[-1]) and np.isnan(f["A"].iloc[-2])
    f0 = forward_returns(p.close, horizon=1, delay=0)
    assert f0["A"].iloc[0] == pytest.approx(0.10)
    with pytest.raises(ValueError):
        forward_returns(p.close, horizon=0)


def test_max_lottery_known_answer() -> None:
    close = 100 * np.cumprod(1 + np.array([0, 0.01, -0.02, 0.05, 0.03, -0.01, 0.02]))
    out = lib.max_lottery(_panel(close), window=5, k=2)
    # last window returns: -0.02, 0.05, 0.03, -0.01, 0.02 -> top two 0.05 and 0.03
    assert out["A"].iloc[-1] == pytest.approx(0.04)


def test_high52_and_reversal() -> None:
    close = [10.0, 12.0, 11.0, 9.0]
    p = _panel(close)
    assert lib.high52(p, window=3)["A"].iloc[-1] == pytest.approx(9.0 / 12.0)
    assert lib.reversal(p, window=2)["A"].iloc[-1] == pytest.approx(9.0 / 12.0 - 1)


def test_overhang_sign() -> None:
    flat = lib.overhang(_panel(np.full(400, 50.0)), vol_window=50)
    assert flat["A"].dropna().abs().max() < 1e-12
    rising = lib.overhang(_panel(np.linspace(50, 100, 400)), vol_window=50)
    assert rising["A"].iloc[-1] > 0  # price above its slowly moving reference
    falling = lib.overhang(_panel(np.linspace(100, 50, 400)), vol_window=50)
    assert falling["A"].iloc[-1] < 0


def test_realized_skew_sign_and_volume_spike() -> None:
    r = np.full(80, -0.001)
    r[40] = 0.2
    skew = lib.realized_skew(_panel(100 * np.cumprod(1 + r)), window=63)
    assert skew["A"].iloc[-1] > 1
    vol = np.full(400, 1000.0)
    vol[-3:] = 5000.0
    av = lib.abnormal_volume(_panel(np.full(400, 10.0), vol))
    assert av["A"].iloc[-1] > 0.3 and abs(av["A"].iloc[-10]) < 1e-9


def test_cross_sectional_transforms() -> None:
    x = pd.DataFrame(np.random.default_rng(0).normal(size=(5, 200)))
    x.iloc[:, 0] = 1000.0
    w = winsorize(x)
    assert (w.max(axis=1) < 1000).all()
    z = zscore(x)
    assert z.mean(axis=1).abs().max() < 1e-9
    assert (z.std(axis=1, ddof=0) - 1).abs().max() < 1e-9


def test_registry_is_complete() -> None:
    assert len(lib.REGISTRY) >= 6
    for name, (_, spec) in lib.REGISTRY.items():
        assert spec.name == name and spec.expected_sign in (-1, 1)
        assert spec.hypothesis and spec.construction and spec.reference
