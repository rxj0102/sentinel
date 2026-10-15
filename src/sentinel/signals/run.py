"""Signal research pipeline: IC, decay, turnover, capacity, orthogonalized IC, model cards."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sentinel.signals import ic as icmod
from sentinel.signals import orthogonalize as orth
from sentinel.signals import portfolio_stats as ps
from sentinel.signals.base import Panel, SignalSpec, forward_returns, zscore
from sentinel.signals.library import REGISTRY

HORIZONS = (1, 5, 10, 21, 63)


def research(panel: Panel, horizons: tuple[int, ...] = HORIZONS) -> dict[str, pd.DataFrame]:
    """Run every registered signal. Returns tables keyed by name.

    Tables: ic_decay (per signal and horizon), portfolio (turnover and capacity per signal),
    ortho_ic (21 day IC before and after removing size and momentum exposure).
    """
    adv = panel.dollar_volume.rolling(63, min_periods=30).mean()
    adv = adv.where(adv > 0)
    size = pd.DataFrame(np.log(adv), index=adv.index, columns=adv.columns)
    mom = panel.close.shift(21) / panel.close.shift(252) - 1.0
    exposures = {"log_dollar_volume": zscore(size), "momentum_12_1": zscore(mom)}
    decay_rows, port_rows, ortho_rows = [], [], []
    fwd21 = forward_returns(panel.close, 21)
    for name, (fn, spec) in REGISTRY.items():
        sig = fn(panel)
        dec = icmod.decay_table(sig, panel.close, spec.expected_sign, horizons)
        dec.insert(0, "signal", name)
        decay_rows.append(dec)
        w = ps.quantile_weights(sig, 0.2, sign=spec.expected_sign)
        port_rows.append(
            {
                "signal": name,
                "mean_one_way_turnover": float(ps.turnover(w).mean()),
                "capacity_aum_5pct_adv": ps.capacity(w, panel.dollar_volume),
            }
        )
        raw = icmod.rank_ic(sig, fwd21)
        res = icmod.rank_ic(orth.residualize(zscore(sig), exposures), fwd21)
        ortho_rows.append(
            {
                "signal": name,
                "mean_ic_21d_raw": float(raw.mean()),
                "mean_ic_21d_orthogonalized": float(res.mean()),
                "retained_fraction": float(res.mean() / raw.mean()) if raw.mean() else np.nan,
            }
        )
    return {
        "signals_ic_decay": pd.concat(decay_rows, ignore_index=True),
        "signals_portfolio": pd.DataFrame(port_rows),
        "signals_ortho_ic": pd.DataFrame(ortho_rows),
    }


def render_model_card(spec: SignalSpec) -> str:
    sign = "positive" if spec.expected_sign > 0 else "negative"
    caveats = "\n".join(f"- {c}" for c in spec.caveats) or "- None recorded yet"
    return f"""# Model card: {spec.name}

## Economic hypothesis

{spec.hypothesis}

Expected sign of the relation to next period returns: {sign}.
Pre-registered before any real data was touched.

## Construction

{spec.construction}

Minimum history in trading days: {spec.min_history}. Data: {spec.data}.
Timing: the signal at date t uses data through the close of t.
Positions open at the close of t plus one day.

## Reference

{spec.reference}

## Known caveats

{caveats}

## Results

PENDING REAL-DATA GATE. IC, decay, turnover, capacity and orthogonalization results
are produced by the signals research command and may be added here only after the gate
in docs/REAL_DATA_GATE.md is closed.
"""
