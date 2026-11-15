"""Published portfolios track (Amendment 3 of docs/hypotheses.md).

The free stock-level source loses about a third of the index to survivorship, so the locked
signs are tested here on long-short portfolio returns from the Open Source Asset Pricing data
of Chen and Zimmermann, which are built from the full CRSP universe with delisting returns.

What this track tests: the locked sign of a signal against its nearest published definition.
What it does not test: the Sentinel implementation, the S and P 500 universe, or the engine.

Every rule below (mapping, sample, statistic, verdicts) is fixed in Amendment 3. The holdout
is protected in refresh(): rows on or after the holdout start are dropped before anything is
computed or written, so the committed snapshot cannot contain them.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from sentinel.backtest.stats import holm_adjust, moments, sharpe_per_period
from sentinel.backtest.trials import TrialRegistry, TrialSpec
from sentinel.results import write_result

RELEASE = "Data-Release-2026.10"
BASE = f"https://huggingface.co/datasets/OpenSourceAP/CrossSection/resolve/{RELEASE}/"
FILES = {
    "quintiles_vw": "Portfolios/Full Sets Alt/PredictorAltPorts_QuintilesVW.parquet",
    "quintiles_ew": "Portfolios/Full Sets Alt/PredictorAltPorts_Quintiles.parquet",
    "original_paper": "Portfolios/Full Sets OP/PredictorPortsFull.parquet",
}
PRIMARY = "quintiles_vw"
# locked signal -> (published acronym, locked sign at 21 days, which equals the published sign)
MAPPING: dict[str, tuple[str, int]] = {
    "max_lottery": ("MaxRet", -1),
    "high52": ("High52", 1),
    "reversal": ("STreversal", -1),
    "realized_skew": ("ReturnSkew", -1),
}
UNMAPPED = ("overhang", "abnormal_volume")
DEV_START = "2005-01"
NW_LAGS = 6
ALPHA = 0.05
EVIDENCE = Path("evidence") / "published"


def newey_west_t(x: np.ndarray[Any, np.dtype[np.float64]], lags: int = NW_LAGS) -> float:
    """t-statistic of the mean with a Newey-West (Bartlett) standard error."""
    n = len(x)
    e = x - x.mean()
    v = float(e @ e) / n
    for lag in range(1, min(lags, n - 1) + 1):
        v += 2.0 * (1.0 - lag / (lags + 1.0)) * float(e[lag:] @ e[:-lag]) / n
    return float(x.mean() / np.sqrt(v / n))


def keep_development_rows(df: pd.DataFrame, cutoff: str) -> pd.DataFrame:
    """Drop every row from the month of cutoff on. Called before anything else touches df."""
    first_dropped = pd.Timestamp(cutoff).replace(day=1)
    return df[df["date"] < first_dropped].copy()


def build_snapshot(root: Path, frames: dict[str, pd.DataFrame], cutoff: str) -> dict[str, Any]:
    """Write the long-short series of the mapped signals from the downloaded tables.

    frames maps a construction name to its table. The caller (the CLI) does the download and
    the file read, so that every data reader stays in a module the audit allows.
    """
    wanted = [a for a, _ in MAPPING.values()]
    parts = []
    for construction, raw in frames.items():
        df = raw.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = keep_development_rows(df, cutoff)
        df = df[df["signalname"].isin(wanted) & (df["port"] == "LS")]
        parts.append(
            pd.DataFrame(
                {
                    "construction": construction,
                    "signal": df["signalname"].to_numpy(),
                    "month": df["date"].dt.strftime("%Y-%m").to_numpy(),
                    "ret_pct": df["ret"].round(6).to_numpy(),
                }
            )
        )
    out = pd.concat(parts).sort_values(["construction", "signal", "month"]).reset_index(drop=True)
    path_out = write_result(root / EVIDENCE, "long_short_returns", out, synthetic=False)
    manifest = {
        "source": "Open Source Asset Pricing, Chen and Zimmermann",
        "release": RELEASE,
        "files": {k: url(k) for k in FILES},
        "rows": len(out),
        "first_month": str(out["month"].min()),
        "last_month": str(out["month"].max()),
        "rows_dropped_from": cutoff,
        "sha256": hashlib.sha256(path_out.read_bytes()).hexdigest(),
    }
    write_result(root / EVIDENCE, "manifest", manifest, synthetic=False)
    return manifest


def url(construction: str) -> str:
    return BASE + FILES[construction].replace(" ", "%20")


def series(df: pd.DataFrame, construction: str, acronym: str) -> pd.Series:
    sub = df[(df["construction"] == construction) & (df["signal"] == acronym)]
    return pd.Series(sub["ret_pct"].to_numpy() / 100.0, index=sub["month"].to_numpy(), name=acronym)


def describe(r: pd.Series) -> dict[str, Any]:
    """Mean, Newey-West t, p-values and Sharpe ratio of a monthly long-short series."""
    x = r.dropna().to_numpy(dtype=np.float64)
    t = newey_west_t(x)
    return {
        "n_months": len(x),
        "mean_monthly": float(x.mean()),
        "mean_annual": float(x.mean() * 12.0),
        "t": t,
        "p_one_sided": float(1.0 - norm.cdf(t)),
        "p_two_sided": float(2.0 * (1.0 - norm.cdf(abs(t)))),
        "sharpe_annual": float(np.sqrt(12.0) * x.mean() / x.std(ddof=1)),
    }


def rounded(x: Any) -> Any:
    """Six significant digits, so the file is identical across machines and library builds."""
    if isinstance(x, dict):
        return {k: rounded(v) for k, v in x.items()}
    if isinstance(x, list):
        return [rounded(v) for v in x]
    if isinstance(x, float):
        return float(f"{x:.6g}")
    return x


def verdict(mean: float, p_one_holm: float, p_two_holm: float) -> str:
    if mean > 0 and p_one_holm < ALPHA:
        return "confirmed"
    if mean < 0 and p_two_holm < ALPHA:
        return "contradicted"
    return "not confirmed"


def run(
    root: Path, df: pd.DataFrame, dev_end: str, registry: TrialRegistry | None = None
) -> dict[str, Any]:
    """Apply the tests of Amendment 3 to the frozen series and write the metrics file."""

    def dev(x: pd.Series) -> pd.Series:
        return x[(x.index >= DEV_START) & (x.index <= dev_end[:7])]

    def before(x: pd.Series) -> pd.Series:
        return x[x.index < DEV_START]

    signals: dict[str, Any] = {}
    dev_series = {}
    for name, (acronym, sign) in MAPPING.items():
        full = series(df, PRIMARY, acronym)
        dev_series[name] = dev(full)
        row = describe(dev_series[name])
        row.update(
            {
                "published_signal": acronym,
                "locked_sign": sign,
                "context_before_window": describe(before(full))
                | {"first_month": str(full.index.min())},
                "robustness": {
                    c: describe(dev(series(df, c, acronym))) for c in FILES if c != PRIMARY
                },
            }
        )
        signals[name] = row
    names = list(MAPPING)
    one = holm_adjust([signals[n]["p_one_sided"] for n in names])
    two = holm_adjust([signals[n]["p_two_sided"] for n in names])
    for n, a, b in zip(names, one, two, strict=True):
        signals[n]["p_one_sided_holm"] = float(a)
        signals[n]["p_two_sided_holm"] = float(b)
        signals[n]["verdict"] = verdict(signals[n]["mean_monthly"], a, b)

    comp = pd.concat(dev_series, axis=1).dropna().mean(axis=1)
    composite = describe(comp)
    composite["verdict"] = verdict(
        composite["mean_monthly"], composite["p_one_sided"], composite["p_two_sided"]
    )
    corr = pd.concat(dev_series, axis=1).corr()
    composite["mean_pairwise_correlation"] = float(
        corr.to_numpy()[np.triu_indices(len(names), 1)].mean()
    )

    if registry is not None:
        todo = [(f"published|{n}|{PRIMARY}", "signal", dev_series[n]) for n in names]
        todo.append((f"published|composite|{PRIMARY}", "portfolio_variant", comp))
        for trial_name, kind, r in todo:
            spec = TrialSpec(
                kind, trial_name, {"release": RELEASE, "start": DEV_START, "end": dev_end[:7]}
            )
            digest = registry.register(spec)
            skew, kurt = moments(r)
            registry.record_result(digest, sharpe_per_period(r), len(r), skew, kurt)

    metrics = {
        "release": RELEASE,
        "construction": PRIMARY,
        "window": {"start": DEV_START, "end": dev_end[:7], "nw_lags": NW_LAGS, "alpha": ALPHA},
        "signals": signals,
        "composite": composite,
        "not_tested": list(UNMAPPED),
    }
    out: dict[str, Any] = rounded(metrics)
    write_result(root / EVIDENCE, "metrics", out, synthetic=False)
    return out
