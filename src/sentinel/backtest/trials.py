"""Trial registry. Every evaluated signal, parameter set and portfolio variant is counted.

The deflated Sharpe uses the number of distinct counted trials. Re-running an identical
configuration is recorded as a rerun, visibly, and does not add an independent trial. Any
change to parameters, data, seed or configuration is a new trial.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from sentinel.backtest.holdout import git_state

COUNTED_KINDS = frozenset({"signal", "parameter_set", "portfolio_variant"})
ALL_KINDS = COUNTED_KINDS | {"sensitivity"}
FIELDS = [
    "timestamp_utc",
    "commit",
    "event",
    "trial_hash",
    "kind",
    "name",
    "params_json",
    "rerun",
    "sharpe_per_period",
    "n_obs",
    "skew",
    "kurtosis",
]


class TrialError(RuntimeError):
    pass


@dataclass(frozen=True)
class TrialSpec:
    kind: str
    name: str
    params: dict[str, Any]

    def __post_init__(self) -> None:
        if self.kind not in ALL_KINDS:
            raise TrialError(f"unknown trial kind {self.kind}, expected one of {sorted(ALL_KINDS)}")

    def digest(self) -> str:
        blob = json.dumps(
            {"kind": self.kind, "name": self.name, "params": self.params},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


class TrialRegistry:
    """Append-only log with a persistent counter. Counts survive across processes."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._rows: list[dict[str, Any]] = []
        if path is not None and path.exists() and path.stat().st_size > 0:
            with path.open(newline="") as fh:
                self._rows = list(csv.DictReader(fh))

    # -- writing ---------------------------------------------------------------------------
    def _write(self, row: dict[str, Any]) -> None:
        self._rows.append({k: str(row.get(k, "")) for k in FIELDS})
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in FIELDS})

    def register(self, spec: TrialSpec) -> str:
        """Log an evaluation about to happen. Returns the trial hash."""
        digest = spec.digest()
        rerun = any(r["event"] == "register" and r["trial_hash"] == digest for r in self._rows)
        self._write(
            {
                "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                "commit": git_state()[0],
                "event": "register",
                "trial_hash": digest,
                "kind": spec.kind,
                "name": spec.name,
                "params_json": json.dumps(spec.params, sort_keys=True, default=str),
                "rerun": rerun,
            }
        )
        return digest

    def record_result(
        self, digest: str, sharpe_per_period: float, n_obs: int, skew: float, kurtosis: float
    ) -> None:
        reg = next((r for r in self._rows if r["trial_hash"] == digest), None)
        if reg is None:
            raise TrialError("result for a trial that was never registered")
        self._write(
            {
                "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                "commit": git_state()[0],
                "event": "result",
                "trial_hash": digest,
                "kind": reg["kind"],
                "name": reg["name"],
                "sharpe_per_period": sharpe_per_period,
                "n_obs": n_obs,
                "skew": skew,
                "kurtosis": kurtosis,
            }
        )

    # -- reading ---------------------------------------------------------------------------
    def _registered(self) -> list[dict[str, Any]]:
        return [r for r in self._rows if r["event"] == "register"]

    @property
    def n_evaluations(self) -> int:
        """Every registration including reruns and sensitivity cells."""
        return len(self._registered())

    @property
    def n_reruns(self) -> int:
        return sum(1 for r in self._registered() if r["rerun"] == "True")

    @property
    def n_trials(self) -> int:
        """Distinct counted trials. This is the N used by the deflated Sharpe."""
        return len({r["trial_hash"] for r in self._registered() if r["kind"] in COUNTED_KINDS})

    @property
    def n_sensitivity_cells(self) -> int:
        return len({r["trial_hash"] for r in self._registered() if r["kind"] == "sensitivity"})

    def counted_sharpes(self) -> list[float]:
        """Latest per period Sharpe of each distinct counted trial that has a result."""
        latest: dict[str, float] = {}
        for r in self._rows:
            if r["event"] == "result" and r["kind"] in COUNTED_KINDS:
                latest[r["trial_hash"]] = float(r["sharpe_per_period"])
        return list(latest.values())

    def sharpe_variance(self) -> float:
        """Variance of per period Sharpe across counted trials, needed for the deflation.

        Raises if the registry counts more than one trial but holds fewer than two results,
        because deflating with an unknown variance would silently understate the haircut.
        """
        sharpes = self.counted_sharpes()
        if self.n_trials <= 1:
            return 0.0
        if len(sharpes) < 2:
            raise TrialError(
                f"{self.n_trials} trials registered but only {len(sharpes)} results recorded"
            )
        return float(np.var(sharpes, ddof=1))
