"""Holdout enforcement. The holdout is touched once, and every touch is logged.

Rules enforced in code, not by convention:
- Any request for data at or after holdout_start is refused unless a grant from the explicit
  unlock step is presented. The unlock needs holdout_end in the config and a stated reason.
- Only one unlock is ever granted per access log. A second one is refused. One grant can be
  used for many reads within the locked window, each read is logged.
- Every denied or granted access is appended to the log with a UTC timestamp and the commit
  hash. The real log lives in holdout/access_log.csv and is committed. Synthetic runs write to
  results/synthetic so they never pollute the real log.
"""

from __future__ import annotations

import csv
import secrets
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

LOG_FIELDS = [
    "timestamp_utc",
    "commit",
    "dirty",
    "action",
    "caller",
    "requested_end",
    "holdout_start",
    "holdout_end",
    "reason",
]
REAL_LOG = Path("holdout/access_log.csv")


class HoldoutError(RuntimeError):
    """Base class for holdout violations."""


class HoldoutAccessDeniedError(HoldoutError):
    """Raised when holdout dates are requested without an explicit unlock."""


class HoldoutAlreadyUsedError(HoldoutError):
    """Raised when a second unlock is attempted. The holdout is touched once."""


def git_state(cwd: Path | None = None) -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=cwd
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
                cwd=cwd,
            ).stdout.strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", False


@dataclass(frozen=True)
class HoldoutGrant:
    """Proof of a granted unlock. Only HoldoutGuard.unlock issues one, once per access log."""

    token: str
    holdout_start: pd.Timestamp
    through: pd.Timestamp


@dataclass(frozen=True)
class HoldoutGuard:
    holdout_start: pd.Timestamp
    holdout_end: pd.Timestamp | None
    log_path: Path
    _issued: set[str] = field(default_factory=set, compare=False, repr=False)

    @classmethod
    def for_run(
        cls,
        holdout_start: pd.Timestamp,
        holdout_end: pd.Timestamp | None,
        synthetic: bool,
        results_root: Path = Path("results"),
        real_log: Path = REAL_LOG,
    ) -> HoldoutGuard:
        """Real runs log to the committed holdout/access_log.csv. Synthetic runs log under
        the results root so they never pollute the committed log."""
        path = results_root / "synthetic" / "holdout_access_log.csv" if synthetic else real_log
        return cls(holdout_start, holdout_end, path)

    # -- logging ---------------------------------------------------------------------------
    def _append(self, action: str, caller: str, requested_end: pd.Timestamp, reason: str) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.log_path.exists() or self.log_path.stat().st_size == 0
        commit, dirty = git_state()
        with self.log_path.open("a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(LOG_FIELDS)
            w.writerow(
                [
                    datetime.now(UTC).isoformat(timespec="seconds"),
                    commit,
                    dirty,
                    action,
                    caller,
                    requested_end.date(),
                    self.holdout_start.date(),
                    self.holdout_end.date() if self.holdout_end is not None else "",
                    reason,
                ]
            )

    def access_log(self) -> pd.DataFrame:
        if not self.log_path.exists() or self.log_path.stat().st_size == 0:
            return pd.DataFrame(columns=LOG_FIELDS)
        return pd.read_csv(self.log_path)

    def already_unlocked(self) -> bool:
        log = self.access_log()
        return bool((log["action"] == "granted").any())

    # -- enforcement -----------------------------------------------------------------------
    def unlock(
        self, requested_end: pd.Timestamp, caller: str = "unknown", reason: str = ""
    ) -> HoldoutGrant:
        """The one explicit unlock. Logs the outcome, returns a grant or raises.

        Requires holdout_end in the config, a stated reason, a request inside the locked
        window, and that no earlier grant exists in the access log.
        """
        if self.holdout_end is None:
            self._append("denied_no_end_date", caller, requested_end, reason)
            raise HoldoutError(
                "holdout_end is not written in the config, the holdout is not locked"
            )
        if not reason.strip():
            self._append("denied_no_reason", caller, requested_end, reason)
            raise HoldoutError("unlocking the holdout requires a stated reason")
        if requested_end > self.holdout_end:
            self._append("denied_past_end", caller, requested_end, reason)
            raise HoldoutError("request extends past the fixed holdout end date")
        if self.already_unlocked():
            self._append("denied_repeat", caller, requested_end, reason)
            raise HoldoutAlreadyUsedError("the holdout was already used once and cannot be reused")
        self._append("granted", caller, requested_end, reason)
        token = secrets.token_hex(8)
        self._issued.add(token)
        return HoldoutGrant(token, self.holdout_start, requested_end)

    def authorize(
        self,
        requested_end: pd.Timestamp,
        grant: HoldoutGrant | None = None,
        caller: str = "unknown",
        reason: str = "",
    ) -> None:
        """Raise unless every requested date is before holdout_start or a valid grant covers it.

        Development only requests are silent. Holdout requests are always logged, whether
        denied or served under a grant.
        """
        if requested_end < self.holdout_start:
            return
        if grant is None:
            self._append("denied_no_unlock", caller, requested_end, reason)
            raise HoldoutAccessDeniedError(
                f"data through {requested_end.date()} reaches the holdout starting "
                f"{self.holdout_start.date()}. Unlock it explicitly with a stated reason."
            )
        if grant.token not in self._issued or requested_end > grant.through:
            self._append("denied_invalid_grant", caller, requested_end, reason)
            raise HoldoutAccessDeniedError("grant is not valid for this guard or this date range")
        self._append("access_under_grant", caller, requested_end, reason)

    def dev_only(self, df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
        """Explicitly drop holdout rows. Allowed and unlogged because nothing is revealed."""
        return df[df[date_col] < self.holdout_start]

    def load(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
        grant: HoldoutGrant | None = None,
        caller: str = "unknown",
        reason: str = "",
    ) -> pd.DataFrame:
        """Return df if it is development only, otherwise require a valid grant."""
        if df.empty:
            return df
        self.authorize(pd.Timestamp(df[date_col].max()), grant, caller, reason)
        return df
