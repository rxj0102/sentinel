"""Run configuration. One YAML file fully determines a run."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, model_validator


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    start: date
    dev_end: date
    holdout_start: date
    holdout_end: date | None = None  # written on the day the holdout is locked
    seed: int = 0
    raw_dir: Path = Path("data/raw")
    curated_dir: Path = Path("data/curated")
    pit_lag_days: int = 1  # calendar days between a price date and its availability
    max_stale_days: int = 5
    extreme_return: float = 0.5  # absolute daily return flagged as suspicious

    @model_validator(mode="after")
    def _check_windows(self) -> DataConfig:
        if not self.start < self.dev_end:
            raise ValueError("start must precede dev_end")
        if not self.dev_end < self.holdout_start:
            raise ValueError("holdout_start must follow dev_end")
        if self.holdout_end is not None and self.holdout_end <= self.holdout_start:
            raise ValueError("holdout_end must follow holdout_start")
        return self


def load_config(path: Path) -> DataConfig:
    raw = yaml.safe_load(path.read_text())
    return DataConfig.model_validate(raw)
