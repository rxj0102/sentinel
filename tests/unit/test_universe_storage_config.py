from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from sentinel.config import load_config
from sentinel.data.storage import frame_hash, read_table, write_table
from sentinel.data.universe import membership_matrix, reconstruct_members_at


def _changes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-02-03", "2020-03-02"]),
            "added": ["C", "D"],
            "removed": ["A", None],
        }
    )


def test_membership_matrix_effective_dates() -> None:
    cal = pd.bdate_range("2020-01-01", "2020-04-01")
    m = membership_matrix({"A", "B"}, pd.Timestamp("2020-01-01"), _changes(), cal)
    assert m.loc["2020-02-03", "A"] is False or not m.loc["2020-02-03", "A"]
    assert m.loc["2020-01-31", "A"]
    assert m.loc["2020-02-03", "C"] and not m.loc["2020-01-31", "C"]
    assert m.loc["2020-04-01", ["B", "C", "D"]].all()


def test_reconstruct_roundtrip() -> None:
    cal = pd.bdate_range("2020-01-01", "2020-04-01")
    m = membership_matrix({"A", "B"}, pd.Timestamp("2020-01-01"), _changes(), cal)
    now = set(m.columns[m.iloc[-1]])
    assert reconstruct_members_at(now, _changes(), pd.Timestamp("2020-01-01")) == {"A", "B"}


def test_storage_manifest_detects_tampering(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1, 2, 3]})
    man = tmp_path / "m.json"
    write_table(df, tmp_path / "t.parquet", man)
    assert read_table(tmp_path / "t.parquet", man).equals(df)
    write_table(pd.DataFrame({"a": [9, 9, 9]}), tmp_path / "t.parquet")
    with pytest.raises(ValueError):
        read_table(tmp_path / "t.parquet", man)
    assert frame_hash(df) == frame_hash(df.copy())


def test_config_windows(tmp_path: Path) -> None:
    good = tmp_path / "c.yaml"
    good.write_text("start: 2005-01-03\ndev_end: 2022-12-30\nholdout_start: 2023-01-03\n")
    assert load_config(good).holdout_end is None
    bad = tmp_path / "b.yaml"
    bad.write_text("start: 2005-01-03\ndev_end: 2022-12-30\nholdout_start: 2022-06-01\n")
    with pytest.raises(ValidationError):
        load_config(bad)
    extra = tmp_path / "e.yaml"
    extra.write_text(good.read_text() + "unknown: 1\n")
    with pytest.raises(ValidationError):
        load_config(extra)
