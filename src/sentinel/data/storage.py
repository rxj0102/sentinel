"""Parquet storage with a content-hash manifest so rebuilds can be verified."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


def frame_hash(df: pd.DataFrame) -> str:
    """Order-insensitive-to-index content hash of a frame."""
    h = hashlib.sha256()
    h.update(",".join(map(str, df.columns)).encode())
    h.update(
        pd.util.hash_pandas_object(df.reset_index(drop=True), index=False).to_numpy().tobytes()
    )
    return h.hexdigest()


def write_table(df: pd.DataFrame, path: Path, manifest: Path | None = None) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    digest = frame_hash(df)
    if manifest is not None:
        entries: dict[str, dict[str, object]] = {}
        if manifest.exists():
            entries = json.loads(manifest.read_text())
        entries[path.name] = {"rows": int(len(df)), "sha256": digest}
        manifest.write_text(json.dumps(entries, indent=2, sort_keys=True))
    return digest


def read_table(path: Path, manifest: Path | None = None) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if manifest is not None and manifest.exists():
        expected = json.loads(manifest.read_text()).get(path.name)
        if expected is not None and expected["sha256"] != frame_hash(df):
            raise ValueError(f"{path.name} does not match its manifest hash")
    return df
