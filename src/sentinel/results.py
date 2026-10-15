"""Result writer. Every output is labeled synthetic or real in a sidecar file.

Synthetic outputs go under results/synthetic and can never be cited as findings in docs
(enforced by sentinel.docs_check in CI).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

SYNTHETIC_DIR = "synthetic"


def result_dir(root: Path, synthetic: bool) -> Path:
    return root / SYNTHETIC_DIR if synthetic else root


def write_result(
    root: Path, name: str, payload: pd.DataFrame | dict[str, Any], synthetic: bool
) -> Path:
    """Write a CSV (frame) or JSON (dict) plus a sidecar <name>.meta.json."""
    out_dir = result_dir(root, synthetic)
    out_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, pd.DataFrame):
        path = out_dir / f"{name}.csv"
        payload.to_csv(path, index=False)
    else:
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    meta = {"synthetic": synthetic, "label": "SYNTHETIC, engineering only" if synthetic else "real"}
    (out_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=2))
    return path


def write_text_result(root: Path, name: str, text: str, synthetic: bool) -> Path:
    out_dir = result_dir(root, synthetic)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.md"
    path.write_text(text)
    meta = {"synthetic": synthetic, "label": "SYNTHETIC, engineering only" if synthetic else "real"}
    (out_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=2))
    return path


def is_synthetic(result_path: Path) -> bool | None:
    """True or False from the sidecar, None when the sidecar is missing."""
    meta = result_path.with_name(result_path.stem + ".meta.json")
    if not meta.exists():
        return None
    return bool(json.loads(meta.read_text())["synthetic"])
