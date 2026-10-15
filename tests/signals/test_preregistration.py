"""The code registry must match the locked table in docs/hypotheses.md."""

import inspect
from pathlib import Path

from sentinel.signals.library import REGISTRY

DOC = Path(__file__).resolve().parents[2] / "docs" / "hypotheses.md"
HORIZONS = (1, 5, 10, 21, 63)


def _rows() -> dict[str, tuple[dict[str, int], list[int]]]:
    out = {}
    for line in DOC.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 7 and cells[0] in REGISTRY:
            params = {k: int(v) for k, v in (kv.split("=") for kv in cells[1].split())}
            signs = [1 if c == "+" else -1 for c in cells[2:]]
            out[cells[0]] = (params, signs)
    return out


def test_every_signal_is_pre_registered() -> None:
    assert set(_rows()) == set(REGISTRY)


def test_signs_match_locked_table() -> None:
    for name, (_, signs) in _rows().items():
        spec = REGISTRY[name][1]
        assert [spec.sign_at(h) for h in HORIZONS] == signs, name
        assert spec.expected_sign == spec.sign_at(21)


def test_default_parameters_match_locked_table() -> None:
    for name, (params, _) in _rows().items():
        fn = REGISTRY[name][0]
        defaults = {
            k: v.default for k, v in inspect.signature(fn).parameters.items() if k != "panel"
        }
        assert defaults == params, name


def test_overhang_is_marked_lower_confidence() -> None:
    spec = REGISTRY["overhang"][1]
    assert spec.confidence == "lower" and "shares outstanding" in spec.confidence_note
    assert all(REGISTRY[n][1].confidence == "normal" for n in REGISTRY if n != "overhang")
