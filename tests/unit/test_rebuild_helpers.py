import pandas as pd

from sentinel.data.rebuild import build_universe, chunked
from sentinel.data.sources import parsers

HTML = """
<table><thead><tr><th>Symbol</th><th>Security</th></tr></thead>
<tbody><tr><td>AAA</td><td>Aaa</td></tr><tr><td>BRK.B</td><td>Berkshire</td></tr></tbody></table>
"""


def test_current_parser_and_yahoo_symbol() -> None:
    assert parsers.parse_sp500_current(HTML) == ["AAA", "BRK.B"]
    assert parsers.to_yahoo_symbol("BRK.B") == "BRK-B"


def test_chunked() -> None:
    assert chunked(["a", "b", "c"], 2) == [["a", "b"], ["c"]]


def test_build_universe_includes_vanished_names() -> None:
    changes = pd.DataFrame(
        {
            "date": pd.to_datetime(["2010-06-01", "2012-01-01"]),
            "added": ["NEW", "NEWER"],
            "removed": ["OLD", "NEW"],
        }
    )
    initial, ever = build_universe(["NEWER", "KEEP"], changes, pd.Timestamp("2010-01-01"))
    assert initial == {"OLD", "KEEP"}
    assert {"OLD", "NEW", "NEWER", "KEEP"} == set(ever)
