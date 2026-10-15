import json
from pathlib import Path

from sentinel.logging import configure_logging, get_logger


def test_log_lines_reach_stdout_and_the_file(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "logs" / "run.log"
    configure_logging(log_file=path)
    get_logger("t").info("hello", n=3)
    get_logger("t").error("boom", error_type="ValueError")
    out = capsys.readouterr().out.strip().splitlines()
    filed = path.read_text().strip().splitlines()
    assert out == filed and len(filed) == 2
    first = json.loads(filed[0])
    assert first["event"] == "hello" and first["n"] == 3 and first["level"] == "info"
    assert json.loads(filed[1])["error_type"] == "ValueError"
    configure_logging()  # restore stdout only logging for other tests
