"""Documentation gate checks, run in CI.

Rules:
1. Style: no em-dashes, en-dashes or quotation marks outside code blocks (fenced or indented).
2. A finding in docs is a line tagged [FINDING src=<results path>]. While the REAL-DATA GATE
   is open, no FINDING tag may exist at all.
3. Even with the gate closed, a FINDING must cite an existing result whose sidecar says it
   is not synthetic.
4. Any mention of results/synthetic must sit on a line tagged [SYNTHETIC-ENGINEERING-ONLY]
   and must not be a FINDING.
"""

from __future__ import annotations

import re
from pathlib import Path

from sentinel.results import is_synthetic

GATE_FILE = Path("docs/REAL_DATA_GATE.md")
FINDING = re.compile(r"\[FINDING src=([^\]\s]+)\]")
SYN_TAG = "[SYNTHETIC-ENGINEERING-ONLY]"
BAD_CHARS = ("—", "–", '"', "“", "”")
EXEMPT_DOCS = {"docs/REAL_DATA_GATE.md"}  # may describe the rules using the tag names


def gate_closed(root: Path) -> bool:
    f = root / GATE_FILE
    return f.exists() and re.search(r"^Status:\s*CLOSED\s*$", f.read_text(), re.M) is not None


def doc_files(root: Path) -> list[Path]:
    files = [root / "README.md", root / "COSTS.md", root / "RISK_REGISTER.md"]
    files += sorted((root / "docs").rglob("*.md")) + sorted((root / "paper").rglob("*.md"))
    return [f for f in files if f.exists()]


def check_docs(root: Path) -> list[str]:
    errors: list[str] = []
    closed = gate_closed(root)
    for f in doc_files(root):
        rel = f.relative_to(root).as_posix()
        in_fence = False
        for n, line in enumerate(f.read_text().splitlines(), 1):
            if line.strip().startswith("```"):
                in_fence = not in_fence
                continue
            loc = f"{rel}:{n}"
            code = in_fence or line.startswith("    ")  # fenced or indented code
            if not code and any(c in line for c in BAD_CHARS):
                errors.append(f"{loc}: dash or quotation mark in prose")
            if rel in EXEMPT_DOCS:
                continue
            m = FINDING.search(line)
            if "results/synthetic" in line and SYN_TAG not in line:
                errors.append(f"{loc}: synthetic result mentioned without {SYN_TAG}")
            if m:
                src = m.group(1)
                if not closed:
                    errors.append(f"{loc}: FINDING while the REAL-DATA GATE is open")
                if "results/synthetic" in src or "results/synthetic" in line:
                    errors.append(f"{loc}: synthetic result cited as a finding")
                elif is_synthetic(root / src) is not False:
                    errors.append(f"{loc}: FINDING source {src} is missing or not marked real")
    return errors
