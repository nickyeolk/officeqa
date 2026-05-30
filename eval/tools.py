"""Shared read-only file tools — identical semantics across all four SDKs.

Three tools, named in snake_case (MCP / OpenAI / Google ADK convention):

    read_file(filename, offset=0, limit=2000) -> str
    glob(pattern) -> list[str]
    grep(pattern, glob_pattern="*.txt", max_results=50, context=0) -> list[str]

Each call increments a contextvar-scoped counter so the per-question runner
can report ``tool_calls_by_name`` even if the SDK does not expose it.
"""

from __future__ import annotations

import contextvars
import fnmatch
import re
from collections import Counter
from pathlib import Path

from config import CORPUS_DIR
from corpus import safe_path

# A contextvar so concurrent runs don't cross-contaminate counters.
_call_counter: contextvars.ContextVar[Counter[str] | None] = contextvars.ContextVar(
    "tool_call_counter", default=None
)


class CallCounter:
    """Context manager that captures per-tool invocation counts."""

    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self._token: contextvars.Token | None = None

    def __enter__(self) -> "CallCounter":
        self._token = _call_counter.set(self.counts)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._token is not None:
            _call_counter.reset(self._token)
            self._token = None

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def as_dict(self) -> dict[str, int]:
        return dict(self.counts)


def _tally(name: str) -> None:
    counter = _call_counter.get()
    if counter is not None:
        counter[name] += 1


# --- The three tools -----------------------------------------------------


def read_file(filename: str, offset: int = 0, limit: int = 2000) -> str:
    """Read a Treasury Bulletin .txt file by basename.

    Returns lines ``offset..offset+limit``. Appends ``[truncated, N more lines]``
    when more lines remain. Rejects path traversal.
    """
    _tally("read_file")
    try:
        path = safe_path(filename)
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {exc}"
    if not path.is_file():
        return f"ERROR: file not found: {filename}"

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    total = len(lines)
    start = max(0, int(offset))
    end = min(total, start + max(1, int(limit)))
    body = "".join(lines[start:end])
    remaining = total - end
    if remaining > 0:
        body += f"\n[truncated, {remaining} more lines; call read_file with offset={end} to continue]\n"
    if start > 0:
        body = f"[showing lines {start}..{end} of {total}]\n" + body
    return body


def glob(pattern: str) -> list[str]:
    """Return corpus filenames matching ``pattern`` (e.g. ``treasury_bulletin_1941_*.txt``).

    Patterns are matched against basenames only — agents do not see paths.
    """
    _tally("glob")
    if not pattern:
        return []
    # Use fnmatch over the listing rather than Path.glob to keep the API
    # strictly basename-only (no leaking directories).
    names = sorted(p.name for p in CORPUS_DIR.iterdir() if p.is_file())
    return [n for n in names if fnmatch.fnmatch(n, pattern)]


def grep(
    pattern: str,
    glob_pattern: str = "*.txt",
    max_results: int = 50,
    context: int = 0,
) -> list[str]:
    """Regex search across files matching ``glob_pattern``.

    Returns up to ``max_results`` hits as ``"<filename>:<line_no>: <line>"``.
    ``context`` includes that many lines before and after each match.
    """
    _tally("grep")
    try:
        rx = re.compile(pattern)
    except re.error as exc:
        return [f"ERROR: invalid regex: {exc}"]
    if max_results <= 0:
        return []
    files = glob(glob_pattern) if glob_pattern else [p.name for p in CORPUS_DIR.iterdir() if p.is_file()]
    # ``glob`` already tallies; we don't want grep to inflate the count by
    # one each time. Decrement the over-count.
    counter = _call_counter.get()
    if counter is not None and glob_pattern:
        counter["glob"] = max(0, counter["glob"] - 1)

    hits: list[str] = []
    ctx = max(0, int(context))
    for name in files:
        path = CORPUS_DIR / name
        try:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                lines = fh.readlines()
        except OSError:
            continue
        for i, line in enumerate(lines):
            if rx.search(line):
                if ctx == 0:
                    hits.append(f"{name}:{i + 1}: {line.rstrip()}")
                else:
                    lo = max(0, i - ctx)
                    hi = min(len(lines), i + ctx + 1)
                    block = []
                    for j in range(lo, hi):
                        sep = "=" if j == i else "-"
                        block.append(f"{name}:{j + 1}{sep} {lines[j].rstrip()}")
                    hits.append("\n".join(block))
                if len(hits) >= max_results:
                    return hits
    return hits


# --- Public schemas (used by SDK adapters) -------------------------------

TOOL_DESCRIPTIONS: dict[str, str] = {
    "read_file": (
        "Read a Treasury Bulletin text file by filename. "
        "Files are named like 'treasury_bulletin_YYYY_MM.txt' (e.g., 'treasury_bulletin_1941_01.txt'). "
        "Use 'offset' and 'limit' (default 2000 lines) to page through large files."
    ),
    "glob": (
        "List filenames in the corpus matching a glob pattern, e.g. "
        "'treasury_bulletin_1941_*.txt' or '*.txt' for all files. "
        "Use this to discover which year/month files exist."
    ),
    "grep": (
        "Regex search for a pattern across files matching glob_pattern (default '*.txt'). "
        "Returns up to max_results (default 50) hits as '<filename>:<line_no>: <line>'. "
        "Set 'context' > 0 to include surrounding lines."
    ),
}
