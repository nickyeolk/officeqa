"""Corpus path helpers with path-traversal protection."""

from __future__ import annotations

from pathlib import Path

from config import CORPUS_DIR


class CorpusError(ValueError):
    """Raised when a filename is unsafe or missing."""


def safe_path(filename: str) -> Path:
    """Resolve ``filename`` inside CORPUS_DIR, rejecting traversal."""
    if not filename:
        raise CorpusError("filename is empty")
    # Reject any directory component; agents only get basenames.
    if "/" in filename or "\\" in filename or ".." in filename:
        raise CorpusError(f"filename must be a basename, got: {filename!r}")
    p = (CORPUS_DIR / filename).resolve()
    if CORPUS_DIR not in p.parents and p != CORPUS_DIR:
        raise CorpusError(f"path escapes corpus dir: {filename!r}")
    return p


def assert_corpus_present() -> None:
    if not CORPUS_DIR.is_dir():
        raise CorpusError(
            f"Corpus directory not found: {CORPUS_DIR}. "
            "Run the snapshot_download step from the project README first."
        )
    if not any(CORPUS_DIR.glob("*.txt")):
        raise CorpusError(f"No .txt files in {CORPUS_DIR}")
