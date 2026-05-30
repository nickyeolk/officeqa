"""Multi-tolerance scoring wrapper around the project's reward.py."""

from __future__ import annotations

import sys
from pathlib import Path

from config import PROJECT_ROOT

# Make the project's reward.py importable
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import reward  # noqa: E402  (parent module)


TOLERANCES: dict[str, float] = {
    "score_at_0pct": 0.0,
    "score_at_0_1pct": 0.001,
    "score_at_1pct": 0.01,
    "score_at_5pct": 0.05,
}


def score_all_tolerances(ground_truth: str, predicted: str) -> dict[str, float]:
    """Return {tolerance_label: 1.0|0.0} for every tolerance in TOLERANCES."""
    out: dict[str, float] = {}
    for label, tol in TOLERANCES.items():
        try:
            out[label] = reward.score_answer(ground_truth, predicted, tolerance=tol)
        except Exception:  # noqa: BLE001
            out[label] = 0.0
    return out


def extract_final_answer(raw: str) -> str:
    """Pull the text inside the last <FINAL_ANSWER>...</FINAL_ANSWER> from raw."""
    final, _ = reward.extract_final_answer_from_xml(raw or "")
    return final
