"""Aggregate results/*.jsonl into a single accuracy + performance table."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from tabulate import tabulate

from config import RESULTS_DIR


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def discover(files: Iterable[str] | None) -> list[Path]:
    if files:
        return [Path(f) for f in files]
    return sorted(RESULTS_DIR.glob("*.jsonl"))


def mean(xs: list[float]) -> float:
    return statistics.mean(xs) if xs else 0.0


def pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    return statistics.quantiles(xs, n=100)[int(q) - 1] if len(xs) >= 2 else xs[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", help="Specific JSONL files (default: all in results/)")
    args = parser.parse_args()

    paths = discover(args.files)
    if not paths:
        print(f"No JSONL files found under {RESULTS_DIR}")
        return 1

    # Group rows by (sdk, model) — pulled from row metadata so combining files
    # across multiple runs Just Works.
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for p in paths:
        for row in load_jsonl(p):
            sdk = row.get("sdk", "?")
            model = row.get("model", "?")
            groups[(sdk, model)].append(row)

    header = [
        "SDK", "Model", "N", "Errors",
        "Acc@0%", "Acc@0.1%", "Acc@1%", "Acc@5%",
        "Lat(s)", "p95 Lat", "Tokens", "ToolCalls", "LLMCalls",
    ]
    table: list[list] = []
    for (sdk, model), rows in sorted(groups.items()):
        ok = [r for r in rows if not r.get("error")]
        n = len(rows)
        errs = sum(1 for r in rows if r.get("error"))
        if not ok:
            table.append([sdk, model, n, errs, *([0.0] * 4), *([0.0] * 5)])
            continue
        accs = {k: mean([r.get(k, 0.0) for r in ok]) for k in
                ("score_at_0pct", "score_at_0_1pct", "score_at_1pct", "score_at_5pct")}
        lats = [r.get("latency_s", 0.0) for r in ok]
        tokens = mean([r.get("total_tokens", 0) for r in ok])
        tcalls = mean([r.get("tool_calls", 0) for r in ok])
        lcalls = mean([r.get("llm_calls", 0) for r in ok])
        table.append([
            sdk, model, n, errs,
            f"{accs['score_at_0pct']:.3f}",
            f"{accs['score_at_0_1pct']:.3f}",
            f"{accs['score_at_1pct']:.3f}",
            f"{accs['score_at_5pct']:.3f}",
            f"{mean(lats):.1f}",
            f"{pct(lats, 95):.1f}",
            f"{tokens:,.0f}",
            f"{tcalls:.1f}",
            f"{lcalls:.1f}",
        ])

    print(tabulate(table, headers=header, tablefmt="github"))
    print(f"\nAggregated from {len(paths)} file(s) under {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
