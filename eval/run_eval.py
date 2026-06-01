"""CLI: run one SDK runner over the OfficeQA Pro CSV and write results JSONL.

Usage:
    python run_eval.py --sdk claude [--limit 3] [--concurrency 1] [--model ...]
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import sys
from pathlib import Path

import pandas as pd

from config import CSV_FULL, CSV_PRO, DEFAULT_MODELS
from corpus import assert_corpus_present
from runners.base import load_done_uids, run_dataset

SDK_MODULES = {
    "claude": "runners.run_claude",
    "openai": "runners.run_openai",
    "google": "runners.run_google",
    "microsoft": "runners.run_microsoft",
    # Self-hosted OpenAI-compatible endpoint (vLLM / LiteLLM / Cloudflare)
    "hosted_openai": "runners.run_hosted_openai",
    "hosted_microsoft": "runners.run_hosted_microsoft",
    "hosted_google": "runners.run_hosted_google",
}


def load_rows(subset: str, limit: int | None) -> list[dict]:
    path = CSV_PRO if subset == "pro" else CSV_FULL
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found — download the benchmark CSV first.")
    df = pd.read_csv(path)
    if limit is not None and limit > 0:
        df = df.head(limit)
    return df.to_dict(orient="records")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdk", required=True, choices=sorted(SDK_MODULES))
    parser.add_argument("--subset", default="pro", choices=["pro", "full"])
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N questions")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--model", default=None, help="Override default model for this SDK")
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to an existing results JSONL. Already-present uids are skipped and new rows are appended.",
    )
    args = parser.parse_args()

    assert_corpus_present()
    model = args.model or DEFAULT_MODELS[args.sdk]
    print(f"SDK: {args.sdk}    model: {model}    subset: {args.subset}    "
          f"limit: {args.limit}    concurrency: {args.concurrency}")

    rows = load_rows(args.subset, args.limit)
    print(f"Loaded {len(rows)} question(s).")

    output_path = None
    append = False
    if args.resume:
        from pathlib import Path
        output_path = Path(args.resume)
        done = load_done_uids(output_path)
        before = len(rows)
        rows = [r for r in rows if str(r.get("uid", "")) not in done]
        print(f"Resume mode: {len(done)} uid(s) already in {output_path.name}; "
              f"skipping them and processing the remaining {len(rows)} (was {before}).")
        append = True

    module = importlib.import_module(SDK_MODULES[args.sdk])
    runner = module.run_question  # type: ignore[attr-defined]

    out = asyncio.run(
        run_dataset(
            runner=runner,
            rows=rows,
            sdk=args.sdk,
            model=model,
            concurrency=args.concurrency,
            output_path=output_path,
            append=append,
        )
    )
    print(f"\nResults: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
