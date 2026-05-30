"""Shared evaluation primitives — QuestionResult + retry + JSONL writer.

Each SDK runner returns a ``QuestionResult``; this module orchestrates the
loop, retries, and serialization.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable

# Make sibling modules importable when this file is reached as eval.runners.base
_EVAL_DIR = Path(__file__).resolve().parent.parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from config import RESULTS_DIR, RETRY_BACKOFF_S, RETRY_MAX  # noqa: E402
from scoring import extract_final_answer, score_all_tolerances  # noqa: E402


# ----- result schema ------------------------------------------------------


@dataclass
class QuestionResult:
    uid: str
    question: str
    ground_truth: str
    predicted: str = ""
    raw_output: str = ""
    score_at_0pct: float = 0.0
    score_at_0_1pct: float = 0.0
    score_at_1pct: float = 0.0
    score_at_5pct: float = 0.0
    latency_s: float = 0.0
    llm_calls: int = 0
    tool_calls: int = 0
    tool_calls_by_name: dict[str, int] = field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    sdk: str = ""
    model: str = ""
    error: str | None = None
    stopped_reason: str | None = None

    def fill_scores(self) -> None:
        if not self.ground_truth:
            return
        scores = score_all_tolerances(self.ground_truth, self.predicted)
        for k, v in scores.items():
            setattr(self, k, v)


# A single runner is an async callable: (row, model) -> QuestionResult.
RunFn = Callable[[dict, str], Awaitable[QuestionResult]]


# ----- retry --------------------------------------------------------------


async def with_retries(coro_factory: Callable[[], Awaitable[QuestionResult]]) -> QuestionResult:
    """Run a runner coroutine with bounded exponential backoff on transient errors."""
    last_exc: Exception | None = None
    for attempt in range(RETRY_MAX):
        try:
            return await coro_factory()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            msg = str(exc).lower()
            transient = (
                "rate limit" in msg
                or "429" in msg
                or "timeout" in msg
                or "temporarily" in msg
                or "5xx" in msg
                or "503" in msg
                or "502" in msg
                or "overloaded" in msg
            )
            if not transient or attempt == RETRY_MAX - 1:
                break
            sleep_s = RETRY_BACKOFF_S * (2 ** attempt)
            await asyncio.sleep(sleep_s)
    assert last_exc is not None
    raise last_exc


# ----- driver -------------------------------------------------------------


def make_results_path(sdk: str, model: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_model = model.replace("/", "_").replace(":", "_")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return RESULTS_DIR / f"{sdk}_{safe_model}_{stamp}.jsonl"


def load_done_uids(path: Path) -> set[str]:
    """Read an existing JSONL and return the set of uids that ran without error."""
    done: set[str] = set()
    if not path.is_file():
        return done
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid = row.get("uid")
            if uid:
                done.add(str(uid))
    return done


async def run_dataset(
    runner: RunFn,
    rows: list[dict],
    sdk: str,
    model: str,
    concurrency: int = 1,
    output_path: Path | None = None,
    append: bool = False,
) -> Path:
    """Run ``runner`` over every row, write JSONL to ``output_path`` (or auto-named).

    Set ``append=True`` together with ``output_path=<existing file>`` to resume:
    rows already present in the file are expected to have been filtered out by
    the caller.
    """
    out = output_path or make_results_path(sdk, model)
    print(f"{'Appending to' if append else 'Writing'} results: {out}")
    sem = asyncio.Semaphore(max(1, concurrency))
    counters = {"done": 0, "ok": 0, "err": 0}

    async def worker(row: dict) -> QuestionResult:
        async with sem:
            t0 = time.perf_counter()
            try:
                result = await with_retries(lambda: runner(row, model))
                result.fill_scores()
            except Exception as exc:  # noqa: BLE001
                result = QuestionResult(
                    uid=str(row.get("uid", "")),
                    question=str(row.get("question", "")),
                    ground_truth=str(row.get("answer", "")),
                    sdk=sdk,
                    model=model,
                    latency_s=time.perf_counter() - t0,
                    error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}",
                    stopped_reason="hard_error",
                )
            counters["done"] += 1
            if result.error:
                counters["err"] += 1
            else:
                counters["ok"] += 1
            print(
                f"  [{counters['done']:>3}/{len(rows)}] uid={result.uid} "
                f"score@1%={result.score_at_1pct:.0f} "
                f"lat={result.latency_s:.1f}s "
                f"tc={result.tool_calls} llm={result.llm_calls} "
                f"tok={result.total_tokens}"
                + (f"  ERROR: {result.error.splitlines()[0]}" if result.error else "")
            )
            return result

    # Run workers, write results as they complete (append to JSONL).
    tasks = [asyncio.create_task(worker(r)) for r in rows]
    mode = "a" if append else "w"
    with out.open(mode, encoding="utf-8") as fh:
        for fut in asyncio.as_completed(tasks):
            result = await fut
            fh.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
            fh.flush()

    print(f"\nDone. ok={counters['ok']} err={counters['err']} -> {out}")
    return out
