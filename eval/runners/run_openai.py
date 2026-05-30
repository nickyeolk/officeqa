"""OpenAI Agents SDK runner routed through OpenRouter.

Uses ``OpenAIChatCompletionsModel`` (rather than the Responses API) so any
OpenRouter-served model works.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure sibling modules import when invoked as a script too
_EVAL_DIR = Path(__file__).resolve().parent.parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from agents import (  # type: ignore  # noqa: E402
    Agent,
    OpenAIChatCompletionsModel,
    Runner,
    function_tool,
    set_tracing_disabled,
)
from openai import AsyncOpenAI  # noqa: E402

from config import MAX_TURNS, OPENROUTER_BASE_URL, require_openrouter_key  # noqa: E402
from prompts import SYSTEM_PROMPT, user_prompt  # noqa: E402
from runners.base import QuestionResult  # noqa: E402
from scoring import extract_final_answer  # noqa: E402
from tools import CallCounter, TOOL_DESCRIPTIONS, glob as _glob, grep as _grep, read_file as _read_file  # noqa: E402

set_tracing_disabled(True)


# --- function tools (the @function_tool decorator infers schema from sigs) ---


@function_tool
def read_file(filename: str, offset: int = 0, limit: int = 2000) -> str:
    """Read a Treasury Bulletin text file by filename (e.g. 'treasury_bulletin_1941_01.txt').
    Returns text lines `offset..offset+limit`. Use offset/limit to page through large files.
    """
    return _read_file(filename, offset=offset, limit=limit)


@function_tool
def glob(pattern: str) -> str:
    """List filenames in the corpus matching a glob pattern, e.g. 'treasury_bulletin_1941_*.txt'."""
    return "\n".join(_glob(pattern))


@function_tool
def grep(
    pattern: str,
    glob_pattern: str = "*.txt",
    max_results: int = 50,
    context: int = 0,
) -> str:
    """Regex search across files matching glob_pattern. Returns '<file>:<line_no>: <line>' hits."""
    return "\n".join(_grep(pattern, glob_pattern=glob_pattern, max_results=max_results, context=context))


# --- runner ---------------------------------------------------------------


def _build_agent(model: str) -> Agent:
    client = AsyncOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=require_openrouter_key(),
    )
    return Agent(
        name="OfficeQA",
        instructions=SYSTEM_PROMPT,
        model=OpenAIChatCompletionsModel(model=model, openai_client=client),
        tools=[read_file, glob, grep],
    )


async def run_question(row: dict, model: str) -> QuestionResult:
    agent = _build_agent(model)
    result = QuestionResult(
        uid=str(row.get("uid", "")),
        question=str(row.get("question", "")),
        ground_truth=str(row.get("answer", "")),
        sdk="openai",
        model=model,
    )
    t0 = time.perf_counter()
    with CallCounter() as cc:
        run = await Runner.run(agent, user_prompt(result.question), max_turns=MAX_TURNS)
    result.latency_s = time.perf_counter() - t0
    result.tool_calls_by_name = cc.as_dict()
    result.tool_calls = cc.total

    # Final assistant text
    raw = run.final_output or ""
    if not isinstance(raw, str):
        raw = str(raw)
    result.raw_output = raw
    result.predicted = extract_final_answer(raw) or raw

    # Per-turn usage and llm-call count from raw_responses
    pt = ct = tt = llm_calls = 0
    for resp in getattr(run, "raw_responses", []) or []:
        usage = getattr(resp, "usage", None)
        if usage is None:
            continue
        llm_calls += 1
        pt += int(getattr(usage, "input_tokens", getattr(usage, "prompt_tokens", 0)) or 0)
        ct += int(getattr(usage, "output_tokens", getattr(usage, "completion_tokens", 0)) or 0)
        tt += int(getattr(usage, "total_tokens", 0) or 0) or (
            int(getattr(usage, "input_tokens", getattr(usage, "prompt_tokens", 0)) or 0)
            + int(getattr(usage, "output_tokens", getattr(usage, "completion_tokens", 0)) or 0)
        )
    result.llm_calls = llm_calls or 1
    result.prompt_tokens = pt
    result.completion_tokens = ct
    result.total_tokens = tt or (pt + ct)
    return result
