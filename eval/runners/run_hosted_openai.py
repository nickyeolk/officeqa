"""OpenAI Agents SDK runner against a self-hosted OpenAI-compatible endpoint.

WHY THIS HARNESS IS THE BEST FIT FOR NON-NATIVE MODELS
-------------------------------------------------------
vLLM natively exposes the OpenAI /v1/chat/completions wire format, so
``AsyncOpenAI(base_url=..., default_headers=...)`` is a zero-friction drop-in.
``OpenAIChatCompletionsModel`` (the Chat-Completions path, NOT the Responses API)
works with any provider that speaks OpenAI format — no OpenRouter, no shim.

Cloudflare Access auth is injected as persistent HTTP headers on the underlying
httpx client via ``default_headers``; every request carries them automatically.

QWEN3 THINKING MODE
-------------------
Qwen3 models default to "thinking" mode and prepend <think>...</think> blocks
before each reply. extract_final_answer() parses <FINAL_ANSWER> tags regardless,
so this does not break scoring — but it does burn extra tokens per turn.
To disable thinking at the vLLM level pass:
    extra_body={"chat_template_kwargs": {"enable_thinking": False}}
This is done here by default to keep costs comparable to other runners.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_EVAL_DIR = Path(__file__).resolve().parent.parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from agents import (  # type: ignore
    Agent,
    ModelSettings,
    OpenAIChatCompletionsModel,
    Runner,
    function_tool,
    set_tracing_disabled,
)
from openai import AsyncOpenAI  # noqa: E402

from config import MAX_TURNS, get_cf_headers, require_hosted_base_url  # noqa: E402
from prompts import SYSTEM_PROMPT, user_prompt  # noqa: E402
from runners.base import QuestionResult  # noqa: E402
from scoring import extract_final_answer  # noqa: E402
from tools import CallCounter, glob as _glob, grep as _grep, read_file as _read_file  # noqa: E402

set_tracing_disabled(True)


# ---------------------------------------------------------------------------
# Tool definitions — identical to run_openai.py
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Agent builder
# ---------------------------------------------------------------------------


def _build_agent(model: str) -> Agent:
    client = AsyncOpenAI(
        base_url=require_hosted_base_url(),
        # vLLM expects a non-empty api_key; the real auth is the CF headers.
        api_key="dummy",
        default_headers=get_cf_headers(),
    )
    return Agent(
        name="OfficeQA",
        instructions=SYSTEM_PROMPT,
        model=OpenAIChatCompletionsModel(
            model=model,
            openai_client=client,
        ),
        tools=[read_file, glob, grep],
        # Disable Qwen3 thinking mode to keep token usage in line with other
        # runners. Remove this kwarg if you want extended reasoning.
        model_settings=ModelSettings(
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        ),
    )


# ---------------------------------------------------------------------------
# Runner entry-point (called by runners/base.py::run_dataset)
# ---------------------------------------------------------------------------


async def run_question(row: dict, model: str) -> QuestionResult:
    agent = _build_agent(model)
    result = QuestionResult(
        uid=str(row.get("uid", "")),
        question=str(row.get("question", "")),
        ground_truth=str(row.get("answer", "")),
        sdk="hosted_openai",
        model=model,
    )
    t0 = time.perf_counter()
    with CallCounter() as cc:
        run = await Runner.run(agent, user_prompt(result.question), max_turns=MAX_TURNS)
    result.latency_s = time.perf_counter() - t0
    result.tool_calls_by_name = cc.as_dict()
    result.tool_calls = cc.total

    raw = run.final_output or ""
    if not isinstance(raw, str):
        raw = str(raw)
    result.raw_output = raw
    result.predicted = extract_final_answer(raw) or raw

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
