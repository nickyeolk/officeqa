"""Microsoft Agent Framework runner against a self-hosted OpenAI-compatible endpoint.

WHY THIS HARNESS WORKS WELL FOR NON-NATIVE MODELS
--------------------------------------------------
``OpenAIChatCompletionClient`` is a thin wrapper over the standard OpenAI
AsyncClient. Any kwargs not consumed by the wrapper are forwarded straight
to ``openai.AsyncOpenAI``, including ``default_headers`` — which is how we
inject the Cloudflare Access service-token credentials transparently.

QWEN3 THINKING MODE
-------------------
Thinking mode is disabled via ``model_capabilities`` / ``extra_body`` so that
token counts are comparable to other runners. The <think> blocks are harmless
for scoring but inflate context and latency significantly.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Annotated

_EVAL_DIR = Path(__file__).resolve().parent.parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from agent_framework import tool  # type: ignore  # noqa: E402
from agent_framework.openai import OpenAIChatCompletionClient  # type: ignore  # noqa: E402
from pydantic import Field  # noqa: E402

from config import MAX_TURNS, get_cf_headers, require_hosted_base_url  # noqa: E402
from prompts import SYSTEM_PROMPT, user_prompt  # noqa: E402
from runners.base import QuestionResult  # noqa: E402
from scoring import extract_final_answer  # noqa: E402
from tools import CallCounter, glob as _glob, grep as _grep, read_file as _read_file  # noqa: E402


# ---------------------------------------------------------------------------
# Tool definitions — mirrors run_microsoft.py exactly
# ---------------------------------------------------------------------------


@tool(description="Read a Treasury Bulletin text file by filename (e.g. 'treasury_bulletin_1941_01.txt').")
def read_file(
    filename: Annotated[str, Field(description="Basename of the file, e.g. 'treasury_bulletin_1941_01.txt'.")],
    offset: Annotated[int, Field(description="Line offset to start reading from.")] = 0,
    limit: Annotated[int, Field(description="Max number of lines to return.")] = 2000,
) -> str:
    return _read_file(filename, offset=offset, limit=limit)


@tool(description="List corpus filenames matching a glob pattern, e.g. 'treasury_bulletin_1941_*.txt'.")
def glob(
    pattern: Annotated[str, Field(description="Glob pattern, e.g. 'treasury_bulletin_1941_*.txt' or '*.txt'.")],
) -> str:
    return "\n".join(_glob(pattern))


@tool(description="Regex search across files matching glob_pattern. Returns '<file>:<line>: <text>' hits.")
def grep(
    pattern: Annotated[str, Field(description="Regex to search for.")],
    glob_pattern: Annotated[str, Field(description="Glob pattern restricting which files to search.")] = "*.txt",
    max_results: Annotated[int, Field(description="Maximum number of hits to return.")] = 50,
    context: Annotated[int, Field(description="Lines of context around each hit.")] = 0,
) -> str:
    return "\n".join(_grep(pattern, glob_pattern=glob_pattern, max_results=max_results, context=context))


# ---------------------------------------------------------------------------
# Agent builder
# ---------------------------------------------------------------------------


def _build_agent(model: str):
    cf_headers = get_cf_headers()
    # OpenAIChatCompletionClient wraps openai.AsyncOpenAI and accepts the same
    # connection params. default_headers is forwarded to every HTTP request,
    # which is how Cloudflare Access credentials are injected.
    # Thinking is disabled server-side via vLLM --default-chat-template-kwargs.
    # Sampling params from Qwen3.6-35B-A3B model card (non-thinking/instruct mode).
    client = OpenAIChatCompletionClient(
        model=model,
        # vLLM requires a non-empty key; real auth is done by the CF headers.
        api_key="dummy",
        base_url=require_hosted_base_url(),
        default_headers=cf_headers,
        temperature=0.7,
        top_p=0.8,
        max_tokens=8192,
    )
    return client.as_agent(
        instructions=SYSTEM_PROMPT,
        tools=[read_file, glob, grep],
    )


# ---------------------------------------------------------------------------
# Runner entry-point
# ---------------------------------------------------------------------------


async def run_question(row: dict, model: str) -> QuestionResult:
    agent = _build_agent(model)
    result = QuestionResult(
        uid=str(row.get("uid", "")),
        question=str(row.get("question", "")),
        ground_truth=str(row.get("answer", "")),
        sdk="hosted_microsoft",
        model=model,
    )
    t0 = time.perf_counter()
    with CallCounter() as cc:
        response = await agent.run(user_prompt(result.question))
    result.latency_s = time.perf_counter() - t0
    result.tool_calls_by_name = cc.as_dict()
    result.tool_calls = cc.total

    # Extract final text — same multi-path extraction as run_microsoft.py
    raw = ""
    try:
        raw = getattr(response, "text", None) or ""
    except Exception:
        pass
    if not raw:
        try:
            raw = "\n".join(
                getattr(c, "text", "")
                for msg in getattr(response, "messages", [])
                for c in getattr(msg, "contents", [])
                if getattr(c, "text", "")
            )
        except Exception:
            raw = str(response)
    result.raw_output = raw
    result.predicted = extract_final_answer(raw) or raw

    # Usage extraction (mirrors run_microsoft.py)
    def _read_usage(obj) -> tuple[int, int, int]:
        if obj is None:
            return 0, 0, 0
        if isinstance(obj, dict):
            pt = int(obj.get("input_token_count") or obj.get("prompt_tokens") or 0)
            ct = int(obj.get("output_token_count") or obj.get("completion_tokens") or 0)
            tt = int(obj.get("total_token_count") or obj.get("total_tokens") or 0)
            return pt, ct, tt
        pt = int(getattr(obj, "input_token_count", 0) or getattr(obj, "prompt_tokens", 0) or 0)
        ct = int(getattr(obj, "output_token_count", 0) or getattr(obj, "completion_tokens", 0) or 0)
        tt = int(getattr(obj, "total_token_count", 0) or getattr(obj, "total_tokens", 0) or 0)
        return pt, ct, tt

    pt = ct = tt = llm_calls = 0
    for msg in getattr(response, "messages", []) or []:
        role = getattr(msg, "role", None) or getattr(msg, "author_role", None)
        if str(role).lower() in ("assistant", "agent"):
            llm_calls += 1
        for attr in ("usage_details", "usage"):
            ppt, cct, ttt = _read_usage(getattr(msg, attr, None))
            pt += ppt; ct += cct; tt += ttt
    if tt == 0:
        candidates = []
        if (v := getattr(response, "value", None)) is not None:
            candidates.append(v)
        for cr in getattr(response, "chat_responses", []) or []:
            candidates.append(cr)
        for cand in candidates:
            for attr in ("usage_details", "usage"):
                ppt, cct, ttt = _read_usage(getattr(cand, attr, None))
                pt += ppt; ct += cct; tt += ttt
    if tt == 0:
        for attr in ("usage_details", "usage"):
            ppt, cct, ttt = _read_usage(getattr(response, attr, None))
            pt += ppt; ct += cct; tt += ttt

    result.llm_calls = llm_calls or 1
    result.prompt_tokens = pt
    result.completion_tokens = ct
    result.total_tokens = tt or (pt + ct)
    return result
