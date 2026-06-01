"""Google ADK runner against a self-hosted OpenAI-compatible endpoint via LiteLLM.

WHY THIS HARNESS WORKS FOR NON-NATIVE MODELS
--------------------------------------------
Google ADK does not speak OpenAI format natively — it uses its own model
abstraction. However ADK ships ``LiteLlm``, which delegates to the LiteLLM
library. LiteLLM is a model-agnostic router that supports any OpenAI-compatible
backend via the ``openai/<model>`` prefix together with ``api_base``.

Cloudflare Access headers are passed through LiteLLM's ``extra_headers`` kwarg,
which is forwarded verbatim on every HTTP request to the upstream endpoint.

QWEN3 THINKING MODE
-------------------
Thinking mode (``<think>...</think>`` blocks) is disabled via LiteLLM's
``extra_body`` passthrough so token counts stay comparable to other runners.
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path

_EVAL_DIR = Path(__file__).resolve().parent.parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from google.adk.agents import LlmAgent  # type: ignore  # noqa: E402
from google.adk.models.lite_llm import LiteLlm  # type: ignore  # noqa: E402
from google.adk.runners import Runner  # type: ignore  # noqa: E402
from google.adk.sessions import InMemorySessionService  # type: ignore  # noqa: E402
from google.adk.tools import FunctionTool  # type: ignore  # noqa: E402
from google.genai import types as genai_types  # type: ignore  # noqa: E402

from config import MAX_TURNS, get_cf_headers, require_hosted_base_url  # noqa: E402
from prompts import SYSTEM_PROMPT, user_prompt  # noqa: E402
from runners.base import QuestionResult  # noqa: E402
from scoring import extract_final_answer  # noqa: E402
from tools import CallCounter, glob as _glob, grep as _grep, read_file as _read_file  # noqa: E402

APP_NAME = "officeqa-eval-hosted"
USER_ID = "evaluator"


# ---------------------------------------------------------------------------
# Tool definitions — identical to run_google.py (plain functions with docstrings)
# ---------------------------------------------------------------------------


def read_file(filename: str, offset: int = 0, limit: int = 2000) -> str:
    """Read a Treasury Bulletin text file by filename (e.g. 'treasury_bulletin_1941_01.txt').

    Args:
        filename: Basename of the file.
        offset: Line offset to start reading from.
        limit: Max number of lines to return (default 2000).

    Returns:
        The text content of the requested slice.
    """
    return _read_file(filename, offset=offset, limit=limit)


def glob(pattern: str) -> list[str]:
    """List filenames in the corpus matching a glob pattern.

    Args:
        pattern: e.g. 'treasury_bulletin_1941_*.txt' or '*.txt'.

    Returns:
        Sorted list of matching basenames.
    """
    return _glob(pattern)


def grep(pattern: str, glob_pattern: str = "*.txt", max_results: int = 50, context: int = 0) -> list[str]:
    """Regex search for ``pattern`` across files matching ``glob_pattern``.

    Args:
        pattern: Regular expression.
        glob_pattern: Restrict files by glob.
        max_results: Cap on number of hits returned.
        context: Lines of context around each hit.

    Returns:
        '<filename>:<line_no>: <line>' formatted hits.
    """
    return _grep(pattern, glob_pattern=glob_pattern, max_results=max_results, context=context)


# ---------------------------------------------------------------------------
# ADK runner builder
# ---------------------------------------------------------------------------


def _build_runner(model: str) -> Runner:
    base_url = require_hosted_base_url()
    cf_headers = get_cf_headers()

    # LiteLLM routes to OpenAI-compatible backends via the "openai/" prefix.
    # api_base overrides the endpoint; api_key satisfies vLLM's auth check
    # (the real auth is the Cloudflare headers).
    # extra_headers are forwarded verbatim on every HTTP call.
    # extra_body disables Qwen3 thinking mode to reduce token usage.
    llm = LiteLlm(
        model=f"openai/{model}",
        api_base=base_url,
        api_key="dummy",
        extra_headers=cf_headers,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    agent = LlmAgent(
        name="OfficeQA",
        model=llm,
        instruction=SYSTEM_PROMPT,
        tools=[
            FunctionTool(read_file),
            FunctionTool(glob),
            FunctionTool(grep),
        ],
    )
    session_service = InMemorySessionService()
    return Runner(agent=agent, app_name=APP_NAME, session_service=session_service)


# ---------------------------------------------------------------------------
# Runner entry-point
# ---------------------------------------------------------------------------


async def run_question(row: dict, model: str) -> QuestionResult:
    runner = _build_runner(model)
    session_id = f"q-{uuid.uuid4().hex[:8]}"
    await runner.session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=session_id
    )

    result = QuestionResult(
        uid=str(row.get("uid", "")),
        question=str(row.get("question", "")),
        ground_truth=str(row.get("answer", "")),
        sdk="hosted_google",
        model=model,
    )
    msg = genai_types.Content(role="user", parts=[genai_types.Part(text=user_prompt(result.question))])

    raw_parts: list[str] = []
    llm_calls = 0
    pt = ct = tt = 0
    t0 = time.perf_counter()
    with CallCounter() as cc:
        async for event in runner.run_async(user_id=USER_ID, session_id=session_id, new_message=msg):
            if llm_calls >= MAX_TURNS:
                result.stopped_reason = "max_turns_exceeded"
                break
            content = getattr(event, "content", None)
            if content is None:
                continue
            for part in getattr(content, "parts", []) or []:
                text = getattr(part, "text", None)
                if text:
                    raw_parts.append(text)
            usage = getattr(event, "usage_metadata", None)
            if usage is not None:
                llm_calls += 1
                pt += int(getattr(usage, "prompt_token_count", 0) or 0)
                ct += int(getattr(usage, "candidates_token_count", 0) or 0)
                tt += int(getattr(usage, "total_token_count", 0) or 0)
    result.latency_s = time.perf_counter() - t0
    result.tool_calls_by_name = cc.as_dict()
    result.tool_calls = cc.total

    raw = "\n".join(raw_parts)
    result.raw_output = raw
    result.predicted = extract_final_answer(raw) or raw
    result.llm_calls = llm_calls or 1
    result.prompt_tokens = pt
    result.completion_tokens = ct
    result.total_tokens = tt or (pt + ct)
    return result
