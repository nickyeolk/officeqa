"""Google ADK runner routed through OpenRouter via LiteLLM.

Google ADK uses its own model abstraction; ``LiteLlm`` is the supported bridge
to non-Vertex providers. We pass ``model="openrouter/<model_id>"`` so LiteLLM
dispatches to the OpenRouter provider.
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path

# Sibling imports
_EVAL_DIR = Path(__file__).resolve().parent.parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from google.adk.agents import LlmAgent  # type: ignore  # noqa: E402
from google.adk.models.lite_llm import LiteLlm  # type: ignore  # noqa: E402
from google.adk.runners import Runner  # type: ignore  # noqa: E402
from google.adk.sessions import InMemorySessionService  # type: ignore  # noqa: E402
from google.adk.tools import FunctionTool  # type: ignore  # noqa: E402
from google.genai import types as genai_types  # type: ignore  # noqa: E402

from config import MAX_TURNS, require_openrouter_key  # noqa: E402
from prompts import SYSTEM_PROMPT, user_prompt  # noqa: E402
from runners.base import QuestionResult  # noqa: E402
from scoring import extract_final_answer  # noqa: E402
from tools import CallCounter, glob as _glob, grep as _grep, read_file as _read_file  # noqa: E402

APP_NAME = "officeqa-eval"
USER_ID = "evaluator"


# Plain-python tool functions; ADK FunctionTool reads docstrings & signatures.


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


def _build_runner(model: str) -> Runner:
    # LiteLLM picks the OpenRouter provider from the 'openrouter/' prefix.
    # Set the env var LiteLLM reads for the API key.
    os.environ["OPENROUTER_API_KEY"] = require_openrouter_key()
    llm = LiteLlm(model=f"openrouter/{model}")
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
        sdk="google",
        model=model,
    )
    msg = genai_types.Content(role="user", parts=[genai_types.Part(text=user_prompt(result.question))])

    raw_parts: list[str] = []
    llm_calls = 0
    pt = ct = tt = 0
    t0 = time.perf_counter()
    with CallCounter() as cc:
        async for event in runner.run_async(user_id=USER_ID, session_id=session_id, new_message=msg):
            # Honour MAX_TURNS as a strict cap on LLM invocations. Previously
            # this was `turn > MAX_TURNS * 4` counting ADK events, which never
            # tripped — letting Gemini run 60+ turns/Q at huge token cost.
            if llm_calls >= MAX_TURNS:
                result.stopped_reason = "max_turns_exceeded"
                break
            content = getattr(event, "content", None)
            if content is None:
                continue
            # Collect every text part the model emits. Some Gemini parts have
            # `thought=True` (reasoning blocks); we keep them in raw_output so
            # extract_final_answer() can still find the <FINAL_ANSWER> tag.
            # Filtering by `thought` proved unreliable — flash sometimes never
            # produced a non-thought block and we ended up with empty preds.
            for part in getattr(content, "parts", []) or []:
                text = getattr(part, "text", None)
                if text:
                    raw_parts.append(text)
            # Token usage is exposed via event.usage_metadata in ADK
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
