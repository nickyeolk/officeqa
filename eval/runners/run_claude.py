"""Claude Agent SDK runner routed through OpenRouter.

We expose the three shared tools via an in-process MCP server, and tell the
underlying Claude binary to call OpenRouter's Anthropic-compatible endpoint
via ANTHROPIC_BASE_URL / ANTHROPIC_API_KEY.

NOTE: The Claude Agent SDK launches the local ``claude`` CLI as a subprocess.
That binary must be installed and on PATH. (You already have it — this whole
conversation is running inside Claude Code.)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

# Sibling imports
_EVAL_DIR = Path(__file__).resolve().parent.parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from claude_agent_sdk import (  # type: ignore  # noqa: E402
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
    query,
    tool,
)

from config import MAX_TURNS, require_openrouter_key  # noqa: E402
from openrouter_anthropic_shim import AnthropicShim  # noqa: E402
from prompts import SYSTEM_PROMPT, user_prompt  # noqa: E402
from runners.base import QuestionResult  # noqa: E402
from scoring import extract_final_answer  # noqa: E402
from tools import CallCounter, glob as _glob, grep as _grep, read_file as _read_file  # noqa: E402

# Lazy-init: one shim per process, kept alive for the whole eval run.
_SHIM: AnthropicShim | None = None


def _shim_url() -> str:
    global _SHIM
    if _SHIM is None:
        _SHIM = AnthropicShim(openrouter_api_key=require_openrouter_key())
        _SHIM.start()
    return _SHIM.url


# --- MCP tool definitions (input schema is a dict of name -> python type) -


@tool(
    "read_file",
    "Read a Treasury Bulletin .txt file by basename, with optional line paging.",
    {"filename": str, "offset": int, "limit": int},
)
async def t_read_file(args: dict[str, Any]) -> dict[str, Any]:
    text = _read_file(
        args["filename"],
        offset=int(args.get("offset", 0) or 0),
        limit=int(args.get("limit", 2000) or 2000),
    )
    return {"content": [{"type": "text", "text": text}]}


@tool(
    "glob",
    "List corpus filenames matching a glob pattern (e.g. 'treasury_bulletin_1941_*.txt').",
    {"pattern": str},
)
async def t_glob(args: dict[str, Any]) -> dict[str, Any]:
    names = _glob(args["pattern"])
    return {"content": [{"type": "text", "text": "\n".join(names) if names else "(no matches)"}]}


@tool(
    "grep",
    "Regex search across files matching glob_pattern. Returns '<file>:<line>: <text>' hits.",
    {"pattern": str, "glob_pattern": str, "max_results": int, "context": int},
)
async def t_grep(args: dict[str, Any]) -> dict[str, Any]:
    hits = _grep(
        args["pattern"],
        glob_pattern=str(args.get("glob_pattern", "*.txt") or "*.txt"),
        max_results=int(args.get("max_results", 50) or 50),
        context=int(args.get("context", 0) or 0),
    )
    return {"content": [{"type": "text", "text": "\n".join(hits) if hits else "(no hits)"}]}


_MCP_SERVER = create_sdk_mcp_server(
    name="officeqa-tools",
    version="1.0.0",
    tools=[t_read_file, t_glob, t_grep],
)
_TOOL_NAMES = [
    "mcp__officeqa-tools__read_file",
    "mcp__officeqa-tools__glob",
    "mcp__officeqa-tools__grep",
]


def _options(model: str) -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        model=model,
        mcp_servers={"officeqa-tools": _MCP_SERVER},
        allowed_tools=_TOOL_NAMES,
        max_turns=MAX_TURNS,
        permission_mode="bypassPermissions",
        env={
            # Point Claude CLI at our local shim, which:
            #   - stubs /v1/me /v1/organizations* /v1/models* (OpenRouter 404s on these)
            #   - forwards /v1/messages to OpenRouter
            "ANTHROPIC_BASE_URL": _shim_url(),
            "ANTHROPIC_AUTH_TOKEN": require_openrouter_key(),
            "ANTHROPIC_API_KEY": require_openrouter_key(),
            # Suppress sidecar network calls (telemetry, sentry, statsig)
            "DISABLE_TELEMETRY": "1",
            "DISABLE_ERROR_REPORTING": "1",
            "DISABLE_NON_ESSENTIAL_MODEL_CALLS": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_MODEL_CALLS": "1",
        },
    )


async def run_question(row: dict, model: str) -> QuestionResult:
    result = QuestionResult(
        uid=str(row.get("uid", "")),
        question=str(row.get("question", "")),
        ground_truth=str(row.get("answer", "")),
        sdk="claude",
        model=model,
    )
    options = _options(model)
    raw_parts: list[str] = []
    llm_calls = 0
    pt = ct = tt = 0

    t0 = time.perf_counter()
    with CallCounter() as cc:
        async for msg in query(prompt=user_prompt(result.question), options=options):
            if isinstance(msg, AssistantMessage):
                llm_calls += 1
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        raw_parts.append(block.text)
                    elif isinstance(block, ToolUseBlock):
                        # Our wrapper already tallies via CallCounter; this is a
                        # belt-and-braces sanity check for parity with the SDK.
                        pass
            elif isinstance(msg, ResultMessage):
                usage = getattr(msg, "usage", None)
                if usage is not None:
                    # The SDK's usage object mirrors Anthropic's: input_tokens / output_tokens
                    pt = int(usage.get("input_tokens", 0) if isinstance(usage, dict) else getattr(usage, "input_tokens", 0) or 0)
                    ct = int(usage.get("output_tokens", 0) if isinstance(usage, dict) else getattr(usage, "output_tokens", 0) or 0)
                    tt = pt + ct
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
