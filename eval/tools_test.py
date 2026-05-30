"""Stand-alone smoke test for the shared tool layer (no LLM involved)."""

from __future__ import annotations

import asyncio
import inspect
import sys

from corpus import assert_corpus_present
from tools import CallCounter, glob, grep, read_file


def _invoke_openai_tool(tool_obj, **kwargs):
    """Extract and call the raw function from an OpenAI Agents SDK FunctionTool.

    The SDK wraps the function in a _FailureHandlingFunctionToolInvoker that
    needs a real RunContext; we bypass it by pulling 'the_func' out of the
    _invoke_tool_impl closure instead.
    """
    ci = inspect.getclosurevars(tool_obj.on_invoke_tool._invoke_tool_impl)
    fn = ci.nonlocals["the_func"]
    result = fn(**kwargs)
    if inspect.isawaitable(result):
        result = asyncio.get_event_loop().run_until_complete(result)
    return result


def _invoke_microsoft_tool(tool_obj, **kwargs):
    """Call a Microsoft agent_framework FunctionTool via its .func attribute."""
    return tool_obj.func(**kwargs)


def check_harness_tools() -> None:
    """Verify openai + microsoft harness glob/grep return str, not list[str].

    The OpenAI Chat Completions API rejects non-string tool outputs; returning a
    list silently replaces the result with a placeholder, breaking tool use.
    """
    from runners import run_microsoft, run_openai

    cases = [
        ("glob", dict(pattern="treasury_bulletin_1941_*.txt")),
        ("grep", dict(pattern="Treasury", glob_pattern="treasury_bulletin_1941_*.txt", max_results=3)),
    ]

    for tool_name, kwargs in cases:
        result = _invoke_openai_tool(getattr(run_openai, tool_name), **kwargs)
        assert isinstance(result, str), (
            f"openai/{tool_name} returned {type(result).__name__}, expected str — "
            "Chat Completions API only accepts string tool outputs"
        )
        assert result, f"openai/{tool_name} returned empty string for known-good query"
        print(f"  openai/{tool_name}: str, {len(result)} chars OK")

    for tool_name, kwargs in cases:
        result = _invoke_microsoft_tool(getattr(run_microsoft, tool_name), **kwargs)
        assert isinstance(result, str), (
            f"microsoft/{tool_name} returned {type(result).__name__}, expected str — "
            "Chat Completions API only accepts string tool outputs"
        )
        assert result, f"microsoft/{tool_name} returned empty string for known-good query"
        print(f"  microsoft/{tool_name}: str, {len(result)} chars OK")

    print("All harness tool checks passed.")


def main() -> int:
    assert_corpus_present()

    print("=== glob('*.txt') (first 5) ===")
    with CallCounter() as cc:
        names = glob("*.txt")
    print(f"  found {len(names)} files; first 5: {names[:5]}")
    print(f"  call counts: {cc.as_dict()}")
    assert len(names) > 100, "expected hundreds of .txt files"

    print("\n=== glob('treasury_bulletin_1941_*.txt') ===")
    with CallCounter() as cc:
        names_41 = glob("treasury_bulletin_1941_*.txt")
    print(f"  found {len(names_41)}: {names_41}")
    print(f"  call counts: {cc.as_dict()}")
    assert names_41, "expected 1941 files"

    target = names_41[0]
    print(f"\n=== read_file({target!r}, limit=20) ===")
    with CallCounter() as cc:
        head = read_file(target, limit=20)
    print(head)
    print(f"  call counts: {cc.as_dict()}")
    assert head and not head.startswith("ERROR"), "read_file failed"

    print(f"\n=== grep('Treasury', glob_pattern='treasury_bulletin_1941_*.txt', max_results=5) ===")
    with CallCounter() as cc:
        hits = grep("Treasury", glob_pattern="treasury_bulletin_1941_*.txt", max_results=5)
    for h in hits:
        print(f"  {h}")
    print(f"  call counts: {cc.as_dict()}")
    assert hits, "expected at least one Treasury hit in 1941 files"
    assert cc.as_dict().get("grep") == 1, f"grep should count once, got {cc.as_dict()}"
    assert cc.as_dict().get("glob", 0) == 0, f"grep should not inflate glob count, got {cc.as_dict()}"

    print("\n=== path-traversal guard ===")
    bad = read_file("../reward.py")
    print(f"  read_file('../reward.py') -> {bad[:80]}")
    assert bad.startswith("ERROR"), "path traversal should be rejected"

    print("\nAll tool-layer checks passed.")

    print("\n=== harness return-type checks (openai + microsoft) ===")
    check_harness_tools()

    return 0


if __name__ == "__main__":
    sys.exit(main())
