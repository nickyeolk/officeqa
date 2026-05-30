"""Quick JSONL inspector for smoke tests."""
import json, sys
path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        gt = (r["ground_truth"] or "")[:40]
        pred = (r["predicted"] or "")[:60]
        err = r.get("error")
        print(f'uid={r["uid"]}')
        print(f'  gt   = {gt!r}')
        print(f'  pred = {pred!r}')
        print(f'  score@1%={r["score_at_1pct"]} lat={r["latency_s"]:.1f}s '
              f'tc={r["tool_calls"]} llm={r["llm_calls"]} tok={r["total_tokens"]:,}')
        print(f'  tools={r["tool_calls_by_name"]}')
        if err:
            first = err.splitlines()[0][:120]
            print(f'  error: {first}')
        print()
