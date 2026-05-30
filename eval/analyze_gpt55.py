import json
from pathlib import Path

for fname in [
    "results/openai_openai_gpt-5.5_20260527T153037Z.jsonl",
    "results/microsoft_openai_gpt-5.5_20260527T153040Z.jsonl",
]:
    f = Path(fname)
    rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    n = len(rows)
    ok = [r for r in rows if not r.get("error")]
    avg_tok = sum(r.get("total_tokens", 0) for r in ok) / len(ok)
    avg_llm = sum(r.get("llm_calls", 0) for r in ok) / len(ok)
    avg_tc = sum(r.get("tool_calls", 0) for r in ok) / len(ok)
    avg_score = sum(r.get("score_at_1pct", 0) for r in ok) / len(ok)
    max_tok = max(r.get("total_tokens", 0) for r in ok)
    min_tok = min(r.get("total_tokens", 0) for r in ok)
    zero_tok = sum(1 for r in ok if r.get("total_tokens", 0) == 0)
    print(f"\n=== {f.name} ===")
    print(f"N={n} ok={len(ok)} errors={n-len(ok)}")
    print(f"Score@1%: {avg_score:.3f}")
    print(f"Avg tokens: {avg_tok:,.0f}  min={min_tok:,}  max={max_tok:,}  zero_tok_rows={zero_tok}")
    print(f"Avg llm_calls: {avg_llm:.1f}  avg tool_calls: {avg_tc:.1f}")
    print("\nFirst 5 rows:")
    for r in rows[:5]:
        print(f"  uid={r['uid']} tok={r.get('total_tokens',0):,} llm={r.get('llm_calls',0)} tc={r.get('tool_calls',0)} score={r.get('score_at_1pct',0)} err={r.get('error') is not None}")
    # Check token distribution buckets
    toks = sorted(r.get("total_tokens", 0) for r in ok)
    buckets = [0, 10000, 50000, 100000, 250000, 500000, 1000000]
    print("\nToken distribution:")
    for i in range(len(buckets)-1):
        count = sum(1 for t in toks if buckets[i] <= t < buckets[i+1])
        print(f"  {buckets[i]:>9,} - {buckets[i+1]-1:>9,}: {count}")
    print(f"  {buckets[-1]:>9,}+           : {sum(1 for t in toks if t >= buckets[-1])}")
