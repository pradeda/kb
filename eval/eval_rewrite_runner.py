#!/usr/bin/env python3
"""KB query rewriting evaluation runner.

For each test question:
1. Rewrites the natural query using Qwen (local LLM on Forrix)
2. Calls /v2/kb/trace/dual to get original-only vs union results
3. Also runs keyword_hint through standard search for reference
4. Computes per-question and aggregate metrics

Metrics:
- result_count: how many results each approach returns
- gained/lost entry IDs between original and union
- MRR@5 improvement (when expected_refs are available)
- p50/p95 rewrite latency
- false positive rate on negative controls
"""

import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import urllib.request

EVAL_DIR = Path(__file__).parent
TEST_SET = EVAL_DIR / "rewrite_test_set.json"
ENV_FILE = Path("/opt/kb/.env")
API_BASE = "http://127.0.0.1:8050/v2/kb"

sys.path.insert(0, str(EVAL_DIR))
from qwen_rewriter import rewrite_query


def load_token() -> str:
    token = os.getenv("KB_V2_TOKEN_MCP_LOCAL")
    if token:
        return token
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line.startswith("KB_V2_TOKEN_MCP_LOCAL="):
            return line.split("=", 1)[1]
    raise RuntimeError("KB_V2_TOKEN_MCP_LOCAL not found")


def api_call(path: str, body: dict, token: str, timeout: int = 90) -> dict:
    url = f"{API_BASE}/{path}"
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def search(token: str, query: str, scope: str = "both", top_k: int = 5) -> dict:
    return api_call("search", {"query": query, "scope": scope, "top_k": top_k}, token)


def trace_dual(
    token: str, original: str, rewritten: str, scope: str = "both", top_k: int = 5,
) -> dict:
    return api_call(
        "trace/dual",
        {
            "original_query": original,
            "rewritten_query": rewritten,
            "scope": scope,
            "top_k": top_k,
        },
        token,
    )


def refs_from_ranked(result: dict) -> list[str]:
    return [h.get("ref", "") for h in result.get("ranked", [])]


def refs_from_dual(entries: list[dict]) -> list[str]:
    return [e.get("ref", "") for e in entries]


def mrr(refs: list[str], expected: set[str]) -> float:
    for i, ref in enumerate(refs):
        if ref in expected:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_question(token: str, q: dict) -> dict:
    qid = q["id"]
    natural = q["natural_query"]
    keyword_hint = q.get("keyword_hint", "")
    category = q.get("category", "")

    # Step 1: Rewrite with LLM
    backend = os.getenv("REWRITE_BACKEND", "gemini")
    rewrite_result = rewrite_query(natural, backend=backend)
    rewritten = rewrite_result.get("rewritten")
    if not rewritten:
        return {
            "id": qid,
            "category": category,
            "query": natural,
            "error": f"rewrite failed: {rewrite_result.get('error', 'empty')}",
            "rewrite_latency_ms": rewrite_result.get("latency_ms", 0),
            "backend": backend,
        }

    # Step 2: Dual trace — original vs union(original + rewritten)
    dual = trace_dual(token, natural, rewritten)

    # Step 3: Keyword hint search for reference
    keyword_result = search(token, keyword_hint) if keyword_hint else None

    orig_refs = refs_from_dual(dual["original_results"])
    union_refs = refs_from_dual(dual["union_results"])
    keyword_refs = refs_from_ranked(keyword_result) if keyword_result else []

    orig_set = set(orig_refs)
    union_set = set(union_refs)
    keyword_set = set(keyword_refs)

    gained = sorted(union_set - orig_set)
    lost = sorted(orig_set - union_set)

    # Keyword-reference overlap (not ground truth, just reference)
    orig_keyword_overlap = len(orig_set & keyword_set)
    union_keyword_overlap = len(union_set & keyword_set)

    entry = {
        "id": qid,
        "category": category,
        "query": natural,
        "rewritten": rewritten,
        "rewrite_latency_ms": rewrite_result["latency_ms"],
        "keyword_hint": keyword_hint,
        "original_refs": orig_refs,
        "union_refs": union_refs,
        "keyword_refs": keyword_refs,
        "original_count": len(orig_refs),
        "union_count": len(union_refs),
        "keyword_count": len(keyword_refs),
        "gained": gained,
        "lost": lost,
        "gained_count": len(gained),
        "lost_count": len(lost),
        "orig_keyword_overlap": orig_keyword_overlap,
        "union_keyword_overlap": union_keyword_overlap,
        "original_candidate_count": dual.get("original_candidate_count", 0),
        "union_candidate_count": dual.get("union_candidate_count", 0),
        "dual_elapsed_ms": dual.get("elapsed_ms", 0),
    }

    # Classify outcome
    if len(gained) > 0 and len(lost) == 0:
        entry["outcome"] = "pure_gain"
    elif len(gained) > 0 and len(lost) > 0:
        entry["outcome"] = "mixed"
    elif len(gained) == 0 and len(lost) > 0:
        entry["outcome"] = "regression"
    else:
        entry["outcome"] = "no_change"

    return entry


def main():
    questions = json.loads(TEST_SET.read_text())
    token = load_token()

    print(f"Evaluating {len(questions)} questions with Qwen query rewriting...")
    print(f"Qwen endpoint: {rewrite_query.__module__}")
    print()

    results = []
    for i, q in enumerate(questions):
        try:
            print(f"  [{i+1}/{len(questions)}] {q['id']}: {q['natural_query'][:60]}...", flush=True)
            entry = evaluate_question(token, q)
            results.append(entry)
            status = entry.get("outcome", entry.get("error", "?"))
            print(f"    → {status} (rewrite: {entry.get('rewritten', 'FAIL')[:50]})")
        except Exception as exc:
            print(f"    → ERROR: {exc}")
            results.append({
                "id": q["id"],
                "category": q.get("category", ""),
                "query": q["natural_query"],
                "error": str(exc),
            })

    # Aggregate metrics
    valid = [r for r in results if "outcome" in r]
    errors = [r for r in results if "error" in r]

    outcomes = {}
    for r in valid:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1

    rewrite_latencies = [r["rewrite_latency_ms"] for r in valid]
    dual_latencies = [r["dual_elapsed_ms"] for r in valid]

    total_gained = sum(r["gained_count"] for r in valid)
    total_lost = sum(r["lost_count"] for r in valid)

    # Keyword overlap improvement
    orig_overlaps = [r["orig_keyword_overlap"] for r in valid]
    union_overlaps = [r["union_keyword_overlap"] for r in valid]

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "test_set": str(TEST_SET),
        "questions": len(questions),
        "evaluated": len(valid),
        "errors": len(errors),
        "outcomes": outcomes,
        "total_gained_entries": total_gained,
        "total_lost_entries": total_lost,
        "avg_orig_keyword_overlap": round(statistics.mean(orig_overlaps), 2) if orig_overlaps else 0,
        "avg_union_keyword_overlap": round(statistics.mean(union_overlaps), 2) if union_overlaps else 0,
        "rewrite_latency_ms": {
            "p50": round(statistics.median(rewrite_latencies), 1) if rewrite_latencies else 0,
            "p95": round(sorted(rewrite_latencies)[int(len(rewrite_latencies) * 0.95)] if rewrite_latencies else 0, 1),
            "mean": round(statistics.mean(rewrite_latencies), 1) if rewrite_latencies else 0,
        },
        "dual_latency_ms": {
            "p50": round(statistics.median(dual_latencies), 1) if dual_latencies else 0,
            "p95": round(sorted(dual_latencies)[int(len(dual_latencies) * 0.95)] if dual_latencies else 0, 1),
            "mean": round(statistics.mean(dual_latencies), 1) if dual_latencies else 0,
        },
        "details": results,
    }

    print()
    print("=" * 60)
    print("KB QUERY REWRITING EVALUATION")
    print("=" * 60)
    print(f"Date:        {report['timestamp'][:10]}")
    print(f"Questions:   {report['questions']}")
    print(f"Evaluated:   {report['evaluated']}")
    print(f"Errors:      {report['errors']}")
    print()
    print("Outcomes:")
    for outcome, count in sorted(outcomes.items()):
        print(f"  {outcome:15s}  {count}")
    print()
    print(f"Total gained entries:  +{total_gained}")
    print(f"Total lost entries:    -{total_lost}")
    print(f"Avg orig↔keyword overlap:  {report['avg_orig_keyword_overlap']}")
    print(f"Avg union↔keyword overlap: {report['avg_union_keyword_overlap']}")
    print()
    print(f"Rewrite latency (Qwen):  p50={report['rewrite_latency_ms']['p50']}ms  p95={report['rewrite_latency_ms']['p95']}ms")
    print(f"Dual search latency:     p50={report['dual_latency_ms']['p50']}ms  p95={report['dual_latency_ms']['p95']}ms")
    print()

    # Per-question summary
    print("Per-question:")
    for r in valid:
        marker = {"pure_gain": "+", "mixed": "~", "regression": "-", "no_change": "="}
        m = marker.get(r["outcome"], "?")
        print(
            f"  [{m}] {r['id']:6s} orig={r['original_count']} union={r['union_count']} "
            f"gained={r['gained_count']} lost={r['lost_count']}  "
            f"kw_overlap: {r['orig_keyword_overlap']}→{r['union_keyword_overlap']}  "
            f"{r['rewritten'][:40]}"
        )
    print()

    if errors:
        print("Errors:")
        for r in errors:
            print(f"  {r['id']}: {r['error']}")
        print()

    report_path = EVAL_DIR / f"rewrite-report-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Full report: {report_path}")


if __name__ == "__main__":
    main()
