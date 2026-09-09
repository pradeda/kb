#!/usr/bin/env python3
"""KB retrieval evaluation runner.

Loads golden_set.json, runs each query against the v2 search API,
and reports recall@5, corpus precision, bilingual parity, and noise.
"""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

EVAL_DIR = Path(__file__).parent
GOLDEN_SET = EVAL_DIR / "golden_set.json"
ENV_FILE = Path("/opt/kb/.env")
API_URL = os.getenv("KB_V2_SEARCH_URL", "http://127.0.0.1:8050/v2/kb/search")
TOP_K = 5


def load_token() -> str:
    token = os.getenv("KB_V2_TOKEN_MCP_LOCAL")
    if token:
        return token
    for line in ENV_FILE.read_text().splitlines():
        if line.startswith("KB_V2_TOKEN_MCP_LOCAL="):
            return line.split("=", 1)[1].strip().strip("'\"")
    raise RuntimeError("no KB_V2_TOKEN_MCP_LOCAL found")


def search(token: str, query: str, scope: str,
           query_alt: str | None = None,
           query_alt_language: str | None = None) -> dict:
    payload = {
        "query": query,
        "scope": scope,
        "top_k": TOP_K,
        "allow_degraded": False,
    }
    if query_alt:
        payload["query_alt"] = query_alt
    if query_alt_language:
        payload["query_alt_language"] = query_alt_language
    resp = httpx.post(
        API_URL,
        json=payload,
        headers={"Authorization": f"Bearer {token}"},
        timeout=45,
    )
    resp.raise_for_status()
    return resp.json()


def extract_refs(result: dict) -> list[str]:
    return [hit.get("ref", "") for hit in result.get("ranked", [])]


def evaluate_question(token: str, q: dict) -> dict:
    scope = q.get("expected_scope", "both")
    result = search(
        token, q["query"], scope,
        q.get("query_alt"), q.get("query_alt_language"),
    )
    refs = extract_refs(result)
    expected = set(q.get("expected_refs", []))
    found = expected & set(refs)
    must_find = q.get("must_find_min", 1)
    recall = len(found) / len(expected) if expected else 1.0
    passed = len(found) >= must_find

    noise_refs = [r for r in refs if r not in expected]

    entry = {
        "id": q["id"],
        "category": q["category"],
        "query": q["query"],
        "scope": scope,
        "expected_refs": sorted(expected),
        "found_refs": sorted(found),
        "missed_refs": sorted(expected - found),
        "returned_refs": refs,
        "noise_refs": noise_refs,
        "recall": recall,
        "noise_ratio": len(noise_refs) / len(refs) if refs else 0.0,
        "passed": passed,
    }

    if q.get("query_alt"):
        alt_result = search(token, q["query_alt"], scope)
        alt_refs = extract_refs(alt_result)
        primary_set = set(refs)
        alt_set = set(alt_refs)
        overlap = primary_set & alt_set
        parity = len(overlap) / max(len(primary_set | alt_set), 1)
        entry["bilingual_parity"] = parity
        entry["alt_refs"] = alt_refs

    return entry


def main():
    questions = json.loads(GOLDEN_SET.read_text())
    token = load_token()

    results = []
    for q in questions:
        try:
            results.append(evaluate_question(token, q))
        except Exception as exc:
            results.append({
                "id": q["id"],
                "category": q["category"],
                "query": q["query"],
                "error": str(exc),
                "passed": False,
                "recall": 0.0,
                "noise_ratio": 0.0,
            })

    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    avg_recall = sum(r["recall"] for r in results) / total if total else 0
    avg_noise = sum(r["noise_ratio"] for r in results) / total if total else 0

    bilingual = [r for r in results if "bilingual_parity" in r]
    avg_parity = (
        sum(r["bilingual_parity"] for r in bilingual) / len(bilingual)
        if bilingual else None
    )

    by_category: dict[str, list] = {}
    for r in results:
        by_category.setdefault(r["category"], []).append(r)
    category_summary = {}
    for cat, items in by_category.items():
        category_summary[cat] = {
            "total": len(items),
            "passed": sum(1 for i in items if i["passed"]),
            "avg_recall": round(sum(i["recall"] for i in items) / len(items), 3),
        }

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "corpus_sizes": {"homelab": 629, "ai": 180},
        "questions": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total, 3) if total else 0,
        "avg_recall_at_5": round(avg_recall, 3),
        "avg_noise_ratio": round(avg_noise, 3),
        "avg_bilingual_parity": round(avg_parity, 3) if avg_parity is not None else None,
        "by_category": category_summary,
        "details": results,
    }

    print("=" * 60)
    print("KB RETRIEVAL EVALUATION REPORT")
    print("=" * 60)
    print(f"Date:       {report['timestamp'][:10]}")
    print(f"Questions:  {total}")
    print(f"Passed:     {passed}/{total} ({report['pass_rate']:.0%})")
    print(f"Recall@5:   {avg_recall:.2f}")
    print(f"Noise:      {avg_noise:.2f}")
    if avg_parity is not None:
        print(f"Bilingual:  {avg_parity:.2f}")
    print()
    for cat, s in category_summary.items():
        print(f"  {cat:12s}  {s['passed']}/{s['total']}  recall={s['avg_recall']:.2f}")
    print()

    failed_qs = [r for r in results if not r["passed"]]
    if failed_qs:
        print("FAILURES:")
        for r in failed_qs:
            print(f"  [{r['id']}] {r['query'][:50]}")
            if "error" in r:
                print(f"    error: {r['error']}")
            else:
                print(f"    missed: {r.get('missed_refs', [])}")
                print(f"    got:    {r.get('returned_refs', [])}")
        print()

    report_path = EVAL_DIR / f"report-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Full report: {report_path}")


if __name__ == "__main__":
    main()
