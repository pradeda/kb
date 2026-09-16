#!/usr/bin/env python3
"""Supersede-gate eval runner.

Measures three metrics SEPARATELY (aggregate stop-rate is not enough):
  miss-rate       over supersede items  = expected related entry NOT surfaced (gate failed to warn)
  stop-on-coexist over coexist items    = gate surfaced a related entry (unnecessary friction)
  stop-on-new     over new items        = gate surfaced a related entry (false alarm)

Rules honored:
  - Stratified split by case_id; every variant of a case stays on the SAME side
    (no train/holdout leak). Stratified within (label, origin).
  - Real and synthetic results reported SEPARATELY. Synthetic passes must not
    mask a real miss (real supersede require ZERO misses; enforced independently).
  - Report error COUNT + total per class per split. No percentage rounding to pass:
    a bound is met only if errors/total <= bound exactly (fractions compared, not
    rounded percents).

Bounds: miss <= 0.10 (real supersede: exactly 0), stop-on-coexist <= 0.20,
stop-on-new <= 0.10 — must hold on train AND holdout.

Usage:
  supersede_eval_runner.py --threshold 0.65        # evaluate at one threshold
  supersede_eval_runner.py --sweep 0.50:0.80:0.025 # print metric table over range
"""
import argparse, json, os, sqlite3, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gate

HERE = os.path.dirname(os.path.abspath(__file__))
TESTSET = os.path.join(HERE, "supersede_test_set.json")
DB = "/opt/kb/kb.db"

MISS_MAX = 0.10
STOP_COEXIST_MAX = 0.20
STOP_NEW_MAX = 0.10
HOLDOUT_EVERY = 3   # ~1 in 3 cases per stratum -> ~33% holdout, deterministic


def load_items():
    with open(TESTSET) as f:
        data = json.load(f)
    return data["items"]


def fetch_entry(new_id):
    db = sqlite3.connect(DB)
    row = db.execute("SELECT content, title, tags FROM entries WHERE id=?", (new_id,)).fetchone()
    db.close()
    if not row:
        raise LookupError(f"real item new_id={new_id} not found in {DB}")
    return {"content": row[0] or "", "title": row[1] or "", "tags": row[2] or ""}


def incoming_for(item):
    """Return (content, title, tags, exclude_ids) for an item."""
    if item["origin"] == "real":
        e = fetch_entry(item["new_id"])
        return e["content"], e["title"], e["tags"], (item["new_id"],)
    return item["new_text"], item.get("new_title", ""), item.get("new_tags", ""), ()


def stratified_split(items):
    """Group by case_id; assign whole cases to train/holdout, stratified by
    (label, origin). Deterministic (sorted). Returns dict case_id -> 'train'|'holdout'."""
    cases = {}
    for it in items:
        cases.setdefault(it["case_id"], it)  # representative for stratum key
    by_stratum = {}
    for cid, rep in cases.items():
        by_stratum.setdefault((rep["label"], rep["origin"]), []).append(cid)
    assign = {}
    for stratum, cids in by_stratum.items():
        for idx, cid in enumerate(sorted(cids)):
            assign[cid] = "holdout" if (idx % HOLDOUT_EVERY == HOLDOUT_EVERY - 1) else "train"
    return assign


def evaluate(items, threshold):
    """Run the gate on every item. Returns list of per-item results."""
    results = []
    for it in items:
        content, title, tags, exclude = incoming_for(it)
        try:
            r = gate.check_relatedness(content, title, tags, threshold, exclude_ids=exclude)
            surfaced = [int(x["id"]) for x in r["related"]]
            unavailable = False
        except gate.GateUnavailable as e:
            surfaced = []
            unavailable = True
        stop = len(surfaced) > 0
        expected = set(it.get("expect_related_ids", []))
        hit = bool(expected & set(surfaced)) if expected else None
        results.append({
            "case_id": it["case_id"], "label": it["label"], "origin": it["origin"],
            "surfaced": surfaced, "stop": stop, "expected": sorted(expected),
            "hit": hit, "unavailable": unavailable,
        })
    return results


def tally(results, assign):
    """Bucket errors by class × origin × split. Returns nested counts."""
    buckets = {}  # (split, label, origin) -> {"total":n, "err":n}
    for r in results:
        split = assign[r["case_id"]]
        key = (split, r["label"], r["origin"])
        b = buckets.setdefault(key, {"total": 0, "err": 0, "items": []})
        b["total"] += 1
        err = False
        if r["label"] == "supersede":
            err = (r["hit"] is not True)      # miss = expected not surfaced
        elif r["label"] == "coexist":
            err = r["stop"]                   # unnecessary stop
        elif r["label"] == "new":
            err = r["stop"]                   # false alarm
        if err:
            b["err"] += 1
            b["items"].append(r["case_id"])
    return buckets


def frac(err, total):
    return (err / total) if total else 0.0


def report(results, assign, threshold):
    buckets = tally(results, assign)
    print(f"\n==== threshold = {threshold:.3f} ====")
    # any infra-unavailable items invalidate the run
    unavail = [r["case_id"] for r in results if r["unavailable"]]
    if unavail:
        print(f"!! gate UNAVAILABLE for {len(unavail)} item(s): {unavail} — run is invalid")
        return False

    label_bound = {"supersede": MISS_MAX, "coexist": STOP_COEXIST_MAX, "new": STOP_NEW_MAX}
    metric_name = {"supersede": "miss-rate", "coexist": "stop-on-coexist", "new": "stop-on-new"}
    ok = True
    for split in ("train", "holdout"):
        print(f"\n-- {split} --")
        for label in ("supersede", "coexist", "new"):
            for origin in ("real", "synthetic"):
                key = (split, label, origin)
                if key not in buckets:
                    continue
                b = buckets[key]
                f = frac(b["err"], b["total"])
                bound = label_bound[label]
                # real supersede: zero misses required (independent of the 10% bound)
                if label == "supersede" and origin == "real":
                    passed = (b["err"] == 0)
                    req = "0 misses"
                else:
                    passed = (f <= bound + 1e-9)
                    req = f"<= {bound:.0%}"
                ok = ok and passed
                flag = "OK " if passed else "XX "
                errcases = f"  err_cases={b['items']}" if b["err"] else ""
                print(f"  {flag}{metric_name[label]:16s} {origin:9s} "
                      f"{b['err']}/{b['total']}  ({f:.3f}, need {req}){errcases}")
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def sweep(items, assign, spec):
    lo, hi, step = (float(x) for x in spec.split(":"))
    print(f"{'thr':>6} | {'S.miss(r/s)':>14} | {'coexist':>9} | {'new':>7}  (train buckets, err/total)")
    t = lo
    while t <= hi + 1e-9:
        results = evaluate(items, round(t, 4))
        if any(r["unavailable"] for r in results):
            print(f"{t:6.3f} | gate unavailable — abort sweep")
            return
        b = tally(results, assign)

        def g(label, origin):
            k = ("train", label, origin)
            return f"{b[k]['err']}/{b[k]['total']}" if k in b else "-"
        print(f"{t:6.3f} | {g('supersede','real'):>6} {g('supersede','synthetic'):>6} | "
              f"{g('coexist','real'):>9} | {g('new','real'):>3} {g('new','synthetic'):>3}")
        t += step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.65)
    ap.add_argument("--sweep", type=str, help="lo:hi:step, e.g. 0.50:0.80:0.025")
    args = ap.parse_args()
    items = load_items()
    assign = stratified_split(items)
    ncase = len(set(i["case_id"] for i in items))
    nhold = len(set(c for c, s in assign.items() if s == "holdout"))
    print(f"{len(items)} items, {ncase} cases, {nhold} holdout cases")
    if args.sweep:
        sweep(items, assign, args.sweep)
    else:
        results = evaluate(items, args.threshold)
        ok = report(results, assign, args.threshold)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
