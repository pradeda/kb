#!/usr/bin/env python3
"""Regression tests for [SUPERSEDED] search demotion in the LIVE v2 pipeline
(kb_v2._apply_decay). v1 (kb_search_api) is retired (HTTP 410) — do not test it.
Run: /opt/kb/venv-search/bin/python3 eval/test_demotion.py"""
import sys, os
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kb_v2 as v2

TODAY = datetime.now(timezone.utc).isoformat()


def _rc(ai_mode="disabled"):
    return v2.RouterConfig(
        router_version="test", accept_thresholds={}, reject_threshold=0.0,
        both_margin=0.0, dead_zone_lower=0.0, dead_zone_upper=0.0, candidate_k=25,
        max_distance={}, ai_decay_mode=ai_mode,
        ai_decay_half_life_days=(None if ai_mode == "disabled" else 90.0),
        ai_decay_floor=(None if ai_mode == "disabled" else 0.3),
    )


def _cand(corpus, title, relevance, date=TODAY):
    return v2.Candidate(corpus=corpus, entry_id=1, title=title, content=None,
                        summary=None, tags=None, source=None, date=date,
                        distance=0.1, relevance=relevance)


def test_homelab_superseded_demoted_by_factor():
    rc = _rc()
    normal = _cand("homelab", "Current fact", 0.9)
    old = _cand("homelab", "[SUPERSEDED] Old fact", 0.9)
    v2._apply_decay(normal, rc)
    v2._apply_decay(old, rc)
    assert old.final_score < normal.final_score
    assert abs(old.final_score - normal.final_score * v2.SUPERSEDE_DEMOTE) < 1e-3, \
        f"expected ~{normal.final_score*v2.SUPERSEDE_DEMOTE}, got {old.final_score}"


def test_ai_disabled_path_also_demotes():
    rc = _rc()
    normal = _cand("ai", "Current", 0.8)
    old = _cand("ai", "[SUPERSEDED] Old", 0.8)
    v2._apply_decay(normal, rc)      # ai + disabled -> final = relevance * demote
    v2._apply_decay(old, rc)
    assert abs(normal.final_score - 0.8) < 1e-3
    assert abs(old.final_score - 0.8 * v2.SUPERSEDE_DEMOTE) < 1e-3


def test_final_score_non_negative():
    rc = _rc()
    for c in [_cand("homelab", "[SUPERSEDED] X", 0.9),
              _cand("homelab", "Y", 0.4),
              _cand("homelab", "[SUPERSEDED] Z", 0.01, "2019-01-01T00:00:00")]:
        v2._apply_decay(c, rc)
        assert c.final_score >= 0.0, f"negative final_score {c.final_score}"


def test_non_superseded_untouched():
    rc = _rc()
    c = _cand("homelab", "Normal title", 0.8)
    v2._apply_decay(c, rc)
    # recent -> decay ~1.0, no demotion applied
    assert c.final_score > 0.8 * v2.HOMELAB_DECAY_FLOOR
    assert c.final_score > 0.75


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1; print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:
            failed += 1; print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns)-failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
