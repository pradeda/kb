#!/usr/bin/env python3
"""Unit tests for gate deterministic helpers (no embedding/Chroma needed).
Run: python3 -m pytest eval/test_gate.py  (or: python3 eval/test_gate.py)"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gate


def test_norm_tags_splits_and_lowercases():
    assert gate._norm_tags("Prowlarr, Jackett/audiobookbay;RSS") == {
        "prowlarr", "jackett", "audiobookbay", "rss"}


def test_norm_tags_empty():
    assert gate._norm_tags("") == set()
    assert gate._norm_tags(None) == set()


def test_topic_tokens_drops_stoplist_numeric_and_short():
    # gotcha/homelab/docker are stoplisted; 2026 numeric; 'db' too short
    topics = gate._topic_tokens("gotcha,homelab,docker,audiobookbay", "ABB 2026 db fix")
    assert "audiobookbay" in topics
    assert "gotcha" not in topics and "homelab" not in topics and "docker" not in topics
    assert "2026" not in topics
    assert "db" not in topics
    assert "fix" not in topics  # stoplisted


def test_topic_tokens_title_secondary_signal():
    topics = gate._topic_tokens("", "Audiobookshelf DB path correction")
    assert "audiobookshelf" in topics
    assert "correction" in topics


def test_overlap_same_service_positive():
    a = gate._topic_tokens("audiobookbay,jackett", "ABB domain change")
    b = gate._topic_tokens("audiobookbay,prowlarr,gotcha", "AudioBookBay unavailable")
    assert a & b  # audiobookbay shared


def test_overlap_generic_only_is_empty():
    # two entries sharing ONLY generic tags must not count as related topic
    a = gate._topic_tokens("gotcha,fix,homelab", "Beszel agent")
    b = gate._topic_tokens("gotcha,fix,homelab", "WireGuard client")
    # 'beszel' vs 'wireguard' differ; generic tags stripped -> no overlap
    assert not (a & b)


def test_check_relatedness_requires_threshold():
    try:
        gate.check_relatedness("x", "y", "z", None)
    except ValueError:
        return
    assert False, "expected ValueError for missing threshold"


def test_check_relatedness_empty_input():
    try:
        gate.check_relatedness("", "", "", 0.6)
    except ValueError:
        return
    assert False, "expected ValueError for empty incoming entry"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns)-failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
