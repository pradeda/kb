"""The reranker lock: one shared CrossEncoder must never be tokenized twice at once.

The service holds a single CrossEncoder whose HF fast tokenizer is not reentrant.
Two batches tokenizing concurrently raise `RuntimeError: Already borrowed` inside
`set_truncation_and_padding` (observed offline: 10/200 batches at 8 threads), which is
what produced the intermittent 503 `reranker_unavailable`. `_rerank_batch` serializes
passage selection and predict under `_RERANK_LOCK`; these tests pin that, and that a
failing batch still releases the lock.

Run from /opt/kb:  /opt/kb/venv-search/bin/python3 -m unittest tests.test_rerank_lock
"""
from __future__ import annotations

import contextlib
import threading
import time
import unittest

import kb_v2
from kb_v2 import Candidate

THREADS = 8
CALLS = 24
# Long enough to exceed the fake tokenizer's budget, so _reranker_passage takes the
# slicing path and tokenizes repeatedly — the call site that also races in production.
CONTENT = "reranker tokenizer window " * 30


class OverlapDetector:
    """Shared by every tokenizer entry point and by predict; records concurrent entry."""

    def __init__(self, hold: float = 0.003) -> None:
        self._state = threading.Lock()
        self._active = 0
        self._hold = hold
        self.violations: list[str] = []

    @contextlib.contextmanager
    def entered(self, where: str):
        with self._state:
            self._active += 1
            if self._active > 1:
                self.violations.append(f"{where}: {self._active} callers inside at once")
        try:
            # Without serialization this sleep is what makes the overlap real rather
            # than a race the GIL might hide.
            time.sleep(self._hold)
            yield
        finally:
            with self._state:
                self._active -= 1


class GuardedTokenizer:
    def __init__(self, detector: OverlapDetector) -> None:
        self.detector = detector

    def num_special_tokens_to_add(self, pair=True):
        return 3

    def encode(self, text, **kwargs):
        with self.detector.entered("tokenizer.encode"):
            return list(range(len(text)))

    def __call__(self, text, **kwargs):
        with self.detector.entered("tokenizer.__call__"):
            return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}


class GuardedReranker:
    max_seq_length = 512

    def __init__(self, detector: OverlapDetector, fail_predict: bool = False) -> None:
        self.tokenizer = GuardedTokenizer(detector)
        self.detector = detector
        self.fail_predict = fail_predict

    def predict(self, pairs):
        with self.detector.entered("predict"):
            if self.fail_predict:
                raise RuntimeError("reranker exploded")
            return [2.0] * len(pairs)


def batch(index: int, size: int = 2) -> list[Candidate]:
    """Distinct candidates per call: no call may reuse another call's objects."""
    return [
        Candidate(
            corpus="homelab",
            entry_id=10_000 + index * 10 + position,
            title=f"batch {index} candidate {position}",
            content=f"{CONTENT} call={index} candidate={position}",
            summary=None,
            tags="test",
            source="telegram",
            date="2026-08-02T00:00:00+00:00",
            distance=0.1,
        )
        for position in range(size)
    ]


class RerankLockTests(unittest.TestCase):
    def test_concurrent_batches_never_enter_the_tokenizer_together(self) -> None:
        detector = OverlapDetector()
        model = GuardedReranker(detector)
        # Every thread is released at once, and the barrier is outside _rerank_batch:
        # the threads collide on the lock, not before it.
        barrier = threading.Barrier(THREADS)
        indexes = iter(range(CALLS))
        indexes_lock = threading.Lock()
        failures: list[BaseException] = []
        scored: list[list[Candidate]] = []

        def worker() -> None:
            barrier.wait()
            while True:
                with indexes_lock:
                    try:
                        index = next(indexes)
                    except StopIteration:
                        return
                candidates = batch(index)
                try:
                    kb_v2._rerank_batch(f"query {index}", candidates, model)
                except BaseException as exc:  # noqa: BLE001 - report it, do not hide it
                    failures.append(exc)
                else:
                    scored.append(candidates)

        threads = [threading.Thread(target=worker) for _ in range(THREADS)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30)

        self.assertFalse([t for t in threads if t.is_alive()], "a batch never finished")
        self.assertEqual([], [f"{type(exc).__name__}: {exc}" for exc in failures])
        self.assertEqual([], detector.violations)
        self.assertEqual(CALLS, len(scored))
        # The lock must not turn a batch into a no-op.
        self.assertTrue(all(item.relevance > 0 for items in scored for item in items))

    def test_a_failing_batch_releases_the_lock(self) -> None:
        detector = OverlapDetector(hold=0.0)
        with self.assertRaises(RuntimeError):
            kb_v2._rerank_batch("query", batch(0), GuardedReranker(detector, fail_predict=True))

        done: list[bool] = []
        follow_up = threading.Thread(
            target=lambda: (
                kb_v2._rerank_batch("query", batch(1), GuardedReranker(detector)),
                done.append(True),
            ),
            daemon=True,
        )
        follow_up.start()
        follow_up.join(10)
        self.assertTrue(done, "the lock stayed held after a batch raised")


if __name__ == "__main__":
    unittest.main()
