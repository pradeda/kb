"""Regression tests for the compile.py mutating lock and the retire/pass race.

Written for the 2026-09-20 incident (KB homelab:1003): `kb retire` deleted the
SQLite row and the vector while a watcher pass had already snapshotted that row
as unembedded. The pass then wrote the vector for the deleted row, and
mark_embedded()'s UPDATE matched zero rows and reported nothing — leaving an
orphan vector that the 02:15 health alarm surfaced the next morning.

Two independent guarantees are tested here:
  1. mutating runs serialize on the per-corpus lock (compile.py owns it now;
     watcher.sh must not hold a shell-level flock around the call, which would
     deadlock the child — flock binds to the open file description, not to the
     process).
  2. if a row disappears mid-pass anyway, the just-written vector is removed
     instead of becoming an orphan.

The race test is deterministic, not a timing race: the concurrent retire is
injected inside the collection's upsert(), which is exactly the window between
get_unembedded() and the Chroma write. No threads, no sleeps, no production
contact — the module is imported with the KB isolation env pointing at temp
paths, so a bug here cannot reach /opt/kb/kb.db or kb_collection.

Run from /opt/kb:  /opt/kb/venv-embed/bin/python /opt/kb/tests/test_compile_lock_race.py
"""
import contextlib
import errno
import fcntl
import importlib.util
import io
import json
import os
import pathlib
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

# Resolve compile.py relative to this file, same rule as test_retire_orphan.py:
# a candidate copy beside the test wins (used to exercise a change before
# deployment), then the kb-go repo layout, then the deployed /opt/kb tree.
_HERE = pathlib.Path(__file__).resolve().parent
for _candidate in (
    _HERE / "compile.py",
    _HERE.parent / "runtime" / "compile.py",
    _HERE.parent / "compile.py",
):
    if _candidate.exists():
        MODULE_PATH = _candidate
        break
else:
    raise RuntimeError(f"compile.py not found next to or above {_HERE}")

_TMPROOT = pathlib.Path(tempfile.mkdtemp(prefix="kb-lock-tests-"))

# compile.py refuses a partial isolation override at import time, so all four
# vars are set. This is the sanctioned isolation path shared with the Go
# entry-point test — never fall through to a production path from a test.
for _var, _sub in (
    ("KB_HOMELAB_DB", "homelab.db"),
    ("KB_AI_DB", "ai.db"),
    ("KB_HOMELAB_RAW", "homelab-raw"),
    ("KB_AI_RAW", "ai-raw"),
):
    os.environ.setdefault(_var, str(_TMPROOT / _sub))

# compile.py imports its sibling supersede_index, and the module under test may
# be a candidate copy in this directory rather than the deployed parent, so every
# plausible directory goes on the path.
for _path in (str(MODULE_PATH.parent), str(_HERE.parent / "runtime"),
              str(_HERE.parent), str(_HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

SPEC = importlib.util.spec_from_file_location("kb_compile", MODULE_PATH)
kbcompile = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(kbcompile)


class _Vector(list):
    def tolist(self):
        return list(self)


class StubModel:
    """Stands in for FastEmbed; embedding values are irrelevant to these tests."""

    def passage_embed(self, texts):
        return [_Vector([0.0, 1.0]) for _ in texts]


class StubCollection:
    """Minimal Chroma surface used by embed_entries and prune_orphan_vectors."""

    def __init__(self, on_upsert=None):
        self.vectors = {}
        self.on_upsert = on_upsert

    def upsert(self, ids, embeddings, documents, metadatas):
        if self.on_upsert is not None:
            self.on_upsert(list(ids))
        for i in ids:
            self.vectors[i] = ""

    def delete(self, ids):
        for i in ids:
            self.vectors.pop(i, None)

    def get(self, ids, include=None):
        return {"ids": [i for i in ids if i in self.vectors]}


def make_db(path, rows):
    """rows: (id, content, title, raw_path, embedded_at, created_at)"""
    connection = sqlite3.connect(str(path))
    connection.execute(
        "CREATE TABLE entries ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, type TEXT, content TEXT, "
        "title TEXT, summary TEXT, tags TEXT, raw_path TEXT, source TEXT, "
        "compiled_at DATETIME, embedded_at DATETIME, "
        "created_at DATETIME DEFAULT CURRENT_TIMESTAMP)"
    )
    connection.executemany(
        "INSERT INTO entries (id,type,content,title,raw_path,embedded_at,created_at) "
        "VALUES (?, 'note', ?, ?, ?, ?, ?)",
        rows,
    )
    connection.commit()
    connection.close()


class CompileLockTestCase(unittest.TestCase):
    def setUp(self):
        self.dir = pathlib.Path(tempfile.mkdtemp(prefix="case-", dir=_TMPROOT))
        self.db_path = self.dir / "kb.db"
        self.raw = self.dir / "raw"
        (self.raw / "notes").mkdir(parents=True)
        self.lock_path = self.dir / "kb-watcher.lock"

        self._saved = {
            name: getattr(kbcompile, name)
            for name in ("DB", "RAW", "ACTIVE_CORPUS",
                         "get_chroma_collection", "get_embed_model")
        }
        self._saved_lock = kbcompile.CORPUS_PROFILES["homelab"]["watcher_lock"]
        self._saved_paths = {
            key: kbcompile.CORPUS_PROFILES["homelab"][key] for key in ("db", "raw")
        }
        self.addCleanup(self._restore)

        kbcompile.DB = self.db_path
        kbcompile.RAW = self.raw
        kbcompile.ACTIVE_CORPUS = "homelab"
        kbcompile.get_embed_model = lambda: StubModel()
        kbcompile.CORPUS_PROFILES["homelab"]["watcher_lock"] = str(self.lock_path)
        # main() re-runs configure_corpus(), which re-reads the profile — so the
        # profile paths must point at the temp corpus too, not just the globals
        # (which cover direct calls such as mark_embedded).
        kbcompile.CORPUS_PROFILES["homelab"]["db"] = str(self.db_path)
        kbcompile.CORPUS_PROFILES["homelab"]["raw"] = str(self.raw)

    def _restore(self):
        release = getattr(kbcompile, "release_compile_lock", None)
        if release is not None:
            release()
        for name, value in self._saved.items():
            setattr(kbcompile, name, value)
        kbcompile.CORPUS_PROFILES["homelab"]["watcher_lock"] = self._saved_lock
        for key, value in self._saved_paths.items():
            kbcompile.CORPUS_PROFILES["homelab"][key] = value

    def _patch(self, name, value):
        """Patch a compile.py attribute and restore it after the test.

        The module object is shared by every test in this process, so a patch
        left behind would silently change what the next test exercises.
        """
        original = getattr(kbcompile, name)
        setattr(kbcompile, name, value)
        self.addCleanup(lambda: setattr(kbcompile, name, original))

    def _patch_attr(self, obj, name, value):
        original = getattr(obj, name)
        setattr(obj, name, value)
        self.addCleanup(lambda: setattr(obj, name, original))

    def _add_note(self, entry_id, name="note", embedded_at=None):
        raw_file = self.raw / "notes" / f"{name}.md"
        raw_file.write_text("body", encoding="utf-8")
        make_db(self.db_path, [(
            entry_id, "body text for the note", name, str(raw_file),
            embedded_at, "2026-09-20T22:23:00",
        )])
        return raw_file

    def _lock_is_free(self):
        fd = os.open(str(self.lock_path), os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        else:
            fcntl.flock(fd, fcntl.LOCK_UN)
            return True
        finally:
            os.close(fd)


class PassRetireRaceTests(CompileLockTestCase):
    def test_row_deleted_mid_pass_leaves_no_orphan_vector(self):
        """The 2026-09-20 incident, reproduced deterministically.

        The concurrent retire is injected inside upsert(), i.e. after the pass
        has snapshotted the row and before mark_embedded() runs. Without the
        prune step this leaves a vector with no row — the orphan that failed
        --health. Without the lock the injected retire would also be able to run
        while the pass holds it, so this asserts both guarantees at once.
        """
        self._add_note(1, name="doomed")

        lock_held_during_write = []

        def retire_mid_pass(ids):
            lock_held_during_write.append(not self._lock_is_free())
            connection = sqlite3.connect(str(self.db_path))
            connection.execute(
                f"DELETE FROM entries WHERE id IN ({','.join('?' * len(ids))})", ids
            )
            connection.commit()
            connection.close()

        collection = StubCollection(on_upsert=retire_mid_pass)
        kbcompile.get_chroma_collection = lambda: collection

        kbcompile.main([])

        self.assertEqual(collection.vectors, {},
                         "orphan vector survived a mid-pass row deletion")
        self.assertEqual([True], lock_held_during_write,
                         "the pass wrote to Chroma without holding the corpus lock")

        connection = sqlite3.connect(str(self.db_path))
        remaining = connection.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
        connection.close()
        self.assertEqual(remaining, 0)

    def test_normal_pass_still_embeds_and_marks(self):
        """The happy path must be untouched by the lock and the prune step."""
        self._add_note(7, name="kept")
        collection = StubCollection()
        kbcompile.get_chroma_collection = lambda: collection

        kbcompile.main([])

        self.assertEqual(set(collection.vectors), {"7"})
        connection = sqlite3.connect(str(self.db_path))
        embedded = connection.execute(
            "SELECT embedded_at FROM entries WHERE id=7").fetchone()[0]
        connection.close()
        self.assertIsNotNone(embedded, "normal pass stopped marking entries embedded")


class MutatingLockTests(CompileLockTestCase):
    def test_lock_is_exclusive_and_idempotent_in_process(self):
        """A second open() must block; a second call in this process must not.

        flock is bound to the open file description, so re-locking from the same
        process on a fresh fd would block forever — acquire_compile_lock has to
        recognise its own lock. A different fd is used to prove the lock is
        genuinely held, which is the same kernel semantics a second process hits.
        """
        first = kbcompile.acquire_compile_lock(self.lock_path)
        self.assertIs(kbcompile.acquire_compile_lock(self.lock_path), first,
                      "second acquire in the same process did not return the held fd")
        self.assertFalse(self._lock_is_free(), "lock not actually held")

        kbcompile.release_compile_lock()

        self.assertTrue(self._lock_is_free(), "release did not drop the lock")

    def test_mutating_and_read_only_modes_are_classified(self):
        read_only = (["--health"], ["--history", "homelab:1"])
        mutating = ([], ["--retire", "1"], ["--supersede", "1", "--replacement", "homelab:2"],
                    ["--recover-db"], ["--recover-raw"], ["--rebuild-supersede-index"])
        for argv in read_only:
            self.assertFalse(kbcompile.is_mutating_run(kbcompile.parse_args(argv)),
                             f"{argv} must not take the mutating lock")
        for argv in mutating:
            self.assertTrue(kbcompile.is_mutating_run(kbcompile.parse_args(argv)),
                            f"{argv} must take the mutating lock")

    def test_health_takes_the_shared_lock_not_the_exclusive_one(self):
        """--health must observe a settled corpus without queueing behind a pass."""
        self._patch("acquire_compile_lock", lambda *a, **k: self.fail(
            "read-only --health took the exclusive mutating lock"))
        shared_calls = []

        def fake_shared(path=None, timeout=None):
            shared_calls.append(str(path))
            return os.open(str(self.lock_path), os.O_CREAT | os.O_RDWR)

        self._patch("acquire_shared_health_lock", fake_shared)
        self._patch("check_health", lambda: None)
        self._patch_attr(kbcompile.supersede_index, "health_mismatch", lambda *a, **k: {})

        rc = kbcompile.main(["--health"])

        self.assertEqual(rc, 0)
        self.assertEqual(shared_calls, [str(self.lock_path)])

    def test_health_reports_busy_when_a_mutating_run_holds_the_lock(self):
        """A held exclusive lock must yield exit 3 + {"status":"busy"}, not a FAIL."""
        kbcompile.acquire_compile_lock(self.lock_path)          # exclusive, held
        self._patch("HEALTH_LOCK_TIMEOUT", 0.1)                 # do not wait 60s
        self._patch("check_health", lambda: self.fail(
            "check_health ran while a mutating run held the lock"))

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            rc = kbcompile.main(["--health"])

        self.assertEqual(rc, kbcompile.HEALTH_BUSY_EXIT)
        self.assertEqual(rc, 3, "the busy contract is exit 3")
        payload = json.loads(buffer.getvalue().strip())
        self.assertEqual(payload["status"], "busy")
        self.assertEqual(payload["lock"], str(self.lock_path))

    def test_real_lock_errors_propagate_instead_of_looking_busy(self):
        """Only EAGAIN/EWOULDBLOCK is contention; anything else must fail loudly.

        A broken lock (EBADF here) reported as "busy" would mean --health exits 3
        and the wrapper deliberately stays silent — a lock that cannot be taken
        at all would look like a healthy concurrent mutation.
        """
        self._patch("check_health", lambda: self.fail(
            "check_health ran despite a broken lock"))
        with mock.patch.object(kbcompile.fcntl, "flock",
                               side_effect=OSError(errno.EBADF, "bad file descriptor")):
            with self.assertRaises(OSError):
                kbcompile.acquire_shared_health_lock(self.lock_path, timeout=0.1)
            with self.assertRaises(OSError):
                kbcompile.acquire_compile_lock(self.lock_path)

        # And the same through main(): a broken lock is a failure, never exit 3.
        with mock.patch.object(kbcompile.fcntl, "flock",
                               side_effect=OSError(errno.EBADF, "bad file descriptor")):
            with self.assertRaises(OSError):
                kbcompile.main(["--health"])

    def test_contention_is_still_reported_as_busy(self):
        """The narrowed handler must not break the real busy path."""
        kbcompile.acquire_compile_lock(self.lock_path)
        self._patch("HEALTH_LOCK_TIMEOUT", 0.1)
        self.assertIsNone(
            kbcompile.acquire_shared_health_lock(self.lock_path, timeout=0.1))

    def test_rebuild_supersede_index_locks_the_homelab_corpus(self):
        """The edge index lives in the homelab DB whatever --corpus says."""
        args = kbcompile.parse_args(["--corpus", "ai", "--rebuild-supersede-index"])
        self.assertEqual(kbcompile.lock_profile_name(args), "homelab")


class MarkEmbeddedTests(CompileLockTestCase):
    def test_mark_embedded_reports_rows_deleted_before_the_stamp(self):
        self._add_note(1, name="kept")
        make_db_connection = sqlite3.connect(str(self.db_path))
        make_db_connection.execute(
            "INSERT INTO entries (id,type,content,title,raw_path,created_at) "
            "VALUES (2,'note','gone','gone','', '2026-09-20T22:23:00')")
        make_db_connection.commit()
        make_db_connection.close()

        connection = sqlite3.connect(str(self.db_path))
        connection.execute("DELETE FROM entries WHERE id=2")
        connection.commit()
        connection.close()

        vanished = kbcompile.mark_embedded(["1", "2"])

        self.assertEqual(vanished, ["2"], "vanished row was not reported")
        connection = sqlite3.connect(str(self.db_path))
        embedded = connection.execute(
            "SELECT embedded_at FROM entries WHERE id=1").fetchone()[0]
        connection.close()
        self.assertIsNotNone(embedded, "surviving row was not stamped")


CHILD_LOCK_SCRIPT = '''\
import pathlib, sys, time

module_dir, mode, lock, signals = sys.argv[1:5]
sys.path.insert(0, module_dir)
import compile as kbcompile

signals = pathlib.Path(signals)


def wait_for(name, timeout=30.0):
    target = signals / name
    deadline = time.monotonic() + timeout
    while not target.exists():
        if time.monotonic() >= deadline:
            sys.exit(f"timeout waiting for {name}")
        time.sleep(0.02)


if mode == "hold":
    kbcompile.acquire_compile_lock(lock)
    (signals / "HOLDING").write_text("")
    wait_for("RELEASE")
else:
    (signals / "ACQUIRING").write_text("")
    kbcompile.acquire_compile_lock(lock)   # must block until the holder exits
    (signals / "ACQUIRED").write_text("")
'''


class InterProcessLockTests(unittest.TestCase):
    """Two real compile.py processes, real acquire_compile_lock, file handshake.

    The in-process tests cover the same kernel semantics (flock is per open file
    description, so a second fd in one process conflicts exactly like a second
    process does), but this exercises the actual API across a process boundary
    and the lock's lifetime: the holder never unlocks explicitly, so the lock
    must be released by process exit.

    No sleeps and no timing assertions: the holder only releases when the parent
    creates the RELEASE file, which happens strictly after ACQUIRING is observed.
    """

    def setUp(self):
        self.dir = pathlib.Path(tempfile.mkdtemp(prefix="ipc-", dir=_TMPROOT))
        self.lock = self.dir / "kb-watcher.lock"
        self.signals = self.dir / "signals"
        self.signals.mkdir()
        self.script = self.dir / "lock_child.py"
        self.script.write_text(CHILD_LOCK_SCRIPT, encoding="utf-8")

    def _spawn(self, mode):
        return subprocess.Popen(
            [sys.executable, str(self.script), str(MODULE_PATH.parent),
             mode, str(self.lock), str(self.signals)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )

    def _wait_for(self, name, timeout=30.0):
        target = self.signals / name
        deadline = time.monotonic() + timeout
        while not target.exists():
            if time.monotonic() >= deadline:
                self.fail(f"timed out waiting for the child to write {name}")
            time.sleep(0.02)

    def test_second_process_waits_for_the_holder_to_exit(self):
        holder = self._spawn("hold")
        acquirer = None
        try:
            self._wait_for("HOLDING")
            acquirer = self._spawn("acquire")
            self._wait_for("ACQUIRING")
            self.assertFalse(
                (self.signals / "ACQUIRED").exists(),
                "a second process acquired the lock while the first held it")
            (self.signals / "RELEASE").write_text("")
            self._wait_for("ACQUIRED")
            _, err = acquirer.communicate(timeout=30)
            self.assertEqual(acquirer.returncode, 0, err)
        finally:
            if acquirer is not None and acquirer.poll() is None:
                acquirer.kill()
            _, holder_err = holder.communicate(timeout=30)
        self.assertEqual(holder.returncode, 0, holder_err)

        # The holder exited without unlocking explicitly, so the lock must be
        # free now — that is the process-lifetime guarantee retire relies on.
        fd = kbcompile.acquire_compile_lock(self.lock)
        self.assertIsNotNone(fd)
        kbcompile.release_compile_lock()


if __name__ == "__main__":
    unittest.main(verbosity=2)
