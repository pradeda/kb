"""Regression tests for compile.retire_entry.

Written for the 2026-08-12 health alert: ChromaDB held vector 584 whose SQLite
row was already gone, and retire_entry refused that state with LookupError — so
the documented recovery tool could not repair the one failure --health reports.

Run from the repo:  python3 -m unittest test_retire_orphan
Run from /opt/kb:   python3 /opt/kb/tests/test_retire_orphan.py
"""
import importlib.util
import pathlib
import sqlite3
import tempfile
import unittest

# Resolve compile.py relative to this file so the test always exercises the copy
# it ships with: a sibling in the kb-go repo, the parent directory under
# /opt/kb/tests. A hardcoded "/opt/kb" would make the repo test the deployed
# file instead, so a stale repo copy would still pass — the exact drift that let
# the orphan fix live only in production until 2026-08-12.
_HERE = pathlib.Path(__file__).resolve().parent
for _candidate in (_HERE / "compile.py", _HERE.parent / "compile.py"):
    if _candidate.exists():
        MODULE_PATH = _candidate
        break
else:
    raise RuntimeError(f"compile.py not found next to or above {_HERE}")

SPEC = importlib.util.spec_from_file_location("kb_compile", MODULE_PATH)
kbcompile = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(kbcompile)


class StubCollection:
    """Minimal stand-in for the Chroma collection surface retire_entry uses."""

    def __init__(self, ids):
        self.ids = set(ids)
        self.deleted = []

    def delete(self, ids):
        self.deleted.append(list(ids))
        for i in ids:
            self.ids.discard(i)

    def get(self, ids, include=None):
        return {"ids": [i for i in ids if i in self.ids]}


def make_db(rows):
    handle = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    handle.close()
    connection = sqlite3.connect(handle.name)
    connection.execute(
        "CREATE TABLE entries (id INTEGER PRIMARY KEY, title TEXT, raw_path TEXT)"
    )
    connection.executemany("INSERT INTO entries VALUES (?,?,?)", rows)
    connection.commit()
    connection.close()
    return handle.name


class RetireOrphanTests(unittest.TestCase):
    def test_orphan_vector_without_sqlite_row_is_removed(self):
        """The 584 case: vector present, row already gone."""
        db_path = make_db([(585, "kept", None)])
        collection = StubCollection({"584", "585"})

        report = kbcompile.retire_entry(584, collection=collection, db_path=db_path)

        self.assertNotIn("584", collection.ids, "orphan vector survived retire")
        self.assertIn("585", collection.ids, "retire touched an unrelated vector")
        self.assertTrue(report.get("orphan_vector_only"))
        self.assertFalse(report.get("raw_removed"))

    def test_unknown_id_in_both_layers_still_raises(self):
        """The guard must keep rejecting a typo'd id that exists nowhere."""
        db_path = make_db([(585, "kept", None)])
        collection = StubCollection({"585"})

        with self.assertRaises(LookupError):
            kbcompile.retire_entry(999, collection=collection, db_path=db_path)

        self.assertEqual(collection.ids, {"585"})

    def test_normal_retire_still_removes_row_and_vector(self):
        """Existing behaviour must not regress."""
        raw = tempfile.NamedTemporaryFile(suffix=".md", delete=False)
        raw.write(b"body")
        raw.close()
        db_path = make_db([(585, "doomed", raw.name)])
        collection = StubCollection({"585"})

        report = kbcompile.retire_entry(585, collection=collection, db_path=db_path)

        self.assertEqual(collection.ids, set())
        self.assertTrue(report["raw_removed"])
        self.assertFalse(pathlib.Path(raw.name).exists())
        connection = sqlite3.connect(db_path)
        remaining = connection.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
        connection.close()
        self.assertEqual(remaining, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
