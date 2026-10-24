"""Fail-closed private search-attempt logging for the v2 KB handler.

The production paths are constants. Tests may inject a private directory and rules
file through the QueryLogger constructor; no request or environment variable can
select a production log destination.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import math
import os
import re
import stat
import sys
import threading
import time
from builtins import print
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable


QUERY_LOG_ENABLED_ENV = "KB_QUERY_LOG_ENABLED"
QUERY_LOG_DIR = Path("/opt/kb/logs/queries")
SECRET_RULES_PATH = Path("/opt/kb/secret_patterns.json")
REDACTED = "<REDACTED>"
RETENTION_DAYS = 180
MAX_RULES_BYTES = 256 * 1024
MAX_PATTERN_COUNT = 256
MAX_PATTERN_CHARS = 4096
MAX_RECORD_BYTES = 32 * 1024
LOCK_WAIT_SECONDS = 0.010
LOCK_POLL_SECONDS = 0.001
MAX_WARN_INTERVAL_SECONDS = 60.0
MAX_RETENTION_ENTRIES = 10000

_SCOPE = {"homelab", "ai", "both", "auto"}
_SELECTED_SCOPE = _SCOPE | {"none"}
_FAILURE_CATEGORIES = {
    "forbidden", "conflict", "unavailable", "retrieval_unavailable",
    "reranker_unavailable", "http_error", "internal_error",
}
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_SAFE_REF = re.compile(r"^(?:homelab|ai):[1-9][0-9]{0,18}$")
_LOG_NAME = re.compile(r"^queries-([0-9]{4}-[0-9]{2}-[0-9]{2})\.jsonl$")
_INTENT_NAME = ".query-log.intent"
_POISON_NAME = ".query-log.poison"
_INTENT_BYTES = b"append-in-progress-v1\n"
_POISON_BYTES = b"append-state-uncertain-v1\n"


class QueryLogError(Exception):
    """A deliberately detail-free logger failure."""


class _LockBusy(QueryLogError):
    pass


@dataclass(frozen=True)
class _CompiledPattern:
    name: str
    regex: re.Pattern[str]
    capture_group: int


@dataclass(frozen=True)
class _Rules:
    identity: tuple[int, int, int, int, int, int, int]
    digest: str
    version: int
    patterns: tuple[_CompiledPattern, ...]


def _open_absolute_directory(path: Path) -> int:
    if not path.is_absolute() or ".." in path.parts:
        raise QueryLogError("unsafe_path")
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    nofollow_flag = getattr(os, "O_NOFOLLOW", 0)
    if not directory_flag or not nofollow_flag:
        raise QueryLogError("safe_open_unavailable")
    fd = os.open("/", os.O_RDONLY | directory_flag | getattr(os, "O_CLOEXEC", 0))
    try:
        for component in path.parts[1:]:
            next_fd = os.open(
                component,
                os.O_RDONLY | directory_flag | nofollow_flag | getattr(os, "O_CLOEXEC", 0),
                dir_fd=fd,
            )
            os.close(fd)
            fd = next_fd
        if not stat.S_ISDIR(os.fstat(fd).st_mode):
            raise QueryLogError("unsafe_directory")
        return fd
    except Exception:
        os.close(fd)
        raise


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise QueryLogError("invalid_rules")
        result[key] = value
    return result


def _compile_pattern(name: object, expression: object, capture_group: object) -> _CompiledPattern:
    if (type(name) is not str or not name or len(name) > 80
            or type(expression) is not str or not expression
            or len(expression) > MAX_PATTERN_CHARS
            or type(capture_group) is not int or capture_group < 0):
        raise QueryLogError("invalid_rules")
    try:
        compiled = re.compile(expression)
    except (re.error, OverflowError):
        raise QueryLogError("invalid_rules") from None
    if capture_group > compiled.groups or compiled.match("") is not None:
        raise QueryLogError("invalid_rules")
    return _CompiledPattern(name, compiled, capture_group)


def _validate_rules(raw: bytes, identity: tuple[int, int, int, int, int, int, int]) -> _Rules:
    try:
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=_json_object)
    except QueryLogError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        raise QueryLogError("invalid_rules") from None
    if not isinstance(document, dict):
        raise QueryLogError("invalid_rules")
    version = document.get("version")
    if type(version) is not int or version != 1:
        raise QueryLogError("invalid_rules")
    legacy = document.get("patterns")
    query_only = document.get("query_redaction_patterns")
    if (not isinstance(legacy, list) or not legacy or len(legacy) > MAX_PATTERN_COUNT
            or not isinstance(query_only, list) or not query_only
            or len(legacy) + len(query_only) > MAX_PATTERN_COUNT):
        raise QueryLogError("invalid_rules")

    compiled_patterns: list[_CompiledPattern] = []
    names: set[str] = set()
    for item in legacy:
        if not isinstance(item, dict):
            raise QueryLogError("invalid_rules")
        name = item.get("name")
        action = item.get("action")
        if action not in {"redact", "log"}:
            raise QueryLogError("invalid_rules")
        pattern = _compile_pattern(name, item.get("regex"), item.get("capture_group"))
        if pattern.name in names:
            raise QueryLogError("invalid_rules")
        names.add(pattern.name)
        compiled_patterns.append(pattern)

    for item in query_only:
        if not isinstance(item, dict) or set(item) != {"name", "regex", "capture_group"}:
            raise QueryLogError("invalid_rules")
        pattern = _compile_pattern(item["name"], item["regex"], item["capture_group"])
        if pattern.name in names:
            raise QueryLogError("invalid_rules")
        names.add(pattern.name)
        compiled_patterns.append(pattern)

    return _Rules(identity, hashlib.sha256(raw).hexdigest(), version, tuple(compiled_patterns))


def _open_rules_file(path: Path) -> tuple[bytes, tuple[int, int, int, int, int, int, int]]:
    parent_fd = _open_absolute_directory(path.parent)
    fd = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        fd = os.open(path.name, flags, dir_fd=parent_fd)
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or info.st_uid not in {os.getuid(), 0}
                or info.st_mode & 0o022
                or info.st_size < 1 or info.st_size > MAX_RULES_BYTES):
            raise QueryLogError("unsafe_rules_file")
        chunks = bytearray()
        while len(chunks) <= MAX_RULES_BYTES:
            chunk = os.read(fd, min(65536, MAX_RULES_BYTES + 1 - len(chunks)))
            if not chunk:
                break
            chunks.extend(chunk)
        if len(chunks) > MAX_RULES_BYTES:
            raise QueryLogError("rules_too_large")
        final = os.fstat(fd)
        if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != (
            final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns
        ):
            raise QueryLogError("rules_changed_during_read")
        identity = (
            final.st_dev, final.st_ino, final.st_uid, final.st_mode,
            final.st_nlink, final.st_size, final.st_mtime_ns,
        )
        return bytes(chunks), identity
    except QueryLogError:
        raise
    except OSError:
        raise QueryLogError("rules_unavailable") from None
    finally:
        if fd is not None:
            os.close(fd)
        os.close(parent_fd)


class _RulesCache:
    def __init__(self, path: Path):
        self.path = path
        self.cached: _Rules | None = None
        self.lock = threading.Lock()

    def load(self) -> _Rules:
        with self.lock:
            try:
                raw, identity = _open_rules_file(self.path)
                digest = hashlib.sha256(raw).hexdigest()
                cached = self.cached
                if (cached is not None and cached.identity == identity
                        and cached.digest == digest and cached.version == 1):
                    return cached
                rules = _validate_rules(raw, identity)
                self.cached = rules
                return rules
            except Exception:
                # A missing or malformed replacement must not retain stale rules.
                self.cached = None
                raise QueryLogError("rules_unavailable") from None


def redact_text(value: str, rules: _Rules) -> tuple[str, bool]:
    if type(value) is not str:
        raise QueryLogError("invalid_record")
    spans: list[tuple[int, int]] = []
    for pattern in rules.patterns:
        for match in pattern.regex.finditer(value):
            captured = match.group(pattern.capture_group)
            start, end = match.span(0)
            if captured is None or captured == "" or start < 0 or end <= start:
                raise QueryLogError("invalid_capture")
            spans.append((start, end))
    if not spans:
        return value, False
    spans.sort()
    merged: list[list[int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    pieces: list[str] = []
    cursor = 0
    for start, end in merged:
        pieces.extend((value[cursor:start], REDACTED))
        cursor = end
    pieces.append(value[cursor:])
    return "".join(pieces), True


def _validate_record(fields: dict, rules: _Rules) -> tuple[dict, bool]:
    allowed = {
        "client", "scope", "selected_scope", "query", "query_alt",
        "query_alt_language", "ranked", "elapsed_ms", "status", "category",
    }
    if not isinstance(fields, dict) or set(fields) - allowed:
        raise QueryLogError("invalid_record")
    client = fields.get("client")
    scope = fields.get("scope")
    query = fields.get("query")
    elapsed = fields.get("elapsed_ms")
    if (type(client) is not str or not _SAFE_NAME.fullmatch(client)
            or scope not in _SCOPE or type(query) is not str
            or type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0):
        raise QueryLogError("invalid_record")
    result = {
        "client": client,
        "scope": scope,
        "query": None,
        "elapsed_ms": round(float(elapsed), 3),
    }
    query, redacted = redact_text(query, rules)
    result["query"] = query
    if "query_alt" in fields:
        alternate = fields["query_alt"]
        if alternate is not None:
            alternate, changed = redact_text(alternate, rules)
            redacted = redacted or changed
        result["query_alt"] = alternate
    if "query_alt_language" in fields:
        language = fields["query_alt_language"]
        if language is not None and language not in {"sr", "en"}:
            raise QueryLogError("invalid_record")
        result["query_alt_language"] = language
    if "selected_scope" in fields:
        selected = fields["selected_scope"]
        if selected not in _SELECTED_SCOPE:
            raise QueryLogError("invalid_record")
        result["selected_scope"] = selected
    if "ranked" in fields:
        ranked = fields["ranked"]
        if not isinstance(ranked, list) or len(ranked) > 10:
            raise QueryLogError("invalid_record")
        safe_ranked = []
        for item in ranked:
            if (not isinstance(item, dict) or set(item) != {"ref", "final_score"}
                    or type(item["ref"]) is not str or not _SAFE_REF.fullmatch(item["ref"])
                    or type(item["final_score"]) not in (int, float)
                    or not math.isfinite(item["final_score"])):
                raise QueryLogError("invalid_record")
            safe_ranked.append({"ref": item["ref"], "final_score": item["final_score"]})
        result["ranked"] = safe_ranked
    failure = "status" in fields or "category" in fields
    if failure:
        status = fields.get("status")
        category = fields.get("category")
        if (type(status) is not int or not 400 <= status <= 599
                or type(category) is not str or category not in _FAILURE_CATEGORIES):
            raise QueryLogError("invalid_record")
        result["status"] = status
        result["category"] = category
    elif "ranked" not in fields or "selected_scope" not in fields:
        raise QueryLogError("invalid_record")
    if redacted:
        result["redacted"] = True
    return result, redacted


def _check_private_directory(fd: int) -> None:
    info = os.fstat(fd)
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o700):
        raise QueryLogError("unsafe_directory")


def _open_locked(directory_fd: int) -> int:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow:
        raise QueryLogError("safe_open_unavailable")
    flags = os.O_RDWR | os.O_CREAT | nofollow | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(".query-log.lock", flags | os.O_EXCL, 0o600, dir_fd=directory_fd)
        try:
            os.fchmod(fd, 0o600)
            os.fsync(directory_fd)
        except Exception:
            os.close(fd)
            raise
    except FileExistsError:
        fd = os.open(".query-log.lock", os.O_RDWR | nofollow | getattr(os, "O_CLOEXEC", 0),
                     dir_fd=directory_fd)
    info = os.fstat(fd)
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1):
        os.close(fd)
        raise QueryLogError("unsafe_lock")
    deadline = time.monotonic() + LOCK_WAIT_SECONDS
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except BlockingIOError:
            if time.monotonic() >= deadline:
                os.close(fd)
                raise _LockBusy("lock_busy") from None
            time.sleep(LOCK_POLL_SECONDS)
        except OSError:
            os.close(fd)
            raise QueryLogError("lock_failed") from None


def _state_marker_exists(directory_fd: int, name: str) -> bool:
    try:
        info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    except OSError:
        raise QueryLogError("unsafe_state_marker") from None
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1
            or info.st_size > 128):
        raise QueryLogError("unsafe_state_marker")
    return True


def _check_write_barrier(directory_fd: int) -> None:
    if _state_marker_exists(directory_fd, _POISON_NAME):
        raise QueryLogError("poisoned")
    if _state_marker_exists(directory_fd, _INTENT_NAME):
        # A pending intent can be left by a crash or a failed rollback. It is
        # deliberately never repaired automatically, including after rotation.
        raise QueryLogError("intent_pending")


def _create_durable_marker(directory_fd: int, name: str, data: bytes) -> None:
    flags = (os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
             | getattr(os, "O_CLOEXEC", 0))
    try:
        fd = os.open(name, flags, 0o600, dir_fd=directory_fd)
    except FileExistsError:
        raise QueryLogError("intent_pending" if name == _INTENT_NAME else "poisoned") from None
    except OSError:
        raise QueryLogError("state_marker_failed") from None
    try:
        os.fchmod(fd, 0o600)
        if os.write(fd, data) != len(data):
            raise QueryLogError("state_marker_failed")
        os.fsync(fd)
        os.fsync(directory_fd)
    except Exception:
        # Leave any created marker in place: even a partial marker is a durable
        # fail-closed barrier once its directory entry has reached storage.
        raise QueryLogError("state_marker_failed") from None
    finally:
        os.close(fd)


def _poison_after_uncertain_state(directory_fd: int) -> None:
    try:
        _create_durable_marker(directory_fd, _POISON_NAME, _POISON_BYTES)
    except QueryLogError:
        # The already durable append intent remains the cross-process barrier.
        pass


def _clear_durable_intent(directory_fd: int) -> None:
    if not _state_marker_exists(directory_fd, _INTENT_NAME):
        raise QueryLogError("intent_clear_failed")
    try:
        os.unlink(_INTENT_NAME, dir_fd=directory_fd)
        os.fsync(directory_fd)
    except OSError:
        _poison_after_uncertain_state(directory_fd)
        try:
            # If the poison marker itself could not be committed, restore the
            # append intent as the conservative cross-process barrier.
            _create_durable_marker(directory_fd, _INTENT_NAME, _INTENT_BYTES)
        except QueryLogError:
            pass
        raise QueryLogError("intent_clear_failed") from None


def _open_daily_file(directory_fd: int, filename: str) -> tuple[int, bool]:
    flags = (os.O_RDWR | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
             | getattr(os, "O_CLOEXEC", 0))
    try:
        fd = os.open(filename, flags | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=directory_fd)
        created = True
        try:
            os.fchmod(fd, 0o600)
        except Exception:
            os.close(fd)
            raise
    except FileExistsError:
        fd = os.open(filename, flags, dir_fd=directory_fd)
        created = False
    info = os.fstat(fd)
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1):
        os.close(fd)
        raise QueryLogError("unsafe_log_file")
    if info.st_size:
        os.lseek(fd, -1, os.SEEK_END)
        if os.read(fd, 1) != b"\n":
            os.close(fd)
            raise QueryLogError("incomplete_log_tail")
    return fd, created


def _write_record(fd: int, line: bytes) -> None:
    start = os.fstat(fd).st_size
    try:
        written = os.write(fd, line)
        if written != len(line):
            raise QueryLogError("short_write")
        os.fsync(fd)
    except Exception:
        try:
            os.ftruncate(fd, start)
            os.fsync(fd)
        except Exception:
            raise QueryLogError("rollback_failed") from None
        raise QueryLogError("write_failed") from None


class QueryLogger:
    """Optional logger; all failures are generic and never change search behavior."""

    def __init__(self, *, enabled: bool, log_dir: Path = QUERY_LOG_DIR,
                 rules_path: Path = SECRET_RULES_PATH,
                 clock: Callable[[], datetime] | None = None):
        self.enabled = bool(enabled)
        self.log_dir = Path(log_dir)
        self.rules_cache = _RulesCache(Path(rules_path))
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self._warning_lock = threading.Lock()
        self._last_warning: dict[str, float] = {}
        self._failure_count: dict[str, int] = {}
        self._poisoned = False

    @classmethod
    def from_environment(cls) -> "QueryLogger":
        return cls(enabled=os.environ.get(QUERY_LOG_ENABLED_ENV) == "1")

    def _warn(self, reason: str) -> None:
        if reason not in {
            "rules_unavailable", "invalid_rules", "unsafe_rules_file", "rules_too_large",
            "invalid_record", "invalid_capture", "unsafe_directory", "unsafe_lock",
            "unsafe_log_file", "incomplete_log_tail", "lock_busy", "lock_failed",
            "write_failed", "rollback_failed", "poisoned", "record_too_large",
            "safe_open_unavailable", "unsafe_path", "rules_changed_during_read",
            "intent_pending", "unsafe_state_marker", "state_marker_failed",
            "intent_clear_failed",
        }:
            reason = "logger_failure"
        now = time.monotonic()
        with self._warning_lock:
            self._failure_count[reason] = self._failure_count.get(reason, 0) + 1
            previous = self._last_warning.get(reason, float("-inf"))
            if now - previous < MAX_WARN_INTERVAL_SECONDS:
                return
            self._last_warning[reason] = now
        try:
            print(f"[query-log] suppressed: {reason}", file=sys.stderr, flush=True)
        except (OSError, ValueError):
            # Diagnostics are best-effort and must never alter the search response.
            pass

    def record(self, fields: dict) -> bool:
        if not self.enabled:
            return False
        if self._poisoned:
            self._warn("poisoned")
            return False
        try:
            rules = self.rules_cache.load()
            safe, _redacted = _validate_record(fields, rules)
            timestamp = self.clock()
            if timestamp.tzinfo is None:
                raise QueryLogError("invalid_record")
            timestamp_utc = timestamp.astimezone(timezone.utc)
            safe["timestamp"] = timestamp_utc.isoformat(timespec="milliseconds").replace("+00:00", "Z")
            line = json.dumps(safe, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
            if len(line) > MAX_RECORD_BYTES:
                raise QueryLogError("record_too_large")
            directory_fd = _open_absolute_directory(self.log_dir)
            try:
                _check_private_directory(directory_fd)
                lock_fd = _open_locked(directory_fd)
                try:
                    _check_write_barrier(directory_fd)
                    filename = f"queries-{timestamp_utc.date().isoformat()}.jsonl"
                    log_fd, created = _open_daily_file(directory_fd, filename)
                    try:
                        if created:
                            os.fsync(directory_fd)
                        _create_durable_marker(directory_fd, _INTENT_NAME, _INTENT_BYTES)
                        try:
                            _write_record(log_fd, line)
                        except QueryLogError as exc:
                            if str(exc) == "rollback_failed":
                                self._poisoned = True
                                _poison_after_uncertain_state(directory_fd)
                            else:
                                _clear_durable_intent(directory_fd)
                            raise
                        _clear_durable_intent(directory_fd)
                    except QueryLogError as exc:
                        if str(exc) in {
                            "poisoned", "intent_pending", "intent_clear_failed",
                            "rollback_failed", "state_marker_failed",
                        }:
                            self._poisoned = True
                        raise
                    finally:
                        os.close(log_fd)
                finally:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    os.close(lock_fd)
            finally:
                os.close(directory_fd)
            return True
        except _LockBusy:
            self._warn("lock_busy")
        except QueryLogError as exc:
            self._warn(str(exc))
        except Exception:
            self._warn("logger_failure")
        return False

    def prune(self, now: datetime | None = None) -> int:
        """Delete only safe, owned daily files strictly older than 180 UTC days."""
        removed = 0
        try:
            directory_fd = _open_absolute_directory(self.log_dir)
            try:
                _check_private_directory(directory_fd)
                lock_fd = _open_locked(directory_fd)
                try:
                    _check_write_barrier(directory_fd)
                    current = (now or self.clock()).astimezone(timezone.utc).date()
                    cutoff = current - timedelta(days=RETENTION_DAYS)
                    names = os.listdir(directory_fd)
                    if len(names) > MAX_RETENTION_ENTRIES:
                        raise QueryLogError("retention_scan_limit")
                    for name in names:
                        match = _LOG_NAME.fullmatch(name)
                        if not match:
                            continue
                        try:
                            file_day = date.fromisoformat(match.group(1))
                        except ValueError:
                            continue
                        if file_day >= cutoff:
                            continue
                        try:
                            info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                        except OSError:
                            continue
                        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                                or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1):
                            continue
                        try:
                            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
                        except OSError:
                            continue
                        try:
                            opened = os.fstat(fd)
                            if ((opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino)
                                    or not stat.S_ISREG(opened.st_mode)
                                    or opened.st_nlink != 1 or opened.st_uid != os.getuid()
                                    or stat.S_IMODE(opened.st_mode) != 0o600):
                                continue
                            os.unlink(name, dir_fd=directory_fd)
                            removed += 1
                        finally:
                            os.close(fd)
                    if removed:
                        os.fsync(directory_fd)
                finally:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    os.close(lock_fd)
            finally:
                os.close(directory_fd)
        except _LockBusy:
            self._warn("lock_busy")
        except QueryLogError as exc:
            self._warn(str(exc))
        except Exception:
            self._warn("logger_failure")
        return removed


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args != ["--prune"]:
        print("usage: query_log.py --prune", file=sys.stderr)
        return 2
    removed = QueryLogger(enabled=False).prune()
    print(f"[query-log-retention] removed={removed}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
