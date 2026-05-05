"""Persistent local memory store for original-llm chat sessions.

Single SQLite table ``memory`` keyed by ``id`` with columns
``key`` / ``value`` / ``importance`` (1-5) / ``created_at`` / ``updated_at``.

The store is purely local — no network calls, no remote sync. The default
location follows XDG: ``$XDG_DATA_HOME/original-llm/memory.db``, falling back
to ``~/.local/share/original-llm/memory.db``. Override with the
``ORIGINAL_LLM_MEMORY_DB`` env var or by passing ``path=`` to ``MemoryStore``.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_MEMORY_DB_ENV = "ORIGINAL_LLM_MEMORY_DB"
MIN_IMPORTANCE = 1
MAX_IMPORTANCE = 5
DEFAULT_IMPORTANCE = 3
DEFAULT_INJECT_LIMIT = 3

SCHEMA = """
CREATE TABLE IF NOT EXISTS memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    importance INTEGER NOT NULL DEFAULT 3
        CHECK (importance BETWEEN 1 AND 5),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memory_imp_updated
    ON memory(importance DESC, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_memory_key ON memory(key);
"""


def default_memory_path() -> Path:
    override = os.environ.get(DEFAULT_MEMORY_DB_ENV)
    if override:
        return Path(override).expanduser()
    base = os.environ.get("XDG_DATA_HOME")
    base_path = Path(base).expanduser() if base else Path.home() / ".local" / "share"
    return base_path / "original-llm" / "memory.db"


def _clamp_importance(value: int) -> int:
    if value < MIN_IMPORTANCE:
        return MIN_IMPORTANCE
    if value > MAX_IMPORTANCE:
        return MAX_IMPORTANCE
    return value


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    id: int
    key: str
    value: str
    importance: int
    created_at: str
    updated_at: str


def _row_to_entry(row: sqlite3.Row) -> MemoryEntry:
    return MemoryEntry(
        id=row["id"],
        key=row["key"],
        value=row["value"],
        importance=row["importance"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


class MemoryStore:
    """Thin wrapper around a single ``memory`` SQLite table."""

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        if path is None:
            path = default_memory_path()
        if str(path) == ":memory:":
            self.path: Path = Path(":memory:")
            self._conn = sqlite3.connect(":memory:")
        else:
            self.path = Path(path).expanduser()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    # --- lifecycle ---

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> MemoryStore:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    # --- write ops ---

    def add(
        self,
        key: str,
        value: str,
        importance: int = DEFAULT_IMPORTANCE,
    ) -> MemoryEntry:
        clean_key = key.strip()
        clean_value = value.strip()
        if not clean_key:
            raise ValueError("key must not be empty")
        if not clean_value:
            raise ValueError("value must not be empty")
        clamped = _clamp_importance(int(importance))
        now = self._now()
        cur = self._conn.execute(
            "INSERT INTO memory(key, value, importance, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (clean_key, clean_value, clamped, now, now),
        )
        self._conn.commit()
        entry = self.get(int(cur.lastrowid))
        assert entry is not None
        return entry

    def update(
        self,
        entry_id: int,
        *,
        value: str | None = None,
        importance: int | None = None,
        key: str | None = None,
    ) -> MemoryEntry | None:
        sets: list[str] = []
        args: list[object] = []
        if value is not None:
            clean = value.strip()
            if not clean:
                raise ValueError("value must not be empty")
            sets.append("value = ?")
            args.append(clean)
        if importance is not None:
            sets.append("importance = ?")
            args.append(_clamp_importance(int(importance)))
        if key is not None:
            clean_key = key.strip()
            if not clean_key:
                raise ValueError("key must not be empty")
            sets.append("key = ?")
            args.append(clean_key)
        if not sets:
            return self.get(entry_id)
        sets.append("updated_at = ?")
        args.append(self._now())
        args.append(entry_id)
        self._conn.execute(
            f"UPDATE memory SET {', '.join(sets)} WHERE id = ?",
            args,
        )
        self._conn.commit()
        return self.get(entry_id)

    def delete(self, entry_id: int) -> bool:
        cur = self._conn.execute("DELETE FROM memory WHERE id = ?", (entry_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def delete_by_key(self, key: str) -> int:
        cur = self._conn.execute("DELETE FROM memory WHERE key = ?", (key.strip(),))
        self._conn.commit()
        return cur.rowcount

    def clear(self) -> int:
        cur = self._conn.execute("DELETE FROM memory")
        self._conn.commit()
        return cur.rowcount

    # --- read ops ---

    def get(self, entry_id: int) -> MemoryEntry | None:
        row = self._conn.execute(
            "SELECT * FROM memory WHERE id = ?",
            (entry_id,),
        ).fetchone()
        return _row_to_entry(row) if row else None

    def find_by_key(self, key: str) -> list[MemoryEntry]:
        rows = self._conn.execute(
            "SELECT * FROM memory WHERE key = ? "
            "ORDER BY importance DESC, updated_at DESC, id DESC",
            (key.strip(),),
        ).fetchall()
        return [_row_to_entry(row) for row in rows]

    def bump_or_add(
        self,
        key: str,
        value: str,
        *,
        importance_delta: int = 1,
        max_importance: int = MAX_IMPORTANCE,
    ) -> MemoryEntry:
        """If an entry with ``key`` already exists, raise its importance by
        ``importance_delta`` (capped at ``max_importance``) and update value /
        timestamp. Otherwise insert a new entry at importance ``MIN_IMPORTANCE``.

        The returned entry is the post-update / newly inserted row.
        """
        existing = self.find_by_key(key)
        if not existing:
            return self.add(
                key,
                value,
                importance=max(MIN_IMPORTANCE, min(max_importance, MIN_IMPORTANCE)),
            )
        head = existing[0]
        new_importance = min(max_importance, head.importance + importance_delta)
        updated = self.update(
            head.id,
            value=value,
            importance=new_importance,
        )
        assert updated is not None
        return updated

    def list_all(self) -> list[MemoryEntry]:
        rows = self._conn.execute(
            "SELECT * FROM memory ORDER BY importance DESC, updated_at DESC, id DESC"
        ).fetchall()
        return [_row_to_entry(row) for row in rows]

    def contains_word(self, word: str) -> bool:
        """True if ``word`` appears as (or inside) any stored key or value.

        Used by the unknown-word detector to decide whether the input is
        already covered by existing memory entries.
        """
        target = word.strip()
        if not target:
            return False
        rows = self._conn.execute("SELECT key, value FROM memory").fetchall()
        for row in rows:
            key = row["key"] or ""
            value = row["value"] or ""
            if target == key or target == value:
                return True
            if target in key or target in value:
                return True
            if key and key in target:
                return True
            if value and value in target:
                return True
        return False

    def select_relevant(
        self,
        query: str | None = None,
        limit: int = DEFAULT_INJECT_LIMIT,
    ) -> list[MemoryEntry]:
        rows = self.list_all()
        if not rows or limit <= 0:
            return []
        normalized = (query or "").strip()
        if not normalized:
            return rows[:limit]

        def score(entry: MemoryEntry) -> tuple[int, str]:
            base = entry.importance * 10
            if entry.key and entry.key in normalized:
                base += 50
            if entry.value and entry.value in normalized:
                base += 30
            for char in entry.key:
                if char and char in normalized:
                    base += 1
            return (-base, entry.updated_at)  # negative for descending order

        ordered = sorted(rows, key=score)
        # sort returned ascending of the tuple, so ordered[0] is highest score
        return ordered[:limit]
