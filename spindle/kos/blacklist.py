"""Blacklist and rejection log management for KOS."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, List, Optional, Set

_REJECTIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS rejections (
    id               INTEGER PRIMARY KEY,
    rejected_term    TEXT    NOT NULL,
    source_doc_id    TEXT    NOT NULL,
    chunk_index      INTEGER,
    rejection_reason TEXT,
    rejected_at      TEXT    NOT NULL,
    rejected_by      TEXT
);
CREATE INDEX IF NOT EXISTS idx_rej_term   ON rejections(rejected_term);
CREATE INDEX IF NOT EXISTS idx_rej_doc    ON rejections(source_doc_id);
"""


def load_blacklist(path: Path) -> Set[str]:
    """Load a blacklist file into a set of lower-cased terms."""
    if not path.exists():
        return set()
    terms: Set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            terms.add(stripped.lower())
    return terms


class RejectionLog:
    """SQLite-backed rejection log for staging candidates."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.executescript(_REJECTIONS_SCHEMA)
        self._conn.commit()

    @contextmanager
    def _tx(self) -> Generator[sqlite3.Connection, None, None]:
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def add(
        self,
        term: str,
        source_doc_id: str,
        chunk_index: Optional[int] = None,
        rejection_reason: Optional[str] = None,
        rejected_by: Optional[str] = None,
    ) -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO rejections "
                "(rejected_term, source_doc_id, chunk_index, rejection_reason, "
                "rejected_at, rejected_by) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    term,
                    source_doc_id,
                    chunk_index,
                    rejection_reason,
                    datetime.now(timezone.utc).isoformat(),
                    rejected_by,
                ),
            )

    def query(
        self,
        term: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> List[dict]:
        sql = "SELECT id, rejected_term, source_doc_id, chunk_index, rejection_reason, rejected_at, rejected_by FROM rejections WHERE 1=1"
        params: list = []
        if term:
            sql += " AND rejected_term = ?"
            params.append(term)
        if doc_id:
            sql += " AND source_doc_id = ?"
            params.append(doc_id)
        rows = self._conn.execute(sql, params).fetchall()
        keys = ["id", "rejected_term", "source_doc_id", "chunk_index",
                "rejection_reason", "rejected_at", "rejected_by"]
        return [dict(zip(keys, row)) for row in rows]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "RejectionLog":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
