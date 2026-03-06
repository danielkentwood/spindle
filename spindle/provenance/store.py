"""ProvenanceStore: SQLite-backed provenance side-table.

Three normalized tables support two access patterns:
- Pattern 1 (point lookup): edge/entity → ProvenanceObject → source docs + spans
- Pattern 2 (reverse lookup): doc changed → all affected ProvenanceObjects
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, List, Optional

from spindle.provenance.models import EvidenceSpan, ProvenanceDoc, ProvenanceObject

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS provenance_objects (
    object_id   TEXT PRIMARY KEY,
    object_type TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_prov_type ON provenance_objects(object_type);

CREATE TABLE IF NOT EXISTS provenance_docs (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    provenance_object_id TEXT NOT NULL
        REFERENCES provenance_objects(object_id) ON DELETE CASCADE,
    doc_id               TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_provdoc_object ON provenance_docs(provenance_object_id);
CREATE INDEX IF NOT EXISTS idx_provdoc_docid  ON provenance_docs(doc_id);

CREATE TABLE IF NOT EXISTS evidence_spans (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    provenance_doc_id INTEGER NOT NULL
        REFERENCES provenance_docs(id) ON DELETE CASCADE,
    text              TEXT    NOT NULL,
    start_offset      INTEGER,
    end_offset        INTEGER,
    section_path      TEXT
);
CREATE INDEX IF NOT EXISTS idx_spans_provdoc  ON evidence_spans(provenance_doc_id);
CREATE INDEX IF NOT EXISTS idx_spans_section  ON evidence_spans(section_path);
"""


class ProvenanceStore:
    """Normalized SQLite provenance store.

    Args:
        db_path: Path to the SQLite file. Pass ``":memory:"`` for an
                 in-memory database (useful for tests).
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _transaction(self) -> Generator[sqlite3.Connection, None, None]:
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def create_provenance(
        self,
        object_id: str,
        object_type: str,
        docs: List[ProvenanceDoc],
    ) -> None:
        """Insert a new ProvenanceObject with its docs and spans.

        Silently replaces an existing object with the same ``object_id``
        (useful for re-extraction scenarios).
        """
        with self._transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO provenance_objects (object_id, object_type) "
                "VALUES (?, ?)",
                (object_id, object_type),
            )
            for doc in docs:
                cursor = conn.execute(
                    "INSERT INTO provenance_docs (provenance_object_id, doc_id) "
                    "VALUES (?, ?)",
                    (object_id, doc.doc_id),
                )
                provenance_doc_id = cursor.lastrowid
                for span in doc.spans:
                    conn.execute(
                        "INSERT INTO evidence_spans "
                        "(provenance_doc_id, text, start_offset, end_offset, section_path) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (
                            provenance_doc_id,
                            span.text,
                            span.start_offset,
                            span.end_offset,
                            span.section_path_json(),
                        ),
                    )

    def delete_provenance(self, object_id: str) -> None:
        """Delete a provenance object and cascade to docs/spans."""
        with self._transaction() as conn:
            conn.execute(
                "DELETE FROM provenance_objects WHERE object_id = ?",
                (object_id,),
            )

    def update_provenance_for_doc(
        self,
        object_id: str,
        doc_id: str,
        new_spans: List[EvidenceSpan],
    ) -> None:
        """Replace all evidence spans for a specific (object_id, doc_id) pair.

        Used when a document is re-extracted: old spans are removed and
        replaced with newly computed spans.
        """
        with self._transaction() as conn:
            # Find the provenance_doc row
            row = conn.execute(
                "SELECT id FROM provenance_docs "
                "WHERE provenance_object_id = ? AND doc_id = ?",
                (object_id, doc_id),
            ).fetchone()
            if row is None:
                return
            provenance_doc_id = row[0]
            # Delete old spans (cascade would also work, but explicit is clearer)
            conn.execute(
                "DELETE FROM evidence_spans WHERE provenance_doc_id = ?",
                (provenance_doc_id,),
            )
            # Insert new spans
            for span in new_spans:
                conn.execute(
                    "INSERT INTO evidence_spans "
                    "(provenance_doc_id, text, start_offset, end_offset, section_path) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        provenance_doc_id,
                        span.text,
                        span.start_offset,
                        span.end_offset,
                        span.section_path_json(),
                    ),
                )

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_provenance(self, object_id: str) -> Optional[ProvenanceObject]:
        """Point lookup: return full ProvenanceObject for a graph element.

        Returns None if the object_id is not found.
        """
        row = self._conn.execute(
            "SELECT object_type FROM provenance_objects WHERE object_id = ?",
            (object_id,),
        ).fetchone()
        if row is None:
            return None

        prov_obj = ProvenanceObject(object_id=object_id, object_type=row[0])

        doc_rows = self._conn.execute(
            "SELECT id, doc_id FROM provenance_docs WHERE provenance_object_id = ?",
            (object_id,),
        ).fetchall()

        for doc_id_row_id, doc_id in doc_rows:
            span_rows = self._conn.execute(
                "SELECT text, start_offset, end_offset, section_path "
                "FROM evidence_spans WHERE provenance_doc_id = ?",
                (doc_id_row_id,),
            ).fetchall()
            spans = [
                EvidenceSpan(
                    text=s[0],
                    start_offset=s[1],
                    end_offset=s[2],
                    section_path=EvidenceSpan.parse_section_path(s[3]),
                )
                for s in span_rows
            ]
            prov_obj.docs.append(ProvenanceDoc(doc_id=doc_id, id=doc_id_row_id, spans=spans))

        return prov_obj

    def get_affected_objects(self, doc_id: str) -> List[dict]:
        """Reverse lookup: return all (object_type, object_id) referencing doc_id.

        Used when a source document changes — caller can cascade updates to
        all graph elements that were sourced from this document.
        """
        rows = self._conn.execute(
            "SELECT po.object_type, po.object_id "
            "FROM provenance_docs pd "
            "JOIN provenance_objects po ON po.object_id = pd.provenance_object_id "
            "WHERE pd.doc_id = ?",
            (doc_id,),
        ).fetchall()
        return [{"object_type": r[0], "object_id": r[1]} for r in rows]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    def __enter__(self) -> "ProvenanceStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
