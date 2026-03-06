"""Stage 1: Document ingestion and version control.

Handles Docling conversion, content-hash change detection, and document catalog
management (SQLite-backed DocumentRecord storage).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from spindle.preprocessing.models import DocumentRecord

_CATALOG_SCHEMA = """
CREATE TABLE IF NOT EXISTS document_catalog (
    doc_id              TEXT PRIMARY KEY,
    source_path         TEXT NOT NULL,
    content_hash        TEXT NOT NULL,
    docling_json_path   TEXT NOT NULL,
    last_ingested       TEXT NOT NULL,
    version             INTEGER NOT NULL DEFAULT 1,
    metadata_json       TEXT
);
CREATE INDEX IF NOT EXISTS idx_catalog_hash ON document_catalog(content_hash);
CREATE INDEX IF NOT EXISTS idx_catalog_path ON document_catalog(source_path);
"""


class DocumentCatalog:
    """SQLite-backed catalog of ingested documents with change detection."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.executescript(_CATALOG_SCHEMA)
        self._conn.commit()

    @contextmanager
    def _tx(self) -> Generator[sqlite3.Connection, None, None]:
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def get(self, doc_id: str) -> Optional[DocumentRecord]:
        row = self._conn.execute(
            "SELECT doc_id, source_path, content_hash, docling_json_path, "
            "last_ingested, version, metadata_json "
            "FROM document_catalog WHERE doc_id = ?",
            (doc_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def upsert(self, record: DocumentRecord) -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO document_catalog "
                "(doc_id, source_path, content_hash, docling_json_path, "
                "last_ingested, version, metadata_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(doc_id) DO UPDATE SET "
                "source_path=excluded.source_path, "
                "content_hash=excluded.content_hash, "
                "docling_json_path=excluded.docling_json_path, "
                "last_ingested=excluded.last_ingested, "
                "version=excluded.version, "
                "metadata_json=excluded.metadata_json",
                (
                    record.doc_id,
                    record.source_path,
                    record.content_hash,
                    record.docling_json_path,
                    record.last_ingested.isoformat(),
                    record.version,
                    json.dumps(record.metadata) if record.metadata else None,
                ),
            )

    def all_records(self) -> List[DocumentRecord]:
        rows = self._conn.execute(
            "SELECT doc_id, source_path, content_hash, docling_json_path, "
            "last_ingested, version, metadata_json FROM document_catalog"
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    @staticmethod
    def _row_to_record(row: tuple) -> DocumentRecord:
        return DocumentRecord(
            doc_id=row[0],
            source_path=row[1],
            content_hash=row[2],
            docling_json_path=row[3],
            last_ingested=datetime.fromisoformat(row[4]),
            version=row[5],
            metadata=json.loads(row[6]) if row[6] else {},
        )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "DocumentCatalog":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def compute_content_hash(path: Path) -> str:
    """SHA-256 hash of a file's raw bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def convert_document(source_path: Path, output_dir: Path) -> Dict[str, Any]:
    """Convert a document to structured JSON using Docling.

    Args:
        source_path: Path to the source document (PDF, DOCX, Markdown, etc.)
        output_dir: Directory where Docling JSON output will be written.

    Returns:
        Dict with keys ``json_path`` (Path) and ``docling_result`` (raw Docling
        output object).

    Raises:
        ImportError: If ``docling`` is not installed.
    """
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as exc:
        raise ImportError(
            "docling is required for document conversion. "
            "Install it with: uv pip install docling"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    converter = DocumentConverter()
    result = converter.convert(str(source_path))
    json_path = output_dir / f"{source_path.stem}.json"
    json_path.write_text(result.document.export_to_json(), encoding="utf-8")
    return {"json_path": json_path, "docling_result": result}


def extract_section_path(docling_result: Any, chunk_start: int) -> Optional[List[str]]:
    """Extract a section heading path for a given character offset.

    Walks the Docling document hierarchy to find which section headings
    contain the character at ``chunk_start``.  Returns a list of heading
    strings from outermost to innermost (e.g. ``["Introduction", "Background"]``).

    Returns None if the structure is unavailable.
    """
    try:
        doc = docling_result.document
        # Attempt to use Docling's hierarchical structure if available
        if hasattr(doc, "texts"):
            path: List[str] = []
            for item in doc.texts:
                label = getattr(item, "label", "")
                if "heading" in str(label).lower():
                    # Rough heuristic: include headings whose character range precedes
                    # the chunk start.  A more precise implementation would walk
                    # Docling's page/section tree.
                    item_start = getattr(getattr(item, "prov", [None])[0], "char_offset", None)
                    if item_start is not None and item_start <= chunk_start:
                        path = [item.text]
            return path if path else None
    except Exception:
        pass
    return None


class DocumentIngestionStage:
    """Stage 1 orchestrator: mirror → hash → convert → catalog."""

    def __init__(
        self,
        catalog: DocumentCatalog,
        docling_output_dir: Path,
    ) -> None:
        self._catalog = catalog
        self._output_dir = docling_output_dir

    def process(
        self,
        source_paths: List[Path],
    ) -> List[DocumentRecord]:
        """Process a list of source documents.

        Skips unchanged documents (hash match).  Returns DocumentRecord objects
        for all documents that were successfully ingested.
        """
        records: List[DocumentRecord] = []
        for path in source_paths:
            doc_id = str(path)
            content_hash = compute_content_hash(path)
            existing = self._catalog.get(doc_id)
            if existing and existing.content_hash == content_hash:
                records.append(existing)
                continue

            try:
                conversion = convert_document(path, self._output_dir)
            except ImportError:
                # Docling not installed — create a minimal record using raw text
                json_path = self._output_dir / f"{path.stem}.json"
                json_path.parent.mkdir(parents=True, exist_ok=True)
                raw_text = path.read_text(encoding="utf-8", errors="replace")
                json_path.write_text(
                    json.dumps({"text": raw_text, "source": str(path)}),
                    encoding="utf-8",
                )
                conversion = {"json_path": json_path, "docling_result": None}

            version = (existing.version + 1) if existing else 1
            record = DocumentRecord(
                doc_id=doc_id,
                source_path=str(path),
                content_hash=content_hash,
                docling_json_path=str(conversion["json_path"]),
                last_ingested=datetime.utcnow(),
                version=version,
                metadata={"source": str(path)},
            )
            self._catalog.upsert(record)
            records.append(record)

        return records
