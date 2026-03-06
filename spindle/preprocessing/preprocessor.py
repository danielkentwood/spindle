"""SpindlePreprocessor: orchestrates ingestion → chunking → coref resolution.

This is the v2 preprocessing pipeline that replaces the LangChain-based
ingestion pipeline for use with spindle-eval.  The existing
``spindle/ingestion/`` path remains functional for backward compatibility.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from spindle.preprocessing.chunking import chunk_document
from spindle.preprocessing.coref import resolve_coreferences_for_document
from spindle.preprocessing.ingestion import DocumentCatalog, DocumentIngestionStage
from spindle.preprocessing.models import Chunk, DocumentRecord


class SpindlePreprocessor:
    """Orchestrates document ingestion, chunking, and coreference resolution.

    Satisfies spindle-eval's ``Preprocessor`` protocol when spindle-eval is
    installed:  ``__call__(cfg) -> list[Chunk]``.

    Stages:
        1. Ingestion — Docling conversion + change detection + catalog
        2. Chunking — Chonkie recursive chunker with overlap refinement
        3. Coref resolution — fastcoref per-document, projected onto chunks

    Args:
        documents: List of source document paths.  If None, no documents
                   will be ingested unless paths are passed to ``__call__``.
        catalog_path: Path to the SQLite document catalog.  Defaults to
                      in-memory (``":memory:"``), which means no persistence
                      between runs.
        docling_output_dir: Directory for Docling JSON output files.
        blacklist_path: Optional path to ``kos/blacklist.txt`` for filtering
                        coref mentions.
        tracker: Optional Tracker (NoOpTracker by default).
    """

    def __init__(
        self,
        documents: Optional[List[str | Path]] = None,
        catalog_path: str | Path = ":memory:",
        docling_output_dir: Optional[Path] = None,
        blacklist_path: Optional[Path] = None,
        tracker=None,
    ) -> None:
        self._documents = [Path(d) for d in (documents or [])]
        self._catalog = DocumentCatalog(catalog_path)
        self._docling_output_dir = docling_output_dir or Path(".docling_cache")

        from spindle.tracking import NoOpTracker
        self._tracker = tracker if tracker is not None else NoOpTracker()

        self._blacklist: Set[str] = set()
        if blacklist_path and Path(blacklist_path).exists():
            self._blacklist = {
                line.strip().lower()
                for line in Path(blacklist_path).read_text().splitlines()
                if line.strip() and not line.startswith("#")
            }

        self._stage1 = DocumentIngestionStage(
            catalog=self._catalog,
            docling_output_dir=self._docling_output_dir,
        )

    def __call__(self, cfg: Any = None) -> List[Chunk]:
        """Run the full preprocessing pipeline.

        Args:
            cfg: Optional Hydra/DictConfig object supplying ``chunk_size``,
                 ``overlap``, ``strategy``, and ``coref_model`` settings.

        Returns:
            List of Chunk objects ready for downstream KOS/KG extraction.
        """
        chunk_size = _get(cfg, "chunk_size", 600)
        overlap = _get(cfg, "overlap", 0.10)
        strategy = _get(cfg, "strategy", "recursive")
        coref_model = _get(cfg, "coref_model", "FCoref")

        paths = self._documents
        if cfg is not None:
            extra_paths = _get(cfg, "documents", None)
            if extra_paths:
                paths = [Path(p) for p in extra_paths]

        self._tracker.log_event(
            "preprocessor", "stage1.start", {"doc_count": len(paths)}
        )
        doc_records = self._stage1.process(paths)
        self._tracker.log_event(
            "preprocessor", "stage1.complete", {"doc_count": len(doc_records)}
        )

        self._tracker.log_event(
            "preprocessor", "stage2.start", {"doc_count": len(doc_records)}
        )
        chunks_by_doc: Dict[str, List[Chunk]] = {}
        for record in doc_records:
            chunks_by_doc[record.doc_id] = chunk_document(
                record,
                chunk_size=chunk_size,
                overlap=overlap,
                strategy=strategy,
            )
        total_chunks = sum(len(v) for v in chunks_by_doc.values())
        self._tracker.log_event(
            "preprocessor", "stage2.complete", {"chunk_count": total_chunks}
        )

        if coref_model:
            self._tracker.log_event(
                "preprocessor", "stage3.start", {"doc_count": len(chunks_by_doc)}
            )
            chunks_by_doc = self._resolve_coreferences(
                chunks_by_doc, coref_model=coref_model
            )
            self._tracker.log_event("preprocessor", "stage3.complete", {})

        chunks = [c for doc_chunks in chunks_by_doc.values() for c in doc_chunks]

        self._tracker.log_metrics(
            {
                "num_documents": float(len(doc_records)),
                "num_chunks": float(len(chunks)),
            }
        )

        return chunks

    def _resolve_coreferences(
        self,
        chunks_by_doc: Dict[str, List[Chunk]],
        coref_model: str = "FCoref",
    ) -> Dict[str, List[Chunk]]:
        for doc_id, chunks in chunks_by_doc.items():
            if not chunks:
                continue
            # Reconstruct full document text from chunks in order
            doc_text = _reconstruct_doc_text(chunks)
            chunks_by_doc[doc_id] = resolve_coreferences_for_document(
                doc_text=doc_text,
                chunks=chunks,
                blacklist=self._blacklist,
                model_arch=coref_model,
            )
        return chunks_by_doc

    def close(self) -> None:
        self._catalog.close()

    def __enter__(self) -> "SpindlePreprocessor":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _get(cfg: Any, key: str, default: Any) -> Any:
    """Safely read a config key from a Hydra DictConfig or plain dict."""
    if cfg is None:
        return default
    if hasattr(cfg, key):
        return getattr(cfg, key)
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    try:
        return cfg[key]
    except (KeyError, TypeError):
        return default


def _reconstruct_doc_text(chunks: List[Chunk]) -> str:
    """Reconstruct full document text from an ordered list of chunks.

    Uses start_index metadata when available for accurate reconstruction.
    Falls back to joining chunk texts with newlines.
    """
    if not chunks:
        return ""
    if chunks[0].start_index is not None:
        # Determine total document length
        last = max(chunks, key=lambda c: c.end_index or 0)
        end = last.end_index or (last.start_index + len(last.text))
        buf = [" "] * end
        for chunk in chunks:
            start = chunk.start_index
            if start is None:
                continue
            for i, ch in enumerate(chunk.text):
                if start + i < len(buf):
                    buf[start + i] = ch
        return "".join(buf)
    return "\n\n".join(c.text for c in chunks)
