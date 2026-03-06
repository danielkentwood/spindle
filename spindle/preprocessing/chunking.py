"""Stage 2: Semantic chunking via Chonkie.

Splits document text into chunks using Chonkie's recursive chunker with
optional overlap refinement.  Produces Chunk objects carrying positional
metadata essential for provenance tracking.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from spindle.preprocessing.models import Chunk, DocumentRecord


def _load_document_text(record: DocumentRecord) -> str:
    """Read document text from the Docling JSON file."""
    json_path = Path(record.docling_json_path)
    if not json_path.exists():
        return ""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    # Docling JSON typically has a "text" key at the top level
    if "text" in data:
        return data["text"]
    # Fallback: concatenate all text items
    texts = data.get("texts", [])
    if texts:
        return "\n\n".join(item.get("text", "") for item in texts if item.get("text"))
    return json.dumps(data)


def chunk_document(
    record: DocumentRecord,
    chunk_size: int = 600,
    overlap: float = 0.10,
    strategy: str = "recursive",
) -> List[Chunk]:
    """Chunk a single document using Chonkie.

    Args:
        record: DocumentRecord containing the path to the Docling JSON.
        chunk_size: Target characters per chunk.
        overlap: Fractional overlap (0–1) between adjacent chunks.
        strategy: Chunking strategy.  Only ``"recursive"`` is currently
                  supported via Chonkie's ``RecursiveChunker``.

    Returns:
        List of Chunk objects with start_index/end_index in metadata.

    Raises:
        ImportError: If ``chonkie`` is not installed.
    """
    text = _load_document_text(record)
    if not text:
        return []

    try:
        return _chunk_with_chonkie(record, text, chunk_size, overlap, strategy)
    except ImportError:
        return _chunk_fallback(record, text, chunk_size, overlap)


def _chunk_with_chonkie(
    record: DocumentRecord,
    text: str,
    chunk_size: int,
    overlap: float,
    strategy: str,
) -> List[Chunk]:
    """Use Chonkie's Pipeline to chunk the document text."""
    try:
        from chonkie import RecursiveChunker
        from chonkie.refinery import OverlapRefinery
    except ImportError as exc:
        raise ImportError(
            "chonkie is required for semantic chunking. "
            "Install it with: uv pip install chonkie"
        ) from exc

    context_size = max(1, int(chunk_size * overlap))
    chunker = RecursiveChunker(
        tokenizer_or_token_counter="character",
        chunk_size=chunk_size,
    )
    raw_chunks = chunker.chunk(text)

    refinery = OverlapRefinery(context_size=context_size, method="suffix")
    refined_chunks = refinery.refine(raw_chunks)

    return [
        Chunk(
            text=c.text,
            source_id=record.doc_id,
            metadata={
                "chunk_index": i,
                "start_index": getattr(c, "start_index", None),
                "end_index": getattr(c, "end_index", None),
                "token_count": getattr(c, "token_count", len(c.text)),
                "doc_content_hash": record.content_hash,
                "doc_version": record.version,
            },
        )
        for i, c in enumerate(refined_chunks)
    ]


def _chunk_fallback(
    record: DocumentRecord,
    text: str,
    chunk_size: int,
    overlap: float,
) -> List[Chunk]:
    """Simple character-based chunker used when Chonkie is not installed."""
    step = max(1, int(chunk_size * (1 - overlap)))
    chunks: List[Chunk] = []
    pos = 0
    idx = 0
    while pos < len(text):
        end = min(pos + chunk_size, len(text))
        chunks.append(
            Chunk(
                text=text[pos:end],
                source_id=record.doc_id,
                metadata={
                    "chunk_index": idx,
                    "start_index": pos,
                    "end_index": end,
                    "token_count": end - pos,
                    "doc_content_hash": record.content_hash,
                    "doc_version": record.version,
                },
            )
        )
        pos += step
        idx += 1
    return chunks
