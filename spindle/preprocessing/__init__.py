"""Spindle v2 preprocessing pipeline.

Three-stage pipeline: Docling ingestion → Chonkie chunking → fastcoref coref.

Replaces the LangChain-based ingestion pipeline as the v2 path.
The existing ``spindle/ingestion/`` path remains for backward compatibility.

Basic usage::

    from spindle.preprocessing import SpindlePreprocessor

    preprocessor = SpindlePreprocessor(
        documents=["path/to/doc.pdf"],
        catalog_path="catalog.db",
    )
    chunks = preprocessor()
"""

from spindle.preprocessing.models import Chunk, DocumentRecord
from spindle.preprocessing.offsets import (
    build_offset_map,
    reconstruct_coref_resolved_text,
    to_document_offset,
)
from spindle.preprocessing.preprocessor import SpindlePreprocessor

__all__ = [
    "Chunk",
    "DocumentRecord",
    "SpindlePreprocessor",
    "to_document_offset",
    "build_offset_map",
    "reconstruct_coref_resolved_text",
]
