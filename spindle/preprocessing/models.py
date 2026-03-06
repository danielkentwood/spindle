"""Data models for the v2 preprocessing pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class DocumentRecord:
    """Metadata record for a source document in the document catalog."""

    doc_id: str
    source_path: str
    content_hash: str
    docling_json_path: str
    last_ingested: datetime
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A text chunk produced by the preprocessing pipeline.

    Satisfies spindle-eval's Chunk protocol when spindle-eval is installed.
    The chunk text is never modified — coref annotations are stored in metadata.
    """

    text: str
    source_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Convenience accessors
    @property
    def start_index(self) -> Optional[int]:
        return self.metadata.get("start_index")

    @property
    def end_index(self) -> Optional[int]:
        return self.metadata.get("end_index")

    @property
    def chunk_index(self) -> Optional[int]:
        return self.metadata.get("chunk_index")

    @property
    def section_path(self) -> Optional[List[str]]:
        return self.metadata.get("section_path")

    @property
    def coref_annotations(self) -> List[Dict[str, Any]]:
        return self.metadata.get("coref_annotations", [])
