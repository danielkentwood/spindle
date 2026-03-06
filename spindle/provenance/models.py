"""Pydantic/dataclass models for the provenance store."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class EvidenceSpan:
    """A text span within a source document that supports a provenance claim."""

    text: str
    start_offset: Optional[int] = None
    end_offset: Optional[int] = None
    section_path: Optional[List[str]] = None  # e.g. ["Introduction", "Background"]

    def section_path_json(self) -> Optional[str]:
        if self.section_path is None:
            return None
        return json.dumps(self.section_path)

    @staticmethod
    def parse_section_path(value: Optional[str]) -> Optional[List[str]]:
        if value is None:
            return None
        return json.loads(value)


@dataclass
class ProvenanceDoc:
    """Links a provenance object to one source document, with evidence spans."""

    doc_id: str
    spans: List[EvidenceSpan] = field(default_factory=list)
    # populated after insert
    id: Optional[int] = None


@dataclass
class ProvenanceObject:
    """Top-level provenance record for a single graph element."""

    object_id: str
    object_type: str  # 'kg_edge' | 'owl_entity' | 'vocab_entry'
    docs: List[ProvenanceDoc] = field(default_factory=list)
