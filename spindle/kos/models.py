"""Data models for the KOS service."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ConceptRecord:
    """In-memory representation of a SKOS concept."""

    uri: str
    concept_id: str  # dct:identifier value
    pref_label: str
    alt_labels: List[str] = field(default_factory=list)
    hidden_labels: List[str] = field(default_factory=list)
    definition: Optional[str] = None
    scope_note: Optional[str] = None
    broader: List[str] = field(default_factory=list)   # list of URIs
    narrower: List[str] = field(default_factory=list)  # list of URIs
    related: List[str] = field(default_factory=list)   # list of URIs
    provenance_object_id: Optional[str] = None


@dataclass
class EntityMention:
    """A vocabulary term found in text by Aho-Corasick or another NER method."""

    text: str
    start: int
    end: int
    concept_uri: str
    matched_label: str
    pref_label: str
    confidence: float = 1.0
    method: str = "aho_corasick"


@dataclass
class ResolutionResult:
    """Result of resolving a single entity mention against the KOS."""

    mention: str
    resolved: bool
    method: str  # 'exact_label' | 'semantic_search' | 'unresolved'
    concept_uri: Optional[str] = None
    pref_label: Optional[str] = None
    score: float = 0.0
    candidates: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ValidationReport:
    """SHACL validation report."""

    conforms: bool
    violations: List[Dict[str, Any]] = field(default_factory=list)
    message: Optional[str] = None
