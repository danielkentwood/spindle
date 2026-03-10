"""Spindle KOS (Knowledge Organization System) package.

Provides KOSService for loading and querying SKOS/OWL/SHACL artifacts,
plus derived indices (Aho-Corasick, ANN, label map, concept cache).
"""

from spindle.kos.models import (
    ConceptRecord,
    EntityMention,
    ResolutionResult,
    ValidationReport,
)
from spindle.kos.service import KOSService

__all__ = [
    "KOSService",
    "ConceptRecord",
    "EntityMention",
    "ResolutionResult",
    "ValidationReport",
]
