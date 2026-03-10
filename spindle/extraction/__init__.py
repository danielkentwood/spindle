"""
Spindle Extraction Package

This package provides the core triple extraction functionality for Spindle.

Key Components:
- SpindleExtractor: Extract triples from text using a predefined ontology
- Helper functions for ontology creation, serialization, and filtering
"""

from spindle.extraction.extractor import SpindleExtractor
from spindle.extraction.utils import (
    create_ontology,
    create_source_metadata,
    triples_to_dict,
    dict_to_triples,
    get_supporting_text,
    filter_triples_by_source,
    parse_extraction_datetime,
    filter_triples_by_date_range,
    ontology_to_dict,
)
from spindle.extraction.helpers import _find_span_indices, _compute_all_span_indices

__all__ = [
    # Main classes
    "SpindleExtractor",
    # Factory functions
    "create_ontology",
    "create_source_metadata",
    # Serialization functions
    "triples_to_dict",
    "dict_to_triples",
    "ontology_to_dict",
    # Query/filter functions
    "get_supporting_text",
    "filter_triples_by_source",
    "parse_extraction_datetime",
    "filter_triples_by_date_range",
    # Internal (for testing)
    "_find_span_indices",
    "_compute_all_span_indices",
]
