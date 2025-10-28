"""
Spindle: Knowledge Graph Triple Extraction Tool

This module provides a simple interface for extracting knowledge graph triples
from text using BAML and LLMs. It supports ontology-based extraction with
entity consistency across multiple extraction runs. Each triple includes
source metadata, supporting text spans for provenance and evidence tracking,
and an extraction datetime for temporal tracking.
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from baml_client import b
from baml_client.types import (
    Triple,
    EntityType,
    RelationType,
    Ontology,
    ExtractionResult,
    SourceMetadata,
    CharacterSpan
)


def _find_span_indices(source_text: str, span_text: str) -> Optional[Tuple[int, int]]:
    """
    Find the start and end indices of span_text within source_text.
    
    Uses multiple strategies in order:
    1. Exact substring match
    2. Whitespace-normalized fuzzy match (handles newlines, extra spaces)
    3. Case-insensitive match
    
    Note: The LLM often provides clean span text (without extra whitespace/newlines),
    while the source may contain formatting. The returned indices point to the
    location in the original source, which may have different whitespace but the
    same content when normalized.
    
    Args:
        source_text: The full source text
        span_text: The text span to find
    
    Returns:
        Tuple of (start, end) indices, or None if not found
    """
    import re
    
    # Strategy 1: Try exact match first (fast path)
    start = source_text.find(span_text)
    if start != -1:
        return (start, start + len(span_text))
    
    # Strategy 2: Fuzzy match with whitespace normalization
    # The LLM often provides clean text without newlines/extra spaces,
    # but the source may have them. Create a pattern that matches
    # any amount of whitespace where the span has whitespace.
    
    # Split span into words and create pattern
    words = span_text.split()
    if len(words) > 0:
        # Escape each word and join with flexible whitespace
        pattern_parts = [re.escape(word) for word in words]
        pattern = r'\s+'.join(pattern_parts)
        
        try:
            match = re.search(pattern, source_text, re.DOTALL)
            if match:
                return (match.start(), match.end())
        except re.error:
            pass  # If regex fails, continue to next strategy
    
    # Strategy 3: Try normalizing both texts and finding position
    # Normalize: collapse all whitespace to single spaces
    def normalize_ws(text: str) -> str:
        return re.sub(r'\s+', ' ', text.strip())
    
    norm_span = normalize_ws(span_text)
    norm_source = normalize_ws(source_text)
    
    # Find in normalized version
    norm_pos = norm_source.find(norm_span)
    if norm_pos != -1:
        # Map normalized position back to original
        # This is complex, so use the word-based pattern instead
        words = span_text.split()
        if len(words) > 0:
            pattern_parts = [re.escape(word) for word in words]
            pattern = r'\s+'.join(pattern_parts)
            try:
                match = re.search(pattern, source_text, re.DOTALL | re.IGNORECASE)
                if match:
                    return (match.start(), match.end())
            except re.error:
                pass
    
    # Strategy 4: Case-insensitive exact match
    lower_source = source_text.lower()
    lower_span = span_text.lower()
    start = lower_source.find(lower_span)
    if start != -1:
        return (start, start + len(span_text))
    
    # Could not find the span
    return None


class SpindleExtractor:
    """
    Main interface for extracting knowledge graph triples from text.
    
    This class wraps the BAML extraction function and provides a simple
    interface for incremental triple extraction with entity consistency.
    Each extracted triple includes source metadata, supporting text spans,
    and an extraction datetime (set automatically in post-processing).
    """
    
    def __init__(self, ontology: Ontology):
        """
        Initialize the extractor with an ontology.
        
        Args:
            ontology: An Ontology object defining valid entity and relation types
        """
        self.ontology = ontology
    
    def extract(
        self,
        text: str,
        source_name: str,
        source_url: Optional[str] = None,
        existing_triples: List[Triple] = None
    ) -> ExtractionResult:
        """
        Extract triples from text using the configured ontology.
        
        Args:
            text: The text to extract triples from
            source_name: Name or identifier of the source document
            source_url: Optional URL of the source document
            existing_triples: Optional list of previously extracted triples
                            to maintain entity consistency. Duplicate triples
                            from different sources are allowed.
        
        Returns:
            ExtractionResult containing the extracted triples (with source
            metadata, supporting spans, and extraction datetime) and reasoning
        """
        if existing_triples is None:
            existing_triples = []
        
        # Create source metadata
        source_metadata = SourceMetadata(
            source_name=source_name,
            source_url=source_url
        )
        
        # Call the BAML extraction function
        result = b.ExtractTriples(
            text=text,
            ontology=self.ontology,
            source_metadata=source_metadata,
            existing_triples=existing_triples
        )
        
        # Post-processing: Set extraction datetime for all triples
        extraction_time = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        for triple in result.triples:
            triple.extraction_datetime = extraction_time
            
            # Post-processing: Compute character indices for supporting spans
            for span in triple.supporting_spans:
                if span.start is None or span.end is None:
                    # Find the span text in the source text
                    indices = _find_span_indices(text, span.text)
                    if indices:
                        span.start, span.end = indices
                    else:
                        # If exact match not found, set to -1 to indicate failure
                        span.start = -1
                        span.end = -1
        
        return result


def create_ontology(
    entity_types: List[Dict[str, str]],
    relation_types: List[Dict[str, str]]
) -> Ontology:
    """
    Factory function to create an Ontology object from dictionaries.
    
    Args:
        entity_types: List of dicts with 'name' and 'description' keys
        relation_types: List of dicts with 'name', 'description', 'domain',
                       and 'range' keys
    
    Returns:
        Ontology object
    
    Example:
        >>> entity_types = [
        ...     {"name": "Person", "description": "A human being"},
        ...     {"name": "Organization", "description": "A company or institution"}
        ... ]
        >>> relation_types = [
        ...     {
        ...         "name": "works_at",
        ...         "description": "Employment relationship",
        ...         "domain": "Person",
        ...         "range": "Organization"
        ...     }
        ... ]
        >>> ontology = create_ontology(entity_types, relation_types)
    """
    entity_objs = [
        EntityType(name=et["name"], description=et["description"])
        for et in entity_types
    ]
    
    relation_objs = [
        RelationType(
            name=rt["name"],
            description=rt["description"],
            domain=rt["domain"],
            range=rt["range"]
        )
        for rt in relation_types
    ]
    
    return Ontology(entity_types=entity_objs, relation_types=relation_objs)


def create_source_metadata(
    source_name: str,
    source_url: Optional[str] = None
) -> SourceMetadata:
    """
    Create a SourceMetadata object.
    
    Args:
        source_name: Name or identifier of the source
        source_url: Optional URL of the source
    
    Returns:
        SourceMetadata object
    """
    return SourceMetadata(source_name=source_name, source_url=source_url)


def triples_to_dict(triples: List[Triple]) -> List[Dict[str, Any]]:
    """
    Convert Triple objects to dictionaries for serialization.
    
    Args:
        triples: List of Triple objects
    
    Returns:
        List of dictionaries with all triple fields including metadata
    """
    return [
        {
            "subject": triple.subject,
            "predicate": triple.predicate,
            "object": triple.object,
            "source": {
                "source_name": triple.source.source_name,
                "source_url": triple.source.source_url
            },
            "supporting_spans": [
                {
                    "text": span.text,
                    "start": span.start if span.start is not None else -1,
                    "end": span.end if span.end is not None else -1
                }
                for span in triple.supporting_spans
            ],
            "extraction_datetime": triple.extraction_datetime if triple.extraction_datetime else ""
        }
        for triple in triples
    ]


def dict_to_triples(dicts: List[Dict[str, Any]]) -> List[Triple]:
    """
    Convert dictionaries back to Triple objects.
    
    Args:
        dicts: List of dictionaries with triple fields including metadata
    
    Returns:
        List of Triple objects
    """
    return [
        Triple(
            subject=d["subject"],
            predicate=d["predicate"],
            object=d["object"],
            source=SourceMetadata(
                source_name=d["source"]["source_name"],
                source_url=d["source"].get("source_url")
            ),
            supporting_spans=[
                CharacterSpan(
                    text=span["text"],
                    start=span.get("start") if span.get("start", -1) >= 0 else None,
                    end=span.get("end") if span.get("end", -1) >= 0 else None
                )
                for span in d.get("supporting_spans", [])
            ],
            extraction_datetime=d.get("extraction_datetime", "")
        )
        for d in dicts
    ]


def get_supporting_text(triple: Triple) -> List[str]:
    """
    Extract the supporting text snippets from a triple's character spans.
    
    Args:
        triple: A Triple object with supporting_spans
    
    Returns:
        List of text strings from the supporting spans
    """
    return [span.text for span in triple.supporting_spans]


def filter_triples_by_source(
    triples: List[Triple],
    source_name: str
) -> List[Triple]:
    """
    Filter triples to only those from a specific source.
    
    Args:
        triples: List of Triple objects
        source_name: Name of the source to filter by
    
    Returns:
        List of triples from the specified source
    """
    return [t for t in triples if t.source.source_name == source_name]


def parse_extraction_datetime(triple: Triple) -> Optional[datetime]:
    """
    Parse the extraction datetime string into a datetime object.
    
    Args:
        triple: A Triple object with extraction_datetime
    
    Returns:
        datetime object, or None if parsing fails
    """
    try:
        # Handle various ISO 8601 formats
        dt_str = triple.extraction_datetime.strip()
        # Try parsing with common formats
        for fmt in [
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S.%f%z",
        ]:
            try:
                return datetime.strptime(dt_str, fmt)
            except ValueError:
                continue
        # Try ISO format parser
        return datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
    except (ValueError, AttributeError):
        return None


def filter_triples_by_date_range(
    triples: List[Triple],
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> List[Triple]:
    """
    Filter triples by extraction date range.
    
    Args:
        triples: List of Triple objects
        start_date: Optional start datetime (inclusive)
        end_date: Optional end datetime (inclusive)
    
    Returns:
        List of triples extracted within the date range
    """
    filtered = []
    for triple in triples:
        dt = parse_extraction_datetime(triple)
        if dt is None:
            continue
        if start_date and dt < start_date:
            continue
        if end_date and dt > end_date:
            continue
        filtered.append(triple)
    return filtered

