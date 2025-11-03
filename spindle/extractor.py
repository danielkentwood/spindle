"""
Spindle Extractor Module

This module provides the core triple extraction functionality for Spindle.

Key Components:
- SpindleExtractor: Extract triples from text using a predefined ontology
- OntologyRecommender: Automatically recommend ontologies by analyzing text,
  and conservatively extend existing ontologies when processing new sources
- Helper functions for ontology creation, serialization, and filtering
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from spindle.baml_client import b
from spindle.baml_client.types import (
    Triple,
    Entity,
    EntityType,
    AttributeDefinition,
    AttributeValue,
    RelationType,
    Ontology,
    ExtractionResult,
    OntologyRecommendation,
    OntologyExtension,
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
    
    If no ontology is provided at initialization, the extractor will
    automatically recommend one based on the text when extract() is called.
    """
    
    def __init__(
        self,
        ontology: Optional[Ontology] = None,
        ontology_scope: str = "balanced"
    ):
        """
        Initialize the extractor with an ontology.
        
        Args:
            ontology: Optional Ontology object defining valid entity and relation types.
                     If None, an ontology will be automatically recommended from the
                     text when extract() is first called.
            ontology_scope: Scope for auto-recommended ontologies. One of:
                          - "minimal": Essential concepts only (3-8 entity types, 4-10 relations)
                          - "balanced": Standard analysis (6-12 entity types, 8-15 relations)
                          - "comprehensive": Detailed ontology (10-20 entity types, 12-25 relations)
                          Only used if ontology is None.
        """
        self.ontology = ontology
        self.ontology_scope = ontology_scope
        self._ontology_recommender = None if ontology is not None else OntologyRecommender()
    
    def extract(
        self,
        text: str,
        source_name: str,
        source_url: Optional[str] = None,
        existing_triples: List[Triple] = None,
        ontology_scope: Optional[str] = None
    ) -> ExtractionResult:
        """
        Extract triples from text using the configured or auto-recommended ontology.
        
        If no ontology was provided at initialization, one will be automatically
        recommended based on the text before extraction.
        
        Args:
            text: The text to extract triples from
            source_name: Name or identifier of the source document
            source_url: Optional URL of the source document
            existing_triples: Optional list of previously extracted triples
                            to maintain entity consistency. Duplicate triples
                            from different sources are allowed.
            ontology_scope: Override the default ontology scope for this extraction.
                          One of "minimal", "balanced", or "comprehensive".
                          Only used if no ontology was provided at init.
        
        Returns:
            ExtractionResult containing the extracted triples (with source
            metadata, supporting spans, and extraction datetime) and reasoning
        """
        if existing_triples is None:
            existing_triples = []
        
        # Auto-recommend ontology if not provided
        if self.ontology is None:
            scope = ontology_scope or self.ontology_scope
            recommendation = self._ontology_recommender.recommend(
                text=text,
                scope=scope
            )
            self.ontology = recommendation.ontology
        
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
    entity_types: List[Dict[str, Any]],
    relation_types: List[Dict[str, str]]
) -> Ontology:
    """
    Factory function to create an Ontology object from dictionaries.
    
    Args:
        entity_types: List of dicts with 'name', 'description', and optional 'attributes' keys
                     Each attribute should have 'name', 'type', and 'description'
        relation_types: List of dicts with 'name', 'description', 'domain',
                       and 'range' keys
    
    Returns:
        Ontology object
    
    Example:
        >>> entity_types = [
        ...     {
        ...         "name": "Campaign",
        ...         "description": "A marketing campaign",
        ...         "attributes": [
        ...             {
        ...                 "name": "campaign_launch_dt",
        ...                 "type": "date",
        ...                 "description": "The date the campaign launched"
        ...             },
        ...             {
        ...                 "name": "campaign_completion_dt",
        ...                 "type": "date",
        ...                 "description": "The date the campaign completed"
        ...             }
        ...         ]
        ...     },
        ...     {"name": "Person", "description": "A human being"}
        ... ]
        >>> relation_types = [
        ...     {
        ...         "name": "manages",
        ...         "description": "Management relationship",
        ...         "domain": "Person",
        ...         "range": "Campaign"
        ...     }
        ... ]
        >>> ontology = create_ontology(entity_types, relation_types)
    """
    entity_objs = []
    for et in entity_types:
        # Handle attributes if present
        attributes = []
        if "attributes" in et and et["attributes"]:
            attributes = [
                AttributeDefinition(
                    name=attr["name"],
                    type=attr["type"],
                    description=attr["description"]
                )
                for attr in et["attributes"]
            ]
        
        entity_objs.append(
            EntityType(
                name=et["name"],
                description=et["description"],
                attributes=attributes
            )
        )
    
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
        triples: List of Triple objects with Entity subjects and objects
    
    Returns:
        List of dictionaries with all triple fields including structured entities
        
    Note:
        Entities are serialized with name, type, description, and custom_atts.
        Custom attributes include type metadata: {"value": "...", "type": "..."}
    """
    return [
        {
            "subject": {
                "name": triple.subject.name,
                "type": triple.subject.type,
                "description": triple.subject.description,
                "custom_atts": {
                    attr_name: {
                        "value": attr_val.value,
                        "type": attr_val.type
                    }
                    for attr_name, attr_val in triple.subject.custom_atts.items()
                }
            },
            "predicate": triple.predicate,
            "object": {
                "name": triple.object.name,
                "type": triple.object.type,
                "description": triple.object.description,
                "custom_atts": {
                    attr_name: {
                        "value": attr_val.value,
                        "type": attr_val.type
                    }
                    for attr_name, attr_val in triple.object.custom_atts.items()
                }
            },
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
        dicts: List of dictionaries with triple fields including structured entities
    
    Returns:
        List of Triple objects with Entity subjects and objects
        
    Note:
        Handles both old format (string entities) and new format (Entity objects)
        for backward compatibility during migration.
    """
    triples = []
    for d in dicts:
        # Handle subject - support both old string format and new Entity format
        if isinstance(d["subject"], str):
            # Old format: convert string to minimal Entity
            subject = Entity(
                name=d["subject"],
                type="Unknown",
                description="",
                custom_atts={}
            )
        else:
            # New format: reconstruct Entity from dict
            subject = Entity(
                name=d["subject"]["name"],
                type=d["subject"]["type"],
                description=d["subject"]["description"],
                custom_atts={
                    attr_name: AttributeValue(
                        value=attr_val["value"],
                        type=attr_val["type"]
                    )
                    for attr_name, attr_val in d["subject"].get("custom_atts", {}).items()
                }
            )
        
        # Handle object - support both old string format and new Entity format
        if isinstance(d["object"], str):
            # Old format: convert string to minimal Entity
            obj = Entity(
                name=d["object"],
                type="Unknown",
                description="",
                custom_atts={}
            )
        else:
            # New format: reconstruct Entity from dict
            obj = Entity(
                name=d["object"]["name"],
                type=d["object"]["type"],
                description=d["object"]["description"],
                custom_atts={
                    attr_name: AttributeValue(
                        value=attr_val["value"],
                        type=attr_val["type"]
                    )
                    for attr_name, attr_val in d["object"].get("custom_atts", {}).items()
                }
            )
        
        triples.append(
            Triple(
                subject=subject,
                predicate=d["predicate"],
                object=obj,
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
        )
    
    return triples


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


class OntologyRecommender:
    """
    Service for recommending ontologies based on text analysis.
    
    This class wraps the BAML ontology recommendation function and provides
    a simple interface for automatically generating ontologies suitable for
    knowledge graph extraction from text.
    
    Uses principle-based ontology design with configurable scope levels
    instead of hard numerical limits.
    """
    
    def recommend(
        self,
        text: str,
        scope: str = "balanced"
    ) -> OntologyRecommendation:
        """
        Analyze text and recommend an appropriate ontology.
        
        This method examines the provided text to infer its overarching
        purpose/goal and domain, then recommends entity types and relation
        types that would be most suitable for extracting knowledge from
        this and similar texts.
        
        The ontology is designed using principled guidelines about granularity
        and abstraction level, rather than hard numerical limits. The LLM
        determines the appropriate number of types based on the text's
        complexity and domain.
        
        Args:
            text: The text to analyze for ontology recommendation
            scope: Desired ontology scope, one of:
                  - "minimal": Essential concepts only (3-8 entity types, 4-10 relations)
                    Use for quick extraction, simple queries, broad patterns
                  - "balanced": Standard analysis (6-12 entity types, 8-15 relations)
                    Use for general-purpose extraction (DEFAULT)
                  - "comprehensive": Detailed ontology (10-20 entity types, 12-25 relations)
                    Use for detailed analysis, domain expertise, research
        
        Returns:
            OntologyRecommendation containing:
                - ontology: The recommended Ontology object (ready for use
                           with SpindleExtractor)
                - text_purpose: Analysis of the text's overarching purpose
                - reasoning: Explanation of the recommendation choices
        
        Example:
            >>> recommender = OntologyRecommender()
            >>> text = "Alice works at TechCorp in San Francisco..."
            >>> 
            >>> # Get a balanced ontology (default)
            >>> recommendation = recommender.recommend(text)
            >>> 
            >>> # Or request a specific scope
            >>> minimal_rec = recommender.recommend(text, scope="minimal")
            >>> comprehensive_rec = recommender.recommend(text, scope="comprehensive")
            >>> 
            >>> print(recommendation.text_purpose)
            >>> extractor = SpindleExtractor(recommendation.ontology)
        """
        result = b.RecommendOntology(
            text=text,
            scope=scope
        )
        
        return result
    
    def recommend_and_extract(
        self,
        text: str,
        source_name: str,
        source_url: Optional[str] = None,
        scope: str = "balanced",
        existing_triples: List[Triple] = None
    ) -> Tuple[OntologyRecommendation, ExtractionResult]:
        """
        Recommend an ontology and immediately use it to extract triples.
        
        This is a convenience method that combines ontology recommendation
        and triple extraction in one step. Useful when you don't have a
        pre-defined ontology and want to let the system automatically
        determine the best structure for your text.
        
        Args:
            text: The text to analyze and extract from
            source_name: Name or identifier of the source document
            source_url: Optional URL of the source document
            scope: Desired ontology scope ("minimal", "balanced", or "comprehensive")
            existing_triples: Optional list of previously extracted triples
                             for entity consistency
        
        Returns:
            Tuple of (OntologyRecommendation, ExtractionResult):
                - OntologyRecommendation with the generated ontology
                - ExtractionResult with extracted triples using that ontology
        
        Example:
            >>> recommender = OntologyRecommender()
            >>> text = "Alice works at TechCorp..."
            >>> 
            >>> # Use default balanced scope
            >>> recommendation, extraction = recommender.recommend_and_extract(
            ...     text=text,
            ...     source_name="example.txt"
            ... )
            >>> 
            >>> # Or request comprehensive scope
            >>> recommendation, extraction = recommender.recommend_and_extract(
            ...     text=complex_paper,
            ...     source_name="research.pdf",
            ...     scope="comprehensive"
            ... )
            >>> 
            >>> print(f"Purpose: {recommendation.text_purpose}")
            >>> print(f"Extracted {len(extraction.triples)} triples")
        """
        # First, recommend the ontology
        recommendation = self.recommend(
            text=text,
            scope=scope
        )
        
        # Then, use the recommended ontology to extract triples
        extractor = SpindleExtractor(recommendation.ontology)
        extraction_result = extractor.extract(
            text=text,
            source_name=source_name,
            source_url=source_url,
            existing_triples=existing_triples
        )
        
        return recommendation, extraction_result
    
    def analyze_extension(
        self,
        text: str,
        current_ontology: Ontology,
        scope: str = "balanced"
    ) -> OntologyExtension:
        """
        Analyze whether an existing ontology needs to be extended for new text.
        
        This method conservatively analyzes if the current ontology is sufficient
        to extract critical information from new text, or if it needs extension.
        
        CONSERVATIVE APPROACH: The analysis defaults to NO extension. Extensions
        are only recommended if failing to extend would result in losing critical
        information that cannot be captured with existing types.
        
        Args:
            text: The new text to analyze
            current_ontology: The existing Ontology to potentially extend
            scope: The scope level to consider ("minimal", "balanced", "comprehensive")
                  This affects how conservative the extension analysis is.
        
        Returns:
            OntologyExtension containing:
                - needs_extension: Boolean indicating if extension is needed
                - new_entity_types: List of new entity types to add (empty if none)
                - new_relation_types: List of new relation types to add (empty if none)
                - critical_information_at_risk: Description of what would be lost
                - reasoning: Detailed explanation of the decision
        
        Example:
            >>> recommender = OntologyRecommender()
            >>> # Existing ontology for business domain
            >>> ontology = create_ontology(business_entities, business_relations)
            >>> 
            >>> # Analyze new text from medical domain
            >>> medical_text = "Dr. Chen prescribed Medication A for the patient..."
            >>> extension = recommender.analyze_extension(
            ...     text=medical_text,
            ...     current_ontology=ontology,
            ...     scope="balanced"
            ... )
            >>> 
            >>> if extension.needs_extension:
            ...     print(f"Extension needed: {extension.critical_information_at_risk}")
            ...     print(f"New types: {[et.name for et in extension.new_entity_types]}")
            ... else:
            ...     print(f"No extension needed: {extension.reasoning}")
        """
        result = b.AnalyzeOntologyExtension(
            text=text,
            current_ontology=current_ontology,
            scope=scope
        )
        
        return result
    
    def extend_ontology(
        self,
        current_ontology: Ontology,
        extension: OntologyExtension
    ) -> Ontology:
        """
        Apply an ontology extension to create an extended ontology.
        
        This creates a new Ontology object by combining the current ontology
        with the new types from the extension analysis.
        
        Args:
            current_ontology: The existing Ontology
            extension: The OntologyExtension from analyze_extension()
        
        Returns:
            A new Ontology object with the extensions applied
        
        Example:
            >>> extension = recommender.analyze_extension(text, ontology)
            >>> if extension.needs_extension:
            ...     extended_ontology = recommender.extend_ontology(ontology, extension)
            ...     print(f"Original: {len(ontology.entity_types)} types")
            ...     print(f"Extended: {len(extended_ontology.entity_types)} types")
        """
        # Combine existing and new entity types
        all_entity_types = list(current_ontology.entity_types) + list(extension.new_entity_types)
        
        # Combine existing and new relation types
        all_relation_types = list(current_ontology.relation_types) + list(extension.new_relation_types)
        
        # Create and return the extended ontology
        return Ontology(
            entity_types=all_entity_types,
            relation_types=all_relation_types
        )
    
    def analyze_and_extend(
        self,
        text: str,
        current_ontology: Ontology,
        scope: str = "balanced",
        auto_apply: bool = True
    ) -> Tuple[OntologyExtension, Optional[Ontology]]:
        """
        Analyze extension needs and optionally apply them in one step.
        
        This is a convenience method that combines analyze_extension() and
        extend_ontology() with optional automatic application.
        
        Args:
            text: The new text to analyze
            current_ontology: The existing Ontology
            scope: The scope level for analysis
            auto_apply: If True, automatically apply extensions and return new ontology.
                       If False, return None as second element (user must apply manually).
        
        Returns:
            Tuple of (OntologyExtension, Optional[Ontology]):
                - OntologyExtension with the analysis results
                - Extended Ontology if auto_apply=True and needs_extension=True, else None
        
        Example:
            >>> # Automatic application
            >>> extension, new_ontology = recommender.analyze_and_extend(
            ...     text=new_text,
            ...     current_ontology=ontology,
            ...     auto_apply=True
            ... )
            >>> 
            >>> if new_ontology:
            ...     # Extension was needed and applied
            ...     extractor = SpindleExtractor(new_ontology)
            ... else:
            ...     # No extension needed, use original
            ...     extractor = SpindleExtractor(ontology)
        """
        extension = self.analyze_extension(text, current_ontology, scope)
        
        if extension.needs_extension and auto_apply:
            extended_ontology = self.extend_ontology(current_ontology, extension)
            return extension, extended_ontology
        
        return extension, None


def ontology_to_dict(ontology: Ontology) -> Dict[str, List[Dict[str, Any]]]:
    """
    Convert an Ontology object to a dictionary for serialization.
    
    Args:
        ontology: An Ontology object
    
    Returns:
        Dictionary with 'entity_types' and 'relation_types' keys,
        including attributes for entity types
    
    Example:
        >>> ontology = create_ontology(entity_types, relation_types)
        >>> ontology_dict = ontology_to_dict(ontology)
        >>> json.dumps(ontology_dict, indent=2)
    """
    return {
        "entity_types": [
            {
                "name": et.name,
                "description": et.description,
                "attributes": [
                    {
                        "name": attr.name,
                        "type": attr.type,
                        "description": attr.description
                    }
                    for attr in et.attributes
                ]
            }
            for et in ontology.entity_types
        ],
        "relation_types": [
            {
                "name": rt.name,
                "description": rt.description,
                "domain": rt.domain,
                "range": rt.range
            }
            for rt in ontology.relation_types
        ]
    }


def recommendation_to_dict(
    recommendation: OntologyRecommendation
) -> Dict[str, Any]:
    """
    Convert an OntologyRecommendation to a dictionary for serialization.
    
    Args:
        recommendation: An OntologyRecommendation object
    
    Returns:
        Dictionary with ontology, text_purpose, and reasoning
    """
    return {
        "ontology": ontology_to_dict(recommendation.ontology),
        "text_purpose": recommendation.text_purpose,
        "reasoning": recommendation.reasoning
    }


def extension_to_dict(
    extension: OntologyExtension
) -> Dict[str, Any]:
    """
    Convert an OntologyExtension to a dictionary for serialization.
    
    Args:
        extension: An OntologyExtension object
    
    Returns:
        Dictionary with extension analysis results including attributes
    """
    return {
        "needs_extension": extension.needs_extension,
        "new_entity_types": [
            {
                "name": et.name,
                "description": et.description,
                "attributes": [
                    {
                        "name": attr.name,
                        "type": attr.type,
                        "description": attr.description
                    }
                    for attr in et.attributes
                ]
            }
            for et in extension.new_entity_types
        ],
        "new_relation_types": [
            {
                "name": rt.name,
                "description": rt.description,
                "domain": rt.domain,
                "range": rt.range
            }
            for rt in extension.new_relation_types
        ],
        "critical_information_at_risk": extension.critical_information_at_risk,
        "reasoning": extension.reasoning
    }

