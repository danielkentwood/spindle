"""
Public utility functions for extraction operations.

This module provides factory functions, serialization utilities, and query/filter
functions for working with ontologies, triples, and extraction results.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from spindle.baml_client.types import (
    AttributeDefinition,
    AttributeValue,
    CharacterSpan,
    Entity,
    EntityType,
    Ontology,
    OntologyExtension,
    OntologyRecommendation,
    RelationType,
    SourceMetadata,
    Triple,
)


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

