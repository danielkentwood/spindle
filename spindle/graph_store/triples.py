"""
Triple integration utilities for graph stores.

This module provides utilities for converting between Spindle Triple objects
and graph store format.
"""

from typing import List

from spindle.baml_client.types import (
    Triple,
    Entity,
    SourceMetadata,
    CharacterSpan,
    AttributeValue
)


def triple_to_edge_metadata(triple: Triple) -> dict:
    """
    Convert a Triple object to edge metadata format.
    
    Args:
        triple: Triple object from Spindle extraction
    
    Returns:
        Dictionary with 'supporting_evidence' in nested format
    """
    extraction_datetime = triple.extraction_datetime or ""
    
    supporting_evidence = [{
        "source_nm": triple.source.source_name,
        "source_url": triple.source.source_url or "",
        "spans": [
            {
                "text": span.text,
                "start": span.start,
                "end": span.end,
                "extraction_datetime": extraction_datetime
            }
            for span in triple.supporting_spans
        ]
    }]
    
    return {
        "supporting_evidence": supporting_evidence
    }


def edge_to_triples(edge_data: dict) -> List[Triple]:
    """
    Convert edge data from graph store to Triple objects.
    
    Creates one Triple per source within each edge for backward compatibility.
    
    Args:
        edge_data: Dictionary with edge information including supporting_evidence
    
    Returns:
        List of Triple objects
    """
    import json
    
    triples = []
    
    # Parse subject entity
    subject_custom_atts = edge_data.get("subject_custom_atts", {})
    if isinstance(subject_custom_atts, str):
        subject_custom_atts = json.loads(subject_custom_atts) if subject_custom_atts else {}
    
    subject = Entity(
        name=edge_data["subject_name"],
        type=edge_data["subject_type"],
        description=edge_data.get("subject_description", "") or "",
        custom_atts={
            attr_name: AttributeValue(value=attr_data["value"], type=attr_data["type"])
            for attr_name, attr_data in subject_custom_atts.items()
        }
    )
    
    # Parse object entity
    object_custom_atts = edge_data.get("object_custom_atts", {})
    if isinstance(object_custom_atts, str):
        object_custom_atts = json.loads(object_custom_atts) if object_custom_atts else {}
    
    obj = Entity(
        name=edge_data["object_name"],
        type=edge_data["object_type"],
        description=edge_data.get("object_description", "") or "",
        custom_atts={
            attr_name: AttributeValue(value=attr_data["value"], type=attr_data["type"])
            for attr_name, attr_data in object_custom_atts.items()
        }
    )
    
    # Parse supporting evidence (new nested format)
    evidence_list = edge_data.get("supporting_evidence", [])
    if isinstance(evidence_list, str):
        evidence_list = json.loads(evidence_list) if evidence_list else []
    
    # Create one Triple per source for backward compatibility
    for evidence_source in evidence_list:
        source_nm = evidence_source.get("source_nm", "")
        source_url = evidence_source.get("source_url", "")
        spans_data = evidence_source.get("spans", [])
        
        # Convert spans and extract extraction_datetime from first span
        spans = []
        extraction_datetime = ""
        for span_data in spans_data:
            spans.append(CharacterSpan(
                text=span_data.get("text", ""),
                start=span_data.get("start"),
                end=span_data.get("end")
            ))
            # Use first span's datetime
            if not extraction_datetime:
                extraction_datetime = span_data.get("extraction_datetime", "")
        
        # Create Triple object
        triple = Triple(
            subject=subject,
            predicate=edge_data["predicate"],
            object=obj,
            source=SourceMetadata(
                source_name=source_nm,
                source_url=source_url
            ),
            supporting_spans=spans,
            extraction_datetime=extraction_datetime
        )
        triples.append(triple)
    
    return triples

