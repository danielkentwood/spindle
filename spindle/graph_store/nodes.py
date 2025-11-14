"""
Node operation utilities for graph stores.

This module provides backend-agnostic utilities for node operations.
"""

from typing import Dict, List, Optional, Tuple

from spindle.baml_client.types import Triple, AttributeValue


def extract_nodes_from_triple(
    triple: Triple,
    subject_vector_index: Optional[str] = None,
    object_vector_index: Optional[str] = None
) -> Tuple[Dict[str, any], Dict[str, any]]:
    """
    Extract subject and object node data from a triple.
    
    Args:
        triple: Triple object containing subject and object Entity objects
        subject_vector_index: Optional vector index UID for subject embedding
        object_vector_index: Optional vector index UID for object embedding
    
    Returns:
        Tuple of (subject_node_dict, object_node_dict)
    """
    # Extract subject entity information
    subject_metadata = {
        "sources": [triple.source.source_name],
        "first_seen": triple.extraction_datetime
    }
    
    # Convert AttributeValue objects to serializable dicts
    subject_custom_atts = {
        attr_name: {"value": attr_val.value, "type": attr_val.type}
        for attr_name, attr_val in triple.subject.custom_atts.items()
    }
    
    subject_node = {
        "name": triple.subject.name,
        "type": triple.subject.type,
        "description": triple.subject.description,
        "custom_atts": subject_custom_atts,
        "metadata": subject_metadata,
        "vector_index": subject_vector_index
    }
    
    # Extract object entity information
    object_metadata = {
        "sources": [triple.source.source_name],
        "first_seen": triple.extraction_datetime
    }
    
    # Convert AttributeValue objects to serializable dicts
    object_custom_atts = {
        attr_name: {"value": attr_val.value, "type": attr_val.type}
        for attr_name, attr_val in triple.object.custom_atts.items()
    }
    
    object_node = {
        "name": triple.object.name,
        "type": triple.object.type,
        "description": triple.object.description,
        "custom_atts": object_custom_atts,
        "metadata": object_metadata,
        "vector_index": object_vector_index
    }
    
    return (subject_node, object_node)

