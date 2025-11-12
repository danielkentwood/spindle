"""Utility functions for entity resolution."""

from typing import Any, Dict, List, Optional, Set, Tuple
import json

import numpy as np
import baml_py

from spindle.observability import get_event_recorder

ENTITY_RESOLUTION_RECORDER = get_event_recorder("entity_resolution")


def _record_resolution_event(name: str, payload: Dict[str, Any]) -> None:
    """Record an entity resolution event."""
    ENTITY_RESOLUTION_RECORDER.record(name=name, payload=payload)


def _extract_model_from_collector(collector: baml_py.baml_py.Collector) -> Optional[str]:
    """Extract model/client name from BAML collector logs.
    
    Returns the model identifier if available, otherwise None.
    """
    if not hasattr(collector, 'logs') or not collector.logs:
        return None
    
    # Try to get model from logs
    for log in collector.logs:
        # Check for common model/client attributes
        if hasattr(log, 'client_name'):
            return getattr(log, 'client_name')
        if hasattr(log, 'model'):
            return getattr(log, 'model')
        if hasattr(log, 'client'):
            client = getattr(log, 'client')
            if isinstance(client, str):
                return client
            if hasattr(client, 'name'):
                return getattr(client, 'name')
        # Check for model in metadata or options
        if hasattr(log, 'metadata') and isinstance(log.metadata, dict):
            model = log.metadata.get('model') or log.metadata.get('client_name')
            if model:
                return model
        if hasattr(log, 'options') and isinstance(log.options, dict):
            model = log.options.get('model') or log.options.get('client_name')
            if model:
                return model
    
    return None


def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.
    
    Args:
        vec1: First embedding vector
        vec2: Second embedding vector
    
    Returns:
        Cosine similarity score (0-1)
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def find_connected_components(edges: List[Tuple[str, str]]) -> List[Set[str]]:
    """Find connected components in a graph defined by edges.
    
    Uses union-find algorithm to efficiently group connected nodes.
    
    Args:
        edges: List of (node1, node2) tuples representing connections
    
    Returns:
        List of sets, each containing node IDs in a connected component
    """
    if not edges:
        return []
    
    # Build adjacency map
    adjacency: Dict[str, Set[str]] = {}
    all_nodes: Set[str] = set()
    
    for node1, node2 in edges:
        all_nodes.add(node1)
        all_nodes.add(node2)
        
        if node1 not in adjacency:
            adjacency[node1] = set()
        if node2 not in adjacency:
            adjacency[node2] = set()
        
        adjacency[node1].add(node2)
        adjacency[node2].add(node1)
    
    # Find connected components using DFS
    visited: Set[str] = set()
    components: List[Set[str]] = []
    
    def dfs(node: str, component: Set[str]) -> None:
        """Depth-first search to explore component."""
        if node in visited:
            return
        visited.add(node)
        component.add(node)
        
        for neighbor in adjacency.get(node, set()):
            if neighbor not in visited:
                dfs(neighbor, component)
    
    # Explore all nodes
    for node in all_nodes:
        if node not in visited:
            component: Set[str] = set()
            dfs(node, component)
            if component:
                components.append(component)
    
    return components


def serialize_node_for_embedding(node: Dict[str, Any]) -> str:
    """Convert a node to text representation for embedding.
    
    Args:
        node: Node dictionary with 'name', 'type', 'description', 'custom_atts'
    
    Returns:
        Text representation suitable for embedding
    """
    parts = []
    
    # Add name (always present)
    parts.append(f"Name: {node['name']}")
    
    # Add type if present
    if node.get('type'):
        parts.append(f"Type: {node['type']}")
    
    # Add description if present
    if node.get('description'):
        parts.append(f"Description: {node['description']}")
    
    # Add custom attributes if present
    if node.get('custom_atts'):
        custom_atts = node['custom_atts']
        if isinstance(custom_atts, str):
            try:
                custom_atts = json.loads(custom_atts)
            except json.JSONDecodeError:
                pass
        
        if isinstance(custom_atts, dict) and custom_atts:
            attrs_str = ", ".join([f"{k}: {v}" for k, v in custom_atts.items()])
            parts.append(f"Attributes: {attrs_str}")
    
    return " | ".join(parts)


def serialize_edge_for_embedding(edge: Dict[str, Any]) -> str:
    """Convert an edge to text representation for embedding.
    
    Args:
        edge: Edge dictionary with 'subject', 'predicate', 'object', 'supporting_evidence'
    
    Returns:
        Text representation suitable for embedding
    """
    parts = []
    
    # Core triple
    parts.append(f"{edge['subject']} -{edge['predicate']}-> {edge['object']}")
    
    # Add evidence summary if available
    if edge.get('supporting_evidence'):
        evidence = edge['supporting_evidence']
        if isinstance(evidence, str):
            try:
                evidence = json.loads(evidence)
            except json.JSONDecodeError:
                pass
        
        if isinstance(evidence, list) and evidence:
            # Count sources
            source_count = len(evidence)
            parts.append(f"Sources: {source_count}")
            
            # Sample evidence text from first source
            first_source = evidence[0]
            if isinstance(first_source, dict) and first_source.get('spans'):
                spans = first_source['spans']
                if spans and isinstance(spans, list):
                    first_span = spans[0]
                    if isinstance(first_span, dict) and first_span.get('text'):
                        sample_text = first_span['text'][:100]  # Limit length
                        parts.append(f"Evidence: {sample_text}...")
    
    return " | ".join(parts)


def merge_node_metadata(nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge metadata from multiple duplicate nodes.
    
    Consolidates sources, timestamps, and other metadata while preserving provenance.
    
    Args:
        nodes: List of node dictionaries to merge
    
    Returns:
        Merged metadata dictionary
    """
    if not nodes:
        return {}
    
    merged = {
        "merged_from": [node['name'] for node in nodes],
        "merge_count": len(nodes),
        "sources": [],
        "first_seen": None,
        "last_seen": None,
        "original_metadata": []
    }
    
    for node in nodes:
        metadata = node.get('metadata', {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}
        
        # Collect sources
        if isinstance(metadata, dict):
            if 'sources' in metadata:
                sources = metadata['sources']
                if isinstance(sources, list):
                    merged['sources'].extend(sources)
            
            # Track timestamps
            if 'first_seen' in metadata:
                if merged['first_seen'] is None or metadata['first_seen'] < merged['first_seen']:
                    merged['first_seen'] = metadata['first_seen']
            
            # Store original metadata
            merged['original_metadata'].append({
                "node_name": node['name'],
                "metadata": metadata
            })
    
    # Deduplicate sources
    merged['sources'] = list(set(merged['sources']))
    
    return merged


def merge_edge_metadata(edges: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge metadata from multiple duplicate edges.
    
    Combines supporting evidence from all duplicate edges.
    
    Args:
        edges: List of edge dictionaries to merge
    
    Returns:
        Merged metadata dictionary with consolidated evidence
    """
    if not edges:
        return {}
    
    merged = {
        "merged_from": [f"{e['subject']}|{e['predicate']}|{e['object']}" for e in edges],
        "merge_count": len(edges),
        "combined_evidence": []
    }
    
    # Collect all supporting evidence
    for edge in edges:
        evidence = edge.get('supporting_evidence', [])
        if isinstance(evidence, str):
            try:
                evidence = json.loads(evidence)
            except json.JSONDecodeError:
                evidence = []
        
        if isinstance(evidence, list):
            merged['combined_evidence'].extend(evidence)
    
    return merged

