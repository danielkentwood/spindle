"""
Entity resolution support utilities for graph stores.

This module provides backend-agnostic utilities for entity resolution operations.
"""

from typing import Dict, List, Optional, Set, Tuple

from spindle.graph_store.base import GraphStoreBackend


def get_duplicate_clusters(backend: GraphStoreBackend) -> List[List[str]]:
    """
    Find connected components of entities via SAME_AS edges.
    
    This identifies groups of entities that have been marked as duplicates
    through entity resolution.
    
    Args:
        backend: GraphStoreBackend instance to query
    
    Returns:
        List of clusters, where each cluster is a list of entity names
        that are duplicates of each other
    """
    # Query all SAME_AS edges
    same_as_edges = backend.query_by_pattern(predicate="SAME_AS")
    
    if not same_as_edges:
        return []
    
    # Build adjacency map
    adjacency: Dict[str, Set[str]] = {}
    all_nodes: Set[str] = set()
    
    for edge in same_as_edges:
        subject = edge['subject']
        obj = edge['object']
        all_nodes.add(subject)
        all_nodes.add(obj)
        
        if subject not in adjacency:
            adjacency[subject] = set()
        if obj not in adjacency:
            adjacency[obj] = set()
        
        adjacency[subject].add(obj)
        adjacency[obj].add(subject)
    
    # Find connected components using DFS
    visited: Set[str] = set()
    clusters: List[List[str]] = []
    
    def dfs(node: str, component: List[str]) -> None:
        if node in visited:
            return
        visited.add(node)
        component.append(node)
        
        for neighbor in adjacency.get(node, set()):
            if neighbor not in visited:
                dfs(neighbor, component)
    
    for node in all_nodes:
        if node not in visited:
            component: List[str] = []
            dfs(node, component)
            if component:
                clusters.append(component)
    
    return clusters


def get_canonical_entity(backend: GraphStoreBackend, name: str) -> Optional[str]:
    """
    Get the canonical (primary) entity name for a given entity.
    
    Follows SAME_AS edges to find the canonical representative of a
    duplicate cluster. Uses alphabetical ordering to consistently pick
    the same canonical entity within a cluster.
    
    Args:
        backend: GraphStoreBackend instance to query
        name: Entity name to resolve
    
    Returns:
        Canonical entity name, or None if entity doesn't exist
    """
    # Check if entity exists
    if backend.get_node(name) is None:
        return None
    
    # Get all SAME_AS edges from this entity
    same_as_edges = backend.query_by_pattern(subject=name, predicate="SAME_AS")
    
    if not same_as_edges:
        # No duplicates found, entity is its own canonical
        return name
    
    # Collect all entities in the duplicate cluster
    cluster = {name}
    for edge in same_as_edges:
        cluster.add(edge['object'])
    
    # Also check reverse direction
    reverse_edges = backend.query_by_pattern(predicate="SAME_AS", obj=name)
    for edge in reverse_edges:
        cluster.add(edge['subject'])
    
    # Return alphabetically first entity as canonical
    return sorted(cluster)[0]


def query_with_resolution(
    backend: GraphStoreBackend,
    subject: Optional[str] = None,
    predicate: Optional[str] = None,
    obj: Optional[str] = None,
    resolve_duplicates: bool = True
) -> List[Dict[str, any]]:
    """
    Query edges by pattern with optional duplicate resolution.
    
    When resolve_duplicates is True, entities are resolved to their
    canonical forms before querying, and results are deduplicated.
    
    Args:
        backend: GraphStoreBackend instance to query
        subject: Optional subject name (None = wildcard)
        predicate: Optional predicate name (None = wildcard)
        obj: Optional object name (None = wildcard)
        resolve_duplicates: Whether to resolve entities to canonical forms
    
    Returns:
        List of matching edge dictionaries
    """
    if not resolve_duplicates:
        return backend.query_by_pattern(subject, predicate, obj)
    
    # Resolve subject and object to canonical forms
    resolved_subject = None
    resolved_obj = None
    
    if subject is not None:
        canonical = get_canonical_entity(backend, subject)
        if canonical:
            resolved_subject = canonical
    
    if obj is not None:
        canonical = get_canonical_entity(backend, obj)
        if canonical:
            resolved_obj = canonical
    
    # Query with resolved entities
    results = backend.query_by_pattern(resolved_subject, predicate, resolved_obj)
    
    # Deduplicate results by creating a set of unique edge signatures
    seen_signatures: Set[Tuple[str, str, str]] = set()
    deduplicated = []
    
    for edge in results:
        # Get canonical forms for this edge's entities
        edge_subj_canonical = get_canonical_entity(backend, edge['subject']) or edge['subject']
        edge_obj_canonical = get_canonical_entity(backend, edge['object']) or edge['object']
        
        signature = (edge_subj_canonical, edge['predicate'], edge_obj_canonical)
        
        if signature not in seen_signatures:
            seen_signatures.add(signature)
            # Update edge with canonical entities
            edge['subject'] = edge_subj_canonical
            edge['object'] = edge_obj_canonical
            deduplicated.append(edge)
    
    return deduplicated

