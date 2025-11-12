"""Graph merging operations for entity resolution.

Functions to create SAME_AS edges and identify duplicate clusters.
"""

from typing import TYPE_CHECKING, List, Set
from datetime import datetime

if TYPE_CHECKING:
    from spindle.graph_store import GraphStore

from spindle.entity_resolution.models import EntityMatch, EdgeMatch
from spindle.entity_resolution.utils import find_connected_components


def create_same_as_edges(
    graph_store: 'GraphStore',
    matches: List[EntityMatch]
) -> int:
    """Create SAME_AS edges in the graph store for duplicate entities.
    
    Args:
        graph_store: GraphStore to add edges to
        matches: List of EntityMatch objects for duplicates
    
    Returns:
        Number of SAME_AS edges created
    """
    created_count = 0
    
    for match in matches:
        if not match.is_duplicate:
            continue
        
        # Create SAME_AS edge with match metadata
        metadata = {
            "supporting_evidence": [{
                "source_nm": "entity_resolution",
                "source_url": "",
                "spans": [{
                    "text": match.reasoning,
                    "start": -1,
                    "end": -1,
                    "extraction_datetime": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
                }]
            }],
            "confidence": match.confidence,
            "resolution_method": "semantic"
        }
        
        # Add bidirectional SAME_AS edges
        result1 = graph_store.add_edge(
            subject=match.entity1_id,
            predicate="SAME_AS",
            obj=match.entity2_id,
            metadata=metadata
        )
        
        result2 = graph_store.add_edge(
            subject=match.entity2_id,
            predicate="SAME_AS",
            obj=match.entity1_id,
            metadata=metadata
        )
        
        if result1.get('success') or result2.get('success'):
            created_count += 1
    
    return created_count


def create_same_as_edges_for_edges(
    graph_store: 'GraphStore',
    matches: List[EdgeMatch]
) -> int:
    """Create SAME_AS_EDGE relationships for duplicate edges.
    
    Note: This creates special markers to indicate edge duplicates.
    Since edges can't point to other edges directly in property graphs,
    we use a naming convention: SAME_AS_EDGE:[edge_id]
    
    Args:
        graph_store: GraphStore to add relationships to
        matches: List of EdgeMatch objects for duplicates
    
    Returns:
        Number of SAME_AS_EDGE markers created
    """
    created_count = 0
    
    for match in matches:
        if not match.is_duplicate:
            continue
        
        # Parse edge IDs (format: subject|predicate|object)
        try:
            parts1 = match.edge1_id.split('|')
            parts2 = match.edge2_id.split('|')
            
            if len(parts1) != 3 or len(parts2) != 3:
                continue
            
            # Store edge duplicate information as metadata on both edges
            # This is implementation-specific and will be used for querying
            edge1_data = graph_store.get_edge(parts1[0], parts1[1], parts1[2])
            edge2_data = graph_store.get_edge(parts2[0], parts2[1], parts2[2])
            
            if not edge1_data or not edge2_data:
                continue
            
            # Update metadata to track duplicates
            for edge_data in edge1_data:
                current_meta = edge_data.get('metadata', {})
                if 'duplicate_of' not in current_meta:
                    current_meta['duplicate_of'] = []
                if match.edge2_id not in current_meta['duplicate_of']:
                    current_meta['duplicate_of'].append(match.edge2_id)
                current_meta['resolution_confidence'] = match.confidence
                current_meta['resolution_reasoning'] = match.reasoning
                
                graph_store.update_edge(
                    parts1[0], parts1[1], parts1[2],
                    updates={'metadata': current_meta}
                )
            
            for edge_data in edge2_data:
                current_meta = edge_data.get('metadata', {})
                if 'duplicate_of' not in current_meta:
                    current_meta['duplicate_of'] = []
                if match.edge1_id not in current_meta['duplicate_of']:
                    current_meta['duplicate_of'].append(match.edge1_id)
                current_meta['resolution_confidence'] = match.confidence
                current_meta['resolution_reasoning'] = match.reasoning
                
                graph_store.update_edge(
                    parts2[0], parts2[1], parts2[2],
                    updates={'metadata': current_meta}
                )
            
            created_count += 1
            
        except Exception:
            continue
    
    return created_count


def get_duplicate_clusters(graph_store: 'GraphStore') -> List[Set[str]]:
    """Get connected components of entities via SAME_AS edges.
    
    Args:
        graph_store: GraphStore to query
    
    Returns:
        List of sets, each containing entity names in a duplicate cluster
    """
    # Query all SAME_AS edges
    same_as_edges = graph_store.query_by_pattern(predicate="SAME_AS")
    
    # Extract edges as tuples
    edge_tuples = [
        (edge['subject'], edge['object'])
        for edge in same_as_edges
    ]
    
    # Find connected components
    return find_connected_components(edge_tuples)

