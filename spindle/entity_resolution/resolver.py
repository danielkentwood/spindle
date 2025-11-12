"""Main orchestrator for entity resolution pipeline.

Coordinates blocking, matching, and merging to deduplicate knowledge graphs.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional
from datetime import datetime

import numpy as np

if TYPE_CHECKING:
    from spindle.graph_store import GraphStore
    from spindle.vector_store import VectorStore

from spindle.entity_resolution.config import ResolutionConfig
from spindle.entity_resolution.models import ResolutionResult
from spindle.entity_resolution.blocking import SemanticBlocker
from spindle.entity_resolution.matching import SemanticMatcher
from spindle.entity_resolution.merging import (
    create_same_as_edges,
    create_same_as_edges_for_edges,
    get_duplicate_clusters,
)
from spindle.entity_resolution.utils import (
    _record_resolution_event,
    serialize_node_for_embedding,
    serialize_edge_for_embedding,
)


class EntityResolver:
    """Main orchestrator for entity resolution pipeline.
    
    Coordinates blocking, matching, and merging to deduplicate knowledge graphs.
    """
    
    def __init__(self, config: Optional[ResolutionConfig] = None):
        """Initialize entity resolver.
        
        Args:
            config: Optional ResolutionConfig (uses defaults if not provided)
        """
        self.config = config or ResolutionConfig()
        self.blocker = SemanticBlocker(self.config)
        self.matcher = SemanticMatcher(self.config)
    
    def resolve_entities(
        self,
        graph_store: 'GraphStore',
        vector_store: 'VectorStore',
        apply_to_nodes: bool = True,
        apply_to_edges: bool = True,
        context: str = ""
    ) -> ResolutionResult:
        """Run complete entity resolution pipeline.
        
        Args:
            graph_store: GraphStore containing the knowledge graph
            vector_store: VectorStore for computing embeddings
            apply_to_nodes: Whether to resolve node duplicates
            apply_to_edges: Whether to resolve edge duplicates
            context: Optional context about the domain/ontology
        
        Returns:
            ResolutionResult with statistics and matches
        """
        start_time = datetime.utcnow()
        result = ResolutionResult(config=self.config)
        
        _record_resolution_event(
            "resolution.start",
            {
                "apply_to_nodes": apply_to_nodes,
                "apply_to_edges": apply_to_edges,
                "config": self.config.__dict__
            }
        )
        
        try:
            # Phase 1: Resolve nodes
            if apply_to_nodes:
                node_result = self._resolve_nodes(graph_store, vector_store, context)
                result.total_nodes_processed = node_result['processed']
                result.blocks_created += node_result['blocks']
                result.node_matches = node_result['matches']
                result.same_as_edges_created += node_result['same_as_edges']
                result.duplicate_clusters += node_result['clusters']
            
            # Phase 2: Resolve edges
            if apply_to_edges:
                edge_result = self._resolve_edges(graph_store, vector_store, context)
                result.total_edges_processed = edge_result['processed']
                result.blocks_created += edge_result['blocks']
                result.edge_matches = edge_result['matches']
                result.same_as_edges_created += edge_result['same_as_edges']
                result.duplicate_clusters += edge_result['clusters']
            
            # Calculate execution time
            end_time = datetime.utcnow()
            result.execution_time_seconds = (end_time - start_time).total_seconds()
            
            _record_resolution_event(
                "resolution.complete",
                result.to_dict()
            )
            
            return result
            
        except Exception as exc:
            _record_resolution_event(
                "resolution.error",
                {
                    "error": str(exc),
                    "error_type": type(exc).__name__
                }
            )
            raise
    
    def _resolve_nodes(
        self,
        graph_store: 'GraphStore',
        vector_store: 'VectorStore',
        context: str
    ) -> Dict[str, Any]:
        """Resolve node duplicates.
        
        Returns dict with: processed, blocks, matches, same_as_edges, clusters
        """
        # Get all nodes from graph store
        nodes = graph_store.nodes()
        
        if not nodes or len(nodes) < 2:
            return {
                'processed': len(nodes) if nodes else 0,
                'blocks': 0,
                'matches': [],
                'same_as_edges': 0,
                'clusters': 0
            }
        
        _record_resolution_event(
            "resolve_nodes.start",
            {"node_count": len(nodes)}
        )
        
        # Step 1: Serialize nodes for embedding
        node_texts = [serialize_node_for_embedding(node) for node in nodes]
        
        # Step 2: Compute embeddings
        try:
            embeddings = []
            for text in node_texts:
                embedding = vector_store.compute_embedding(text)
                embeddings.append(embedding)
            embeddings_array = np.array(embeddings)
        except Exception as e:
            _record_resolution_event(
                "resolve_nodes.embedding_error",
                {"error": str(e)}
            )
            # Fall back to single block if embedding fails
            blocks = [nodes]
            _record_resolution_event(
                "resolve_nodes.fallback_single_block",
                {"node_count": len(nodes)}
            )
            embeddings_array = None
        
        # Step 3: Semantic blocking (clustering)
        if embeddings_array is not None:
            blocks = self.blocker.create_blocks(nodes, embeddings_array, item_type='node')
        else:
            blocks = [nodes] if len(nodes) <= self.config.max_cluster_size else []
        
        # Step 4: Semantic matching within each block
        all_matches = []
        for block in blocks:
            if len(block) >= self.config.min_cluster_size:
                matches = self.matcher.match_entities(block, context)
                all_matches.extend(matches)
        
        # Step 5: Create SAME_AS edges
        same_as_count = create_same_as_edges(graph_store, all_matches)
        
        # Step 6: Find connected components (clusters)
        clusters = get_duplicate_clusters(graph_store)
        
        _record_resolution_event(
            "resolve_nodes.complete",
            {
                "processed": len(nodes),
                "blocks": len(blocks),
                "matches": len(all_matches),
                "same_as_edges": same_as_count,
                "clusters": len(clusters)
            }
        )
        
        return {
            'processed': len(nodes),
            'blocks': len(blocks),
            'matches': all_matches,
            'same_as_edges': same_as_count,
            'clusters': len(clusters)
        }
    
    def _resolve_edges(
        self,
        graph_store: 'GraphStore',
        vector_store: 'VectorStore',
        context: str
    ) -> Dict[str, Any]:
        """Resolve edge duplicates.
        
        Returns dict with: processed, blocks, matches, same_as_edges, clusters
        """
        # Get all edges from graph store
        edges = graph_store.edges()
        
        if not edges or len(edges) < 2:
            return {
                'processed': len(edges) if edges else 0,
                'blocks': 0,
                'matches': [],
                'same_as_edges': 0,
                'clusters': 0
            }
        
        _record_resolution_event(
            "resolve_edges.start",
            {"edge_count": len(edges)}
        )
        
        # Step 1: Serialize edges for embedding
        edge_texts = [serialize_edge_for_embedding(edge) for edge in edges]
        
        # Step 2: Compute embeddings
        try:
            embeddings = []
            for text in edge_texts:
                embedding = vector_store.compute_embedding(text)
                embeddings.append(embedding)
            embeddings_array = np.array(embeddings)
        except Exception as e:
            _record_resolution_event(
                "resolve_edges.embedding_error",
                {"error": str(e)}
            )
            # Fall back to single block if embedding fails
            blocks = [edges]
            _record_resolution_event(
                "resolve_edges.fallback_single_block",
                {"edge_count": len(edges)}
            )
            embeddings_array = None
        
        # Step 3: Semantic blocking (clustering)
        if embeddings_array is not None:
            blocks = self.blocker.create_blocks(edges, embeddings_array, item_type='edge')
        else:
            blocks = [edges] if len(edges) <= self.config.max_cluster_size else []
        
        # Step 4: Semantic matching within each block
        all_matches = []
        for block in blocks:
            if len(block) >= self.config.min_cluster_size:
                matches = self.matcher.match_edges(block, context)
                all_matches.extend(matches)
        
        # Step 5: Create SAME_AS_EDGE markers (via metadata)
        same_as_count = create_same_as_edges_for_edges(graph_store, all_matches)
        
        # Step 6: Count duplicate edge groups
        # For edges, we count how many edges have duplicate_of metadata
        edges_with_duplicates = 0
        for edge in edges:
            if edge.get('metadata', {}).get('duplicate_of'):
                edges_with_duplicates += 1
        
        _record_resolution_event(
            "resolve_edges.complete",
            {
                "processed": len(edges),
                "blocks": len(blocks),
                "matches": len(all_matches),
                "same_as_edges": same_as_count,
                "edges_with_duplicates": edges_with_duplicates
            }
        )
        
        return {
            'processed': len(edges),
            'blocks': len(blocks),
            'matches': all_matches,
            'same_as_edges': same_as_count,
            'clusters': edges_with_duplicates
        }


# Convenience function for standalone usage
def resolve_entities(
    graph_store: 'GraphStore',
    vector_store: 'VectorStore',
    config: Optional[ResolutionConfig] = None,
    apply_to_nodes: bool = True,
    apply_to_edges: bool = True,
    context: str = ""
) -> ResolutionResult:
    """Convenience function to run entity resolution.
    
    Args:
        graph_store: GraphStore containing the knowledge graph
        vector_store: VectorStore for computing embeddings
        config: Optional ResolutionConfig (uses defaults if not provided)
        apply_to_nodes: Whether to resolve node duplicates
        apply_to_edges: Whether to resolve edge duplicates
        context: Optional context about the domain/ontology
    
    Returns:
        ResolutionResult with statistics and matches
    
    Example:
        >>> from spindle import resolve_entities, ResolutionConfig
        >>> 
        >>> config = ResolutionConfig(blocking_threshold=0.85)
        >>> result = resolve_entities(
        ...     graph_store=store,
        ...     vector_store=vec_store,
        ...     config=config
        ... )
    """
    resolver = EntityResolver(config=config)
    return resolver.resolve_entities(
        graph_store=graph_store,
        vector_store=vector_store,
        apply_to_nodes=apply_to_nodes,
        apply_to_edges=apply_to_edges,
        context=context
    )

