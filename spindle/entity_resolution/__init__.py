"""
Semantic Entity Resolution Module

This module provides semantic entity resolution for knowledge graphs using:
- Semantic Blocking: Clustering embeddings to group similar entities
- Semantic Matching: LLM-based duplicate detection with confidence scores  
- Merging: Creating SAME_AS edges and consolidating properties

Key Features:
- Standalone module that works independently
- Preserves original nodes/edges with SAME_AS relationships
- Supports both node and edge deduplication
- Uses BAML/Claude for intelligent matching
- Integrates with existing VectorStore and GraphStore

Example:
    >>> from spindle import EntityResolver, ResolutionConfig
    >>> 
    >>> config = ResolutionConfig(
    ...     blocking_threshold=0.85,
    ...     matching_threshold=0.8,
    ...     clustering_method='hierarchical'
    ... )
    >>> 
    >>> resolver = EntityResolver(config=config)
    >>> result = resolver.resolve_entities(
    ...     graph_store=store,
    ...     vector_store=vec_store,
    ...     apply_to_nodes=True,
    ...     apply_to_edges=True
    ... )
"""

# Import configuration
from spindle.entity_resolution.config import ResolutionConfig

# Import data models
from spindle.entity_resolution.models import (
    EntityMatch,
    EdgeMatch,
    ResolutionResult,
)

# Import utility functions
from spindle.entity_resolution.utils import (
    compute_cosine_similarity,
    find_connected_components,
    serialize_node_for_embedding,
    serialize_edge_for_embedding,
    merge_node_metadata,
    merge_edge_metadata,
)

# Import core classes
from spindle.entity_resolution.blocking import SemanticBlocker
from spindle.entity_resolution.matching import SemanticMatcher

# Import merging functions
from spindle.entity_resolution.merging import (
    create_same_as_edges,
    create_same_as_edges_for_edges,
    get_duplicate_clusters,
)

# Import resolver
from spindle.entity_resolution.resolver import (
    EntityResolver,
    resolve_entities,
)

__all__ = [
    # Configuration
    "ResolutionConfig",
    # Data models
    "EntityMatch",
    "EdgeMatch",
    "ResolutionResult",
    # Core classes
    "SemanticBlocker",
    "SemanticMatcher",
    "EntityResolver",
    # Utility functions
    "compute_cosine_similarity",
    "find_connected_components",
    "serialize_node_for_embedding",
    "serialize_edge_for_embedding",
    "merge_node_metadata",
    "merge_edge_metadata",
    # Merging functions
    "create_same_as_edges",
    "create_same_as_edges_for_edges",
    "get_duplicate_clusters",
    # Convenience function
    "resolve_entities",
]

