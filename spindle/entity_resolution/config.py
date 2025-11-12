"""Configuration for entity resolution pipeline."""

from dataclasses import dataclass


@dataclass
class ResolutionConfig:
    """Configuration for entity resolution pipeline.
    
    Attributes:
        blocking_threshold: Cosine similarity threshold for clustering (0-1)
        matching_threshold: LLM confidence threshold for duplicates (0-1)
        clustering_method: Algorithm for blocking ('kmeans', 'hierarchical', 'hdbscan')
        batch_size: Number of entities per LLM matching call
        merge_strategy: How to handle duplicates ('preserve' keeps originals + SAME_AS)
        max_cluster_size: Maximum entities in a cluster (prevents huge LLM calls)
        min_cluster_size: Minimum entities in a cluster (skip clusters below this)
    """
    blocking_threshold: float = 0.85
    matching_threshold: float = 0.8
    clustering_method: str = 'hierarchical'
    batch_size: int = 20
    merge_strategy: str = 'preserve'
    max_cluster_size: int = 50
    min_cluster_size: int = 2

