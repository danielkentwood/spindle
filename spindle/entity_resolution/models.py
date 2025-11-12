"""Data models for entity resolution."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from spindle.entity_resolution.config import ResolutionConfig


@dataclass
class EntityMatch:
    """Represents a matched pair of duplicate entities.
    
    Attributes:
        entity1_id: ID of first entity
        entity2_id: ID of second entity
        is_duplicate: Whether they are duplicates
        confidence: Match confidence score (0-1)
        reasoning: LLM explanation for the match decision
    """
    entity1_id: str
    entity2_id: str
    is_duplicate: bool
    confidence: float
    reasoning: str


@dataclass
class EdgeMatch:
    """Represents a matched pair of duplicate edges.
    
    Attributes:
        edge1_id: ID of first edge (subject|predicate|object)
        edge2_id: ID of second edge
        is_duplicate: Whether they are duplicates
        confidence: Match confidence score (0-1)
        reasoning: LLM explanation for the match decision
    """
    edge1_id: str
    edge2_id: str
    is_duplicate: bool
    confidence: float
    reasoning: str


@dataclass
class ResolutionResult:
    """Results from entity resolution pipeline.
    
    Attributes:
        total_nodes_processed: Number of nodes analyzed
        total_edges_processed: Number of edges analyzed
        blocks_created: Number of clustering blocks formed
        same_as_edges_created: Number of SAME_AS relationships added
        duplicate_clusters: Number of connected component clusters
        node_matches: List of node duplicate matches
        edge_matches: List of edge duplicate matches
        execution_time_seconds: Total pipeline execution time
        config: Configuration used for resolution
    """
    total_nodes_processed: int = 0
    total_edges_processed: int = 0
    blocks_created: int = 0
    same_as_edges_created: int = 0
    duplicate_clusters: int = 0
    node_matches: List[EntityMatch] = field(default_factory=list)
    edge_matches: List[EdgeMatch] = field(default_factory=list)
    execution_time_seconds: float = 0.0
    config: Optional['ResolutionConfig'] = None  # type: ignore
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "total_nodes_processed": self.total_nodes_processed,
            "total_edges_processed": self.total_edges_processed,
            "blocks_created": self.blocks_created,
            "same_as_edges_created": self.same_as_edges_created,
            "duplicate_clusters": self.duplicate_clusters,
            "node_match_count": len(self.node_matches),
            "edge_match_count": len(self.edge_matches),
            "execution_time_seconds": self.execution_time_seconds,
            "config": {
                "blocking_threshold": self.config.blocking_threshold if self.config else None,
                "matching_threshold": self.config.matching_threshold if self.config else None,
                "clustering_method": self.config.clustering_method if self.config else None,
            } if self.config else None
        }

