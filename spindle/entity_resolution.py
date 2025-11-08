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

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING
from datetime import datetime
import json

import numpy as np
try:
    from sklearn.cluster import KMeans, AgglomerativeClustering
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

try:
    from sklearn.cluster import HDBSCAN as SKLearnHDBSCAN
    _HDBSCAN_AVAILABLE = True
except ImportError:
    try:
        import hdbscan
        _HDBSCAN_AVAILABLE = True
    except ImportError:
        _HDBSCAN_AVAILABLE = False

if TYPE_CHECKING:
    from spindle.graph_store import GraphStore
    from spindle.vector_store import VectorStore

from spindle.observability import get_event_recorder
from spindle.baml_client import b
from spindle.baml_client.types import (
    EntityForMatching,
    EdgeForMatching,
    EntityMatch as BamlEntityMatch,
    EdgeMatch as BamlEdgeMatch,
)

ENTITY_RESOLUTION_RECORDER = get_event_recorder("entity_resolution")


def _record_resolution_event(name: str, payload: Dict[str, Any]) -> None:
    """Record an entity resolution event."""
    ENTITY_RESOLUTION_RECORDER.record(name=name, payload=payload)


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
    config: Optional[ResolutionConfig] = None
    
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


# ========== Utility Functions ==========


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


class SemanticBlocker:
    """Cluster entities using semantic embeddings.
    
    Reduces pairwise comparisons from O(n²) to O(n*k) where k is average block size.
    """
    
    def __init__(self, config: ResolutionConfig):
        """Initialize semantic blocker with configuration.
        
        Args:
            config: ResolutionConfig with blocking parameters
        """
        self.config = config
    
    def create_blocks(
        self,
        items: List[Dict[str, Any]],
        embeddings: np.ndarray,
        item_type: str = 'node'
    ) -> List[List[Dict[str, Any]]]:
        """Cluster items into blocks based on embedding similarity.
        
        Args:
            items: List of node or edge dictionaries
            embeddings: Numpy array of embeddings (shape: [n_items, embedding_dim])
            item_type: Type of items ('node' or 'edge')
        
        Returns:
            List of blocks, each containing similar items
        """
        _record_resolution_event(
            "blocking.start",
            {
                "item_count": len(items),
                "item_type": item_type,
                "method": self.config.clustering_method
            }
        )
        
        if len(items) < self.config.min_cluster_size:
            _record_resolution_event(
                "blocking.skip",
                {
                    "item_count": len(items),
                    "reason": "below_min_cluster_size"
                }
            )
            return []
        
        # Choose clustering method
        if self.config.clustering_method in {'kmeans', 'hierarchical'} and not _SKLEARN_AVAILABLE:
            _record_resolution_event(
                "blocking.fallback",
                {
                    "method": self.config.clustering_method,
                    "reason": "sklearn_not_available"
                }
            )
            blocks = self._cluster_threshold(items, embeddings)
        elif self.config.clustering_method == 'kmeans':
            blocks = self._cluster_kmeans(items, embeddings)
        elif self.config.clustering_method == 'hierarchical':
            blocks = self._cluster_hierarchical(items, embeddings)
        elif self.config.clustering_method == 'hdbscan':
            blocks = self._cluster_hdbscan(items, embeddings)
        else:
            raise ValueError(f"Unknown clustering method: {self.config.clustering_method}")
        
        # Filter blocks by size constraints
        filtered_blocks = [
            block for block in blocks
            if self.config.min_cluster_size <= len(block) <= self.config.max_cluster_size
        ]
        
        _record_resolution_event(
            "blocking.complete",
            {
                "item_count": len(items),
                "blocks_created": len(filtered_blocks),
                "avg_block_size": np.mean([len(b) for b in filtered_blocks]) if filtered_blocks else 0
            }
        )
        
        return filtered_blocks
    
    def _cluster_kmeans(
        self,
        items: List[Dict[str, Any]],
        embeddings: np.ndarray
    ) -> List[List[Dict[str, Any]]]:
        """Cluster using K-means algorithm."""
        # Estimate number of clusters based on similarity threshold
        n_items = len(items)
        n_clusters = max(2, min(n_items // 5, 20))  # Heuristic: 5 items per cluster, max 20 clusters
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        # Group items by cluster label
        blocks: Dict[int, List[Dict[str, Any]]] = {}
        for item, label in zip(items, labels):
            if label not in blocks:
                blocks[label] = []
            blocks[label].append(item)
        
        return list(blocks.values())
    
    def _cluster_hierarchical(
        self,
        items: List[Dict[str, Any]],
        embeddings: np.ndarray
    ) -> List[List[Dict[str, Any]]]:
        """Cluster using hierarchical agglomerative clustering."""
        # Use distance threshold based on blocking_threshold
        # Convert cosine similarity threshold to distance threshold
        distance_threshold = 1.0 - self.config.blocking_threshold
        
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric='cosine',
            linkage='average'
        )
        labels = clustering.fit_predict(embeddings)
        
        # Group items by cluster label
        blocks: Dict[int, List[Dict[str, Any]]] = {}
        for item, label in zip(items, labels):
            if label not in blocks:
                blocks[label] = []
            blocks[label].append(item)
        
        return list(blocks.values())
    
    def _cluster_hdbscan(
        self,
        items: List[Dict[str, Any]],
        embeddings: np.ndarray
    ) -> List[List[Dict[str, Any]]]:
        """Cluster using HDBSCAN density-based clustering."""
        if not _HDBSCAN_AVAILABLE:
            raise ImportError(
                "HDBSCAN clustering requires hdbscan package. "
                "Install it with: pip install hdbscan"
            )
        
        try:
            # Try sklearn's HDBSCAN first (newer versions)
            clusterer = SKLearnHDBSCAN(
                min_cluster_size=self.config.min_cluster_size,
                metric='cosine'
            )
        except NameError:
            # Fall back to standalone hdbscan package
            import hdbscan
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.config.min_cluster_size,
                metric='cosine'
            )
        
        labels = clusterer.fit_predict(embeddings)
        
        # Group items by cluster label (-1 is noise, exclude it)
        blocks: Dict[int, List[Dict[str, Any]]] = {}
        for item, label in zip(items, labels):
            if label == -1:  # Skip noise points
                continue
            if label not in blocks:
                blocks[label] = []
            blocks[label].append(item)
        
        return list(blocks.values())

    def _cluster_threshold(
        self,
        items: List[Dict[str, Any]],
        embeddings: np.ndarray
    ) -> List[List[Dict[str, Any]]]:
        """Fallback clustering based on cosine similarity threshold.

        Groups items by comparing each embedding to the first member of each cluster.
        """
        clusters: List[List[int]] = []

        for idx in range(len(items)):
            assigned = False
            for cluster in clusters:
                representative_idx = cluster[0]
                similarity = compute_cosine_similarity(
                    embeddings[idx], embeddings[representative_idx]
                )
                if similarity >= self.config.blocking_threshold:
                    cluster.append(idx)
                    assigned = True
                    break
            if not assigned:
                clusters.append([idx])

        return [[items[i] for i in cluster] for cluster in clusters]


class SemanticMatcher:
    """LLM-based duplicate matching within blocks.
    
    Uses BAML client to call Claude for intelligent duplicate detection.
    """
    
    def __init__(self, config: ResolutionConfig):
        """Initialize semantic matcher with configuration.
        
        Args:
            config: ResolutionConfig with matching parameters
        """
        self.config = config
    
    def match_entities(
        self,
        block: List[Dict[str, Any]],
        context: str = ""
    ) -> List[EntityMatch]:
        """Find duplicate entities within a block using LLM.
        
        Args:
            block: List of entity dictionaries to match
            context: Optional context about the domain/ontology
        
        Returns:
            List of EntityMatch objects for duplicates found
        """
        _record_resolution_event(
            "matching.entities.start",
            {
                "block_size": len(block),
                "has_context": bool(context)
            }
        )
        
        if len(block) < 2:
            return []
        
        # Convert entities to BAML format
        entities_for_matching = []
        for entity in block:
            # Serialize attributes
            attrs = entity.get('custom_atts', {})
            if isinstance(attrs, str):
                attrs_str = attrs
            else:
                attrs_str = json.dumps(attrs)
            
            entities_for_matching.append(EntityForMatching(
                id=entity['name'],
                type=entity.get('type', ''),
                description=entity.get('description', ''),
                attributes=attrs_str
            ))
        
        # Process in batches if block is large
        all_matches = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(entities_for_matching), batch_size):
            batch = entities_for_matching[i:i + batch_size]
            
            try:
                # Call BAML function
                result = b.MatchEntities(
                    entities=batch,
                    context=context
                )
                
                # Convert BAML matches to our EntityMatch format
                # and filter by confidence threshold
                for match in result.matches:
                    # Convert confidence level to numeric score
                    confidence_score = self._confidence_level_to_score(match.confidence_level)
                    
                    if confidence_score >= self.config.matching_threshold:
                        all_matches.append(EntityMatch(
                            entity1_id=match.entity1_id,
                            entity2_id=match.entity2_id,
                            is_duplicate=match.is_duplicate,
                            confidence=confidence_score,
                            reasoning=match.reasoning
                        ))
            except Exception as e:
                _record_resolution_event(
                    "matching.entities.error",
                    {
                        "batch_index": i,
                        "batch_size": len(batch),
                        "error": str(e)
                    }
                )
                # Continue with next batch
                continue
        
        _record_resolution_event(
            "matching.entities.complete",
            {
                "block_size": len(block),
                "matches_found": len(all_matches)
            }
        )
        
        return all_matches
    
    def match_edges(
        self,
        block: List[Dict[str, Any]],
        context: str = ""
    ) -> List[EdgeMatch]:
        """Find duplicate edges within a block using LLM.
        
        Args:
            block: List of edge dictionaries to match
            context: Optional context about the domain/ontology
        
        Returns:
            List of EdgeMatch objects for duplicates found
        """
        _record_resolution_event(
            "matching.edges.start",
            {
                "block_size": len(block),
                "has_context": bool(context)
            }
        )
        
        if len(block) < 2:
            return []
        
        # Convert edges to BAML format
        edges_for_matching = []
        for edge in block:
            # Create edge ID
            edge_id = f"{edge['subject']}|{edge['predicate']}|{edge['object']}"
            
            # Summarize evidence
            evidence = edge.get('supporting_evidence', [])
            if isinstance(evidence, str):
                try:
                    evidence = json.loads(evidence)
                except json.JSONDecodeError:
                    evidence = []
            
            evidence_summary = ""
            if isinstance(evidence, list) and evidence:
                source_count = len(evidence)
                evidence_summary = f"{source_count} source(s)"
                
                # Add sample text from first source
                first_source = evidence[0]
                if isinstance(first_source, dict) and first_source.get('spans'):
                    spans = first_source['spans']
                    if spans and isinstance(spans, list):
                        first_span = spans[0]
                        if isinstance(first_span, dict) and first_span.get('text'):
                            sample = first_span['text'][:100]
                            evidence_summary += f": {sample}..."
            
            edges_for_matching.append(EdgeForMatching(
                id=edge_id,
                subject=edge['subject'],
                predicate=edge['predicate'],
                object=edge['object'],
                evidence_summary=evidence_summary
            ))
        
        # Process in batches if block is large
        all_matches = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(edges_for_matching), batch_size):
            batch = edges_for_matching[i:i + batch_size]
            
            try:
                # Call BAML function
                result = b.MatchEdges(
                    edges=batch,
                    context=context
                )
                
                # Convert BAML matches to our EdgeMatch format
                # and filter by confidence threshold
                for match in result.matches:
                    # Convert confidence level to numeric score
                    confidence_score = self._confidence_level_to_score(match.confidence_level)
                    
                    if confidence_score >= self.config.matching_threshold:
                        all_matches.append(EdgeMatch(
                            edge1_id=match.edge1_id,
                            edge2_id=match.edge2_id,
                            is_duplicate=match.is_duplicate,
                            confidence=confidence_score,
                            reasoning=match.reasoning
                        ))
            except Exception as e:
                _record_resolution_event(
                    "matching.edges.error",
                    {
                        "batch_index": i,
                        "batch_size": len(batch),
                        "error": str(e)
                    }
                )
                # Continue with next batch
                continue
        
        _record_resolution_event(
            "matching.edges.complete",
            {
                "block_size": len(block),
                "matches_found": len(all_matches)
            }
        )
        
        return all_matches
    
    def _confidence_level_to_score(self, level: str) -> float:
        """Convert confidence level string to numeric score.
        
        Args:
            level: Confidence level ('high', 'medium', or 'low')
        
        Returns:
            Numeric confidence score (0.0-1.0)
        """
        level_map = {
            'high': 0.95,
            'medium': 0.75,
            'low': 0.50
        }
        return level_map.get(level.lower(), 0.50)


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

