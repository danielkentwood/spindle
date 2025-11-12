"""Semantic blocking for entity resolution.

Clusters entities using semantic embeddings to reduce pairwise comparisons
from O(n²) to O(n*k) where k is average block size.
"""

from typing import Any, Dict, List
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

from spindle.entity_resolution.config import ResolutionConfig
from spindle.entity_resolution.utils import (
    _record_resolution_event,
    compute_cosine_similarity,
)


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

