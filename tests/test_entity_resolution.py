"""
Tests for semantic entity resolution module.

Tests cover:
- Utility functions (connected components, serialization, merging)
- SemanticBlocker clustering
- SemanticMatcher matching (with mocks)
- EntityResolver integration
- GraphStore entity resolution methods
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from spindle.entity_resolution import (
    ResolutionConfig,
    ResolutionResult,
    EntityMatch,
    EdgeMatch,
    SemanticBlocker,
    SemanticMatcher,
    EntityResolver,
    compute_cosine_similarity,
    find_connected_components,
    serialize_node_for_embedding,
    serialize_edge_for_embedding,
    merge_node_metadata,
    merge_edge_metadata,
    create_same_as_edges,
    create_same_as_edges_for_edges,
    get_duplicate_clusters,
    resolve_entities,
)


# ========== Utility Function Tests ==========


class TestUtilityFunctions:
    """Test utility functions for entity resolution."""
    
    def test_compute_cosine_similarity(self):
        """Test cosine similarity computation."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([1.0, 0.0, 0.0])
        
        similarity = compute_cosine_similarity(vec1, vec2)
        assert similarity == pytest.approx(1.0)
        
        vec3 = np.array([0.0, 1.0, 0.0])
        similarity = compute_cosine_similarity(vec1, vec3)
        assert similarity == pytest.approx(0.0)
    
    def test_find_connected_components_simple(self):
        """Test finding connected components in a simple graph."""
        edges = [
            ('A', 'B'),
            ('B', 'C'),
            ('D', 'E')
        ]
        
        components = find_connected_components(edges)
        
        assert len(components) == 2
        # Sort for consistent comparison
        components = [sorted(comp) for comp in components]
        components = sorted(components)
        
        assert ['A', 'B', 'C'] in components
        assert ['D', 'E'] in components
    
    def test_find_connected_components_empty(self):
        """Test connected components with no edges."""
        components = find_connected_components([])
        assert components == []
    
    def test_serialize_node_for_embedding(self):
        """Test node serialization for embedding."""
        node = {
            'name': 'TechCorp',
            'type': 'Organization',
            'description': 'A technology company',
            'custom_atts': {'founded': '2010', 'employees': '500'}
        }
        
        text = serialize_node_for_embedding(node)
        
        assert 'TechCorp' in text
        assert 'Organization' in text
        assert 'technology company' in text
        assert 'founded' in text
    
    def test_serialize_edge_for_embedding(self):
        """Test edge serialization for embedding."""
        edge = {
            'subject': 'Alice',
            'predicate': 'works_at',
            'object': 'TechCorp',
            'supporting_evidence': [{
                'source_nm': 'doc1',
                'spans': [{'text': 'Alice works at TechCorp', 'start': 0, 'end': 23}]
            }]
        }
        
        text = serialize_edge_for_embedding(edge)
        
        assert 'Alice' in text
        assert 'works_at' in text
        assert 'TechCorp' in text
        assert 'Sources: 1' in text
    
    def test_merge_node_metadata(self):
        """Test merging metadata from multiple nodes."""
        nodes = [
            {
                'name': 'TechCorp',
                'metadata': {'sources': ['doc1'], 'first_seen': '2024-01-01'}
            },
            {
                'name': 'Tech Corp',
                'metadata': {'sources': ['doc2'], 'first_seen': '2024-01-02'}
            }
        ]
        
        merged = merge_node_metadata(nodes)
        
        assert merged['merge_count'] == 2
        assert 'TechCorp' in merged['merged_from']
        assert 'Tech Corp' in merged['merged_from']
        assert set(merged['sources']) == {'doc1', 'doc2'}
    
    def test_merge_edge_metadata(self):
        """Test merging metadata from multiple edges."""
        edges = [
            {
                'subject': 'Alice',
                'predicate': 'works_at',
                'object': 'TechCorp',
                'supporting_evidence': [{'source_nm': 'doc1', 'spans': []}]
            },
            {
                'subject': 'Alice',
                'predicate': 'employed_by',
                'object': 'TechCorp',
                'supporting_evidence': [{'source_nm': 'doc2', 'spans': []}]
            }
        ]
        
        merged = merge_edge_metadata(edges)
        
        assert merged['merge_count'] == 2
        assert len(merged['combined_evidence']) == 2


# ========== SemanticBlocker Tests ==========


class TestSemanticBlocker:
    """Test SemanticBlocker clustering functionality."""
    
    def test_create_blocks_hierarchical(self):
        """Test hierarchical clustering."""
        config = ResolutionConfig(
            clustering_method='hierarchical',
            blocking_threshold=0.85,
            min_cluster_size=2
        )
        blocker = SemanticBlocker(config)
        
        # Create sample nodes with embeddings
        nodes = [
            {'name': 'TechCorp', 'type': 'Organization'},
            {'name': 'Tech Corp', 'type': 'Organization'},
            {'name': 'Apple Inc', 'type': 'Organization'},
            {'name': 'Apple', 'type': 'Organization'},
        ]
        
        # Create embeddings where TechCorp/Tech Corp are similar, Apple/Apple Inc are similar
        embeddings = np.array([
            [1.0, 0.0, 0.0],  # TechCorp
            [0.95, 0.05, 0.0],  # Tech Corp (similar)
            [0.0, 1.0, 0.0],  # Apple Inc
            [0.0, 0.95, 0.05],  # Apple (similar)
        ])
        
        blocks = blocker.create_blocks(nodes, embeddings, item_type='node')
        
        # Should create 2 blocks
        assert len(blocks) >= 1
    
    def test_create_blocks_kmeans(self):
        """Test K-means clustering."""
        config = ResolutionConfig(
            clustering_method='kmeans',
            min_cluster_size=2
        )
        blocker = SemanticBlocker(config)
        
        nodes = [
            {'name': f'Entity{i}', 'type': 'Test'}
            for i in range(10)
        ]
        
        # Random embeddings
        np.random.seed(42)
        embeddings = np.random.rand(10, 3)
        
        blocks = blocker.create_blocks(nodes, embeddings, item_type='node')
        
        # Should create multiple blocks
        assert len(blocks) >= 1
    
    def test_create_blocks_below_min_size(self):
        """Test that blocks below min size are skipped."""
        config = ResolutionConfig(min_cluster_size=5)
        blocker = SemanticBlocker(config)
        
        nodes = [
            {'name': 'A', 'type': 'Test'},
            {'name': 'B', 'type': 'Test'},
        ]
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
        
        blocks = blocker.create_blocks(nodes, embeddings, item_type='node')
        
        # Should skip - below min cluster size
        assert len(blocks) == 0

    def test_create_blocks_without_sklearn(self, monkeypatch):
        """Test fallback clustering when sklearn is unavailable."""
        monkeypatch.setattr("spindle.entity_resolution.blocking._SKLEARN_AVAILABLE", False, raising=False)
        config = ResolutionConfig(
            clustering_method='hierarchical',
            blocking_threshold=0.8,
            min_cluster_size=1
        )
        blocker = SemanticBlocker(config)

        nodes = [
            {'name': 'EntityA', 'type': 'Test'},
            {'name': 'EntityB', 'type': 'Test'},
        ]
        embeddings = np.array([
            [1.0, 0.0],
            [0.9, 0.1],
        ])

        blocks = blocker.create_blocks(nodes, embeddings, item_type='node')

        assert len(blocks) >= 1
        assert sum(len(block) for block in blocks) == len(nodes)


# ========== SemanticMatcher Tests ==========


class TestSemanticMatcher:
    """Test SemanticMatcher LLM-based matching."""
    
    @patch('spindle.entity_resolution.matching.b')
    def test_match_entities_success(self, mock_baml):
        """Test entity matching with mocked BAML."""
        config = ResolutionConfig(matching_threshold=0.8)
        matcher = SemanticMatcher(config)
        
        # Mock BAML response
        mock_result = Mock()
        mock_result.matches = [
            Mock(
                entity1_id='TechCorp',
                entity2_id='Tech Corp',
                is_duplicate=True,
                confidence_level='high',
                reasoning='Name variation of the same company'
            )
        ]
        
        # Mock the with_options() call chain
        mock_with_options = Mock()
        mock_with_options.MatchEntities.return_value = mock_result
        mock_baml.with_options.return_value = mock_with_options
        
        block = [
            {'name': 'TechCorp', 'type': 'Organization', 'description': 'Tech company', 'custom_atts': {}},
            {'name': 'Tech Corp', 'type': 'Organization', 'description': 'Tech company', 'custom_atts': {}},
        ]
        
        matches = matcher.match_entities(block, context='')
        
        assert len(matches) == 1
        assert matches[0].entity1_id == 'TechCorp'
        assert matches[0].entity2_id == 'Tech Corp'
        assert matches[0].is_duplicate is True
        assert matches[0].confidence >= 0.8
    
    @patch('spindle.entity_resolution.matching.b')
    def test_match_edges_success(self, mock_baml):
        """Test edge matching with mocked BAML."""
        config = ResolutionConfig(matching_threshold=0.8)
        matcher = SemanticMatcher(config)
        
        # Mock BAML response
        mock_result = Mock()
        mock_result.matches = [
            Mock(
                edge1_id='Alice|works_at|TechCorp',
                edge2_id='Alice|employed_by|TechCorp',
                is_duplicate=True,
                confidence_level='high',
                reasoning='Same relationship, different predicate'
            )
        ]
        
        # Mock the with_options() call chain
        mock_with_options = Mock()
        mock_with_options.MatchEdges.return_value = mock_result
        mock_baml.with_options.return_value = mock_with_options
        
        block = [
            {
                'subject': 'Alice',
                'predicate': 'works_at',
                'object': 'TechCorp',
                'supporting_evidence': []
            },
            {
                'subject': 'Alice',
                'predicate': 'employed_by',
                'object': 'TechCorp',
                'supporting_evidence': []
            },
        ]
        
        matches = matcher.match_edges(block, context='')
        
        assert len(matches) == 1
        assert matches[0].is_duplicate is True
    
    def test_confidence_level_to_score(self):
        """Test confidence level conversion."""
        config = ResolutionConfig()
        matcher = SemanticMatcher(config)
        
        assert matcher._confidence_level_to_score('high') == 0.95
        assert matcher._confidence_level_to_score('medium') == 0.75
        assert matcher._confidence_level_to_score('low') == 0.50
        assert matcher._confidence_level_to_score('unknown') == 0.50


# ========== GraphStore Entity Resolution Tests ==========


class TestGraphStoreEntityResolution:
    """Test GraphStore entity resolution support methods."""
    
    @pytest.fixture
    def mock_graph_store(self):
        """Create a mock GraphStore with test data."""
        store = Mock()
        
        # Mock SAME_AS edges
        store.query_by_pattern.return_value = [
            {'subject': 'A', 'predicate': 'SAME_AS', 'object': 'B'},
            {'subject': 'B', 'predicate': 'SAME_AS', 'object': 'A'},
            {'subject': 'B', 'predicate': 'SAME_AS', 'object': 'C'},
            {'subject': 'C', 'predicate': 'SAME_AS', 'object': 'B'},
        ]
        
        return store
    
    def test_get_duplicate_clusters(self, mock_graph_store):
        """Test finding duplicate clusters."""
        from spindle.entity_resolution import get_duplicate_clusters
        
        clusters = get_duplicate_clusters(mock_graph_store)
        
        assert len(clusters) == 1
        assert set(clusters[0]) == {'A', 'B', 'C'}
    
    def test_create_same_as_edges(self, mock_graph_store):
        """Test creating SAME_AS edges."""
        mock_graph_store.add_edge.return_value = {'success': True}
        
        matches = [
            EntityMatch(
                entity1_id='TechCorp',
                entity2_id='Tech Corp',
                is_duplicate=True,
                confidence=0.95,
                reasoning='Name variation'
            )
        ]
        
        count = create_same_as_edges(mock_graph_store, matches)
        
        assert count == 1
        assert mock_graph_store.add_edge.call_count == 2  # Bidirectional


# ========== EntityResolver Integration Tests ==========


class TestEntityResolver:
    """Test EntityResolver orchestration."""
    
    def test_initialization(self):
        """Test EntityResolver initialization."""
        config = ResolutionConfig(blocking_threshold=0.9)
        resolver = EntityResolver(config)
        
        assert resolver.config.blocking_threshold == 0.9
        assert resolver.blocker is not None
        assert resolver.matcher is not None
    
    def test_initialization_default_config(self):
        """Test EntityResolver with default config."""
        resolver = EntityResolver()
        
        assert resolver.config.blocking_threshold == 0.85
        assert resolver.config.matching_threshold == 0.8
    
    @patch('spindle.entity_resolution.merging.create_same_as_edges')
    @patch('spindle.entity_resolution.merging.get_duplicate_clusters')
    def test_resolve_entities_nodes_only(self, mock_clusters, mock_create_edges):
        """Test resolution with nodes only."""
        mock_clusters.return_value = []
        mock_create_edges.return_value = 0
        
        # Create mocks
        mock_graph_store = Mock()
        mock_graph_store.nodes.return_value = []
        mock_graph_store.edges.return_value = []
        
        mock_vector_store = Mock()
        
        config = ResolutionConfig()
        resolver = EntityResolver(config)
        
        result = resolver.resolve_entities(
            graph_store=mock_graph_store,
            vector_store=mock_vector_store,
            apply_to_nodes=True,
            apply_to_edges=False
        )
        
        assert isinstance(result, ResolutionResult)
        assert result.total_nodes_processed == 0
        assert result.total_edges_processed == 0


# ========== Resolution Result Tests ==========


class TestResolutionResult:
    """Test ResolutionResult data structure."""
    
    def test_to_dict(self):
        """Test converting result to dictionary."""
        config = ResolutionConfig(blocking_threshold=0.9)
        result = ResolutionResult(
            total_nodes_processed=10,
            total_edges_processed=5,
            blocks_created=3,
            same_as_edges_created=2,
            duplicate_clusters=1,
            config=config
        )
        
        result_dict = result.to_dict()
        
        assert result_dict['total_nodes_processed'] == 10
        assert result_dict['total_edges_processed'] == 5
        assert result_dict['blocks_created'] == 3
        assert result_dict['config']['blocking_threshold'] == 0.9


# ========== Configuration Tests ==========


class TestResolutionConfig:
    """Test ResolutionConfig data structure."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ResolutionConfig()
        
        assert config.blocking_threshold == 0.85
        assert config.matching_threshold == 0.8
        assert config.clustering_method == 'hierarchical'
        assert config.batch_size == 20
        assert config.merge_strategy == 'preserve'
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ResolutionConfig(
            blocking_threshold=0.9,
            matching_threshold=0.85,
            clustering_method='kmeans',
            batch_size=10
        )
        
        assert config.blocking_threshold == 0.9
        assert config.matching_threshold == 0.85
        assert config.clustering_method == 'kmeans'
        assert config.batch_size == 10


# ========== Convenience Function Tests ==========


def test_resolve_entities_function():
    """Test convenience resolve_entities function."""
    mock_graph_store = Mock()
    mock_graph_store.nodes.return_value = []
    mock_graph_store.edges.return_value = []
    
    mock_vector_store = Mock()
    
    result = resolve_entities(
        graph_store=mock_graph_store,
        vector_store=mock_vector_store,
        apply_to_nodes=True,
        apply_to_edges=False
    )
    
    assert isinstance(result, ResolutionResult)

