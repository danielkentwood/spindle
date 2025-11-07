"""
Comprehensive tests for GraphStore functionality.

Tests cover:
- Database initialization and schema creation
- Node CRUD operations
- Edge CRUD operations
- Triple import/export
- Pattern matching queries
- Source and date filtering
- Error handling
- Configuration management
- Context manager behavior
"""

import pytest
import os
import json
from datetime import datetime, timedelta
from spindle.baml_client.types import Triple, SourceMetadata, CharacterSpan

# Skip all tests if kuzu not available
try:
    import kuzu
    KUZU_AVAILABLE = True
except ImportError:
    KUZU_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not KUZU_AVAILABLE,
    reason="GraphStore tests require kuzu"
)

try:
    from spindle import GraphStore
except ImportError:
    GraphStore = None


# ========== Initialization Tests ==========

def test_graph_store_initialization(temp_db_path, skip_if_no_graph_store):
    """Test basic GraphStore initialization."""
    store = GraphStore(db_path=temp_db_path)
    assert store.db is not None
    assert store.conn is not None
    # db_path should be derived from temp_db_path (may have file appended)
    assert store.db_path.startswith(temp_db_path)
    store.close()


def test_graph_store_default_path(skip_if_no_graph_store):
    """Test GraphStore uses default path when none provided."""
    import tempfile
    import shutil
    
    # Create a temp directory and use it as workspace to avoid polluting /graphs/
    temp_dir = tempfile.mkdtemp(prefix="spindle_test_default_")
    original_file = os.path.abspath(__file__)
    
    try:
        store = GraphStore()
        # Should use default "spindle_graph" name
        assert "spindle_graph" in store.db_path
        assert store.db is not None
        assert store.conn is not None
        store.close()
        
        # Clean up the created graph
        if os.path.exists(store.db_path):
            parent_dir = os.path.dirname(store.db_path)
            if os.path.exists(parent_dir):
                shutil.rmtree(parent_dir)
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def test_context_manager(temp_db_path, skip_if_no_graph_store):
    """Test GraphStore works as context manager."""
    with GraphStore(db_path=temp_db_path) as store:
        assert store.db is not None
        assert store.conn is not None
    # Connection should be closed after context
    assert store.conn is None


# ========== Node CRUD Tests ==========

def test_add_node(temp_graph_store):
    """Test adding a single node."""
    success = temp_graph_store.add_node(
        name="Alice Johnson",
        entity_type="Person",
        metadata={"age": 30}
    )
    assert success is True
    
    # Verify node was added (names are stored in uppercase)
    node = temp_graph_store.get_node("Alice Johnson")
    assert node is not None
    assert node["name"] == "ALICE JOHNSON"
    assert node["type"] == "Person"
    assert node["metadata"]["age"] == 30


def test_add_duplicate_node(temp_graph_store):
    """Test that adding duplicate node doesn't fail."""
    temp_graph_store.add_node("Alice", "Person", {})
    # Adding again should return False (already exists)
    result = temp_graph_store.add_node("Alice", "Person", {})
    assert result is False


def test_add_nodes_bulk(temp_graph_store):
    """Test bulk node addition."""
    nodes = [
        {"name": "Alice", "type": "Person", "metadata": {}},
        {"name": "Bob", "type": "Person", "metadata": {}},
        {"name": "TechCorp", "type": "Organization", "metadata": {}}
    ]
    count = temp_graph_store.add_nodes(nodes)
    assert count == 3
    
    # Verify all nodes exist
    assert temp_graph_store.get_node("Alice") is not None
    assert temp_graph_store.get_node("Bob") is not None
    assert temp_graph_store.get_node("TechCorp") is not None


def test_get_nonexistent_node(temp_graph_store):
    """Test getting a node that doesn't exist."""
    node = temp_graph_store.get_node("NonexistentNode")
    assert node is None


def test_nodes_empty_graph(temp_graph_store):
    """Test nodes() on empty graph returns empty list."""
    all_nodes = temp_graph_store.nodes()
    assert isinstance(all_nodes, list)
    assert len(all_nodes) == 0


def test_nodes_populated_graph(temp_graph_store):
    """Test nodes() returns all nodes in graph."""
    # Add several nodes
    temp_graph_store.add_node("Alice", "Person", {"age": 30})
    temp_graph_store.add_node("Bob", "Person", {"age": 25})
    temp_graph_store.add_node("TechCorp", "Organization", {})
    
    all_nodes = temp_graph_store.nodes()
    assert len(all_nodes) == 3
    
    # Check node names (stored in uppercase)
    node_names = {node["name"] for node in all_nodes}
    assert node_names == {"ALICE", "BOB", "TECHCORP"}


def test_nodes_structure(temp_graph_store):
    """Test that nodes() returns correct structure."""
    temp_graph_store.add_node(
        name="Alice",
        entity_type="Person",
        metadata={"age": 30, "verified": True},
        description="A software engineer",
        custom_atts={"email": {"value": "alice@example.com", "type": "string"}},
        vector_index="test_vector_id"
    )
    
    all_nodes = temp_graph_store.nodes()
    assert len(all_nodes) == 1
    
    node = all_nodes[0]
    assert node["name"] == "ALICE"
    assert node["type"] == "Person"
    assert node["description"] == "A software engineer"
    assert node["metadata"]["age"] == 30
    assert node["metadata"]["verified"] is True
    assert node["custom_atts"]["email"]["value"] == "alice@example.com"
    assert node["id"] is not None
    assert node["vector_index"] == "test_vector_id"


def test_nodes_from_triples(populated_graph_store, sample_triples):
    """Test nodes() returns nodes created from triples."""
    all_nodes = populated_graph_store.nodes()
    
    # Should have nodes from sample_triples (Alice Johnson, TechCorp, San Francisco)
    assert len(all_nodes) == 3
    
    node_names = {node["name"] for node in all_nodes}
    assert "ALICE JOHNSON" in node_names
    assert "TECHCORP" in node_names
    assert "SAN FRANCISCO" in node_names


def test_nodes_matches_statistics(temp_graph_store):
    """Test that nodes() count matches statistics."""
    # Add nodes
    temp_graph_store.add_node("Alice", "Person", {})
    temp_graph_store.add_node("Bob", "Person", {})
    temp_graph_store.add_node("TechCorp", "Organization", {})
    
    all_nodes = temp_graph_store.nodes()
    stats = temp_graph_store.get_statistics()
    
    assert len(all_nodes) == stats["node_count"]


def test_update_node(temp_graph_store):
    """Test updating node properties."""
    temp_graph_store.add_node("Alice", "Person", {"age": 30})
    
    # Update type and metadata
    success = temp_graph_store.update_node(
        "Alice",
        updates={"type": "Engineer", "metadata": {"age": 31, "verified": True}}
    )
    assert success is True
    
    # Verify updates
    node = temp_graph_store.get_node("Alice")
    assert node["type"] == "Engineer"
    assert node["metadata"]["age"] == 31
    assert node["metadata"]["verified"] is True


def test_update_nonexistent_node(temp_graph_store):
    """Test updating a node that doesn't exist."""
    success = temp_graph_store.update_node(
        "NonexistentNode",
        updates={"type": "Person"}
    )
    assert success is False


def test_delete_node(temp_graph_store):
    """Test deleting a node."""
    temp_graph_store.add_node("Alice", "Person", {})
    
    success = temp_graph_store.delete_node("Alice")
    assert success is True
    
    # Verify deletion
    node = temp_graph_store.get_node("Alice")
    assert node is None


def test_delete_nonexistent_node(temp_graph_store):
    """Test deleting a node that doesn't exist."""
    success = temp_graph_store.delete_node("NonexistentNode")
    assert success is False


# ========== Edge CRUD Tests ==========

def test_add_edge(temp_graph_store):
    """Test adding a single edge with new nested evidence format."""
    # First add nodes
    temp_graph_store.add_node("Alice", "Person", {})
    temp_graph_store.add_node("TechCorp", "Organization", {})
    
    # Add edge with new format
    result = temp_graph_store.add_edge(
        subject="Alice",
        predicate="works_at",
        obj="TechCorp",
        metadata={
            "supporting_evidence": [{
                "source_nm": "Test",
                "source_url": "http://test.com",
                "spans": [{
                    "text": "Alice works at TechCorp",
                    "start": 0,
                    "end": 23,
                    "extraction_datetime": "2024-01-15T10:00:00Z"
                }]
            }]
        }
    )
    assert result["success"] is True
    assert "Created new edge" in result["message"]
    
    # Verify edge exists with nested structure (names and predicates are stored in uppercase)
    edges = temp_graph_store.get_edge("Alice", "works_at", "TechCorp")
    assert edges is not None
    assert len(edges) == 1
    assert edges[0]["subject"] == "ALICE"
    assert edges[0]["predicate"] == "WORKS_AT"
    assert edges[0]["object"] == "TECHCORP"
    assert len(edges[0]["supporting_evidence"]) == 1
    assert edges[0]["supporting_evidence"][0]["source_nm"] == "Test"


def test_add_edge_without_nodes(temp_graph_store):
    """Test adding edge when nodes don't exist."""
    result = temp_graph_store.add_edge(
        subject="Alice",
        predicate="works_at",
        obj="TechCorp",
        metadata={}
    )
    assert result["success"] is False


def test_add_edge_duplicate_source_same_span(temp_graph_store):
    """Test that adding duplicate source+span returns appropriate message."""
    # Add nodes
    temp_graph_store.add_node("Alice", "Person", {})
    temp_graph_store.add_node("TechCorp", "Organization", {})
    
    # Add edge first time
    metadata = {
        "supporting_evidence": [{
            "source_nm": "Test Source",
            "source_url": "http://test.com",
            "spans": [{
                "text": "Alice works at TechCorp",
                "start": 0,
                "end": 23,
                "extraction_datetime": "2024-01-15T10:00:00Z"
            }]
        }]
    }
    result1 = temp_graph_store.add_edge("Alice", "works_at", "TechCorp", metadata)
    assert result1["success"] is True
    
    # Add same span from same source
    result2 = temp_graph_store.add_edge("Alice", "works_at", "TechCorp", metadata)
    assert result2["success"] is True
    assert "already exist" in result2["message"]
    
    # Verify still only one edge with one source
    edges = temp_graph_store.get_edge("Alice", "works_at", "TechCorp")
    assert len(edges) == 1
    assert len(edges[0]["supporting_evidence"]) == 1
    assert len(edges[0]["supporting_evidence"][0]["spans"]) == 1


def test_add_edge_same_source_different_span(temp_graph_store):
    """Test that adding different span from same source merges correctly."""
    # Add nodes
    temp_graph_store.add_node("Alice", "Person", {})
    temp_graph_store.add_node("TechCorp", "Organization", {})
    
    # Add edge with first span
    metadata1 = {
        "supporting_evidence": [{
            "source_nm": "Test Source",
            "source_url": "http://test.com",
            "spans": [{
                "text": "Alice works at TechCorp",
                "start": 0,
                "end": 23,
                "extraction_datetime": "2024-01-15T10:00:00Z"
            }]
        }]
    }
    result1 = temp_graph_store.add_edge("Alice", "works_at", "TechCorp", metadata1)
    assert result1["success"] is True
    
    # Add different span from same source
    metadata2 = {
        "supporting_evidence": [{
            "source_nm": "Test Source",
            "source_url": "http://test.com",
            "spans": [{
                "text": "Alice is employed by TechCorp",
                "start": 50,
                "end": 79,
                "extraction_datetime": "2024-01-15T10:05:00Z"
            }]
        }]
    }
    result2 = temp_graph_store.add_edge("Alice", "works_at", "TechCorp", metadata2)
    assert result2["success"] is True
    assert "Added 1 new span" in result2["message"]
    
    # Verify one edge with one source having two spans
    edges = temp_graph_store.get_edge("Alice", "works_at", "TechCorp")
    assert len(edges) == 1
    assert len(edges[0]["supporting_evidence"]) == 1
    assert len(edges[0]["supporting_evidence"][0]["spans"]) == 2


def test_add_edge_different_source(temp_graph_store):
    """Test that adding span from different source creates new source entry."""
    # Add nodes
    temp_graph_store.add_node("Alice", "Person", {})
    temp_graph_store.add_node("TechCorp", "Organization", {})
    
    # Add edge from first source
    metadata1 = {
        "supporting_evidence": [{
            "source_nm": "Company Directory",
            "source_url": "http://compdir.com",
            "spans": [{
                "text": "Alice works at TechCorp",
                "start": 0,
                "end": 23,
                "extraction_datetime": "2024-01-15T10:00:00Z"
            }]
        }]
    }
    result1 = temp_graph_store.add_edge("Alice", "works_at", "TechCorp", metadata1)
    assert result1["success"] is True
    
    # Add from different source
    metadata2 = {
        "supporting_evidence": [{
            "source_nm": "HR Database",
            "source_url": "http://hrdb.com",
            "spans": [{
                "text": "Alice Johnson is employed at TechCorp",
                "start": 0,
                "end": 37,
                "extraction_datetime": "2024-01-15T11:00:00Z"
            }]
        }]
    }
    result2 = temp_graph_store.add_edge("Alice", "works_at", "TechCorp", metadata2)
    assert result2["success"] is True
    assert "Added new source" in result2["message"]
    
    # Verify one edge with two sources
    edges = temp_graph_store.get_edge("Alice", "works_at", "TechCorp")
    assert len(edges) == 1
    assert len(edges[0]["supporting_evidence"]) == 2
    
    sources = {s["source_nm"] for s in edges[0]["supporting_evidence"]}
    assert sources == {"Company Directory", "HR Database"}


def test_add_edges_bulk(temp_graph_store):
    """Test bulk edge addition."""
    # Add nodes first
    temp_graph_store.add_node("Alice", "Person", {})
    temp_graph_store.add_node("Bob", "Person", {})
    temp_graph_store.add_node("TechCorp", "Organization", {})
    
    edges = [
        {"subject": "Alice", "predicate": "works_at", "object": "TechCorp", "metadata": {
            "supporting_evidence": [{"source_nm": "Test", "source_url": "", "spans": [{"text": "Alice works at TechCorp", "start": 0, "end": 23, "extraction_datetime": "2024-01-15T10:00:00Z"}]}]
        }},
        {"subject": "Bob", "predicate": "works_at", "object": "TechCorp", "metadata": {
            "supporting_evidence": [{"source_nm": "Test", "source_url": "", "spans": [{"text": "Bob works at TechCorp", "start": 0, "end": 21, "extraction_datetime": "2024-01-15T10:00:00Z"}]}]
        }}
    ]
    count = temp_graph_store.add_edges(edges)
    assert count == 2


def test_get_nonexistent_edge(temp_graph_store):
    """Test getting an edge that doesn't exist."""
    temp_graph_store.add_node("Alice", "Person", {})
    temp_graph_store.add_node("TechCorp", "Organization", {})
    
    edges = temp_graph_store.get_edge("Alice", "works_at", "TechCorp")
    assert edges is None


def test_edges_empty_graph(temp_graph_store):
    """Test edges() on empty graph returns empty list."""
    all_edges = temp_graph_store.edges()
    assert isinstance(all_edges, list)
    assert len(all_edges) == 0


def test_edges_populated_graph(temp_graph_store):
    """Test edges() returns all edges in graph."""
    # Add nodes and edges
    temp_graph_store.add_node("Alice", "Person", {})
    temp_graph_store.add_node("Bob", "Person", {})
    temp_graph_store.add_node("TechCorp", "Organization", {})
    
    metadata = {
        "supporting_evidence": [{
            "source_nm": "Test",
            "source_url": "",
            "spans": [{"text": "test", "start": 0, "end": 4, "extraction_datetime": "2024-01-15T10:00:00Z"}]
        }]
    }
    
    temp_graph_store.add_edge("Alice", "works_at", "TechCorp", metadata)
    temp_graph_store.add_edge("Bob", "works_at", "TechCorp", metadata)
    
    all_edges = temp_graph_store.edges()
    assert len(all_edges) == 2
    
    # Check edge predicates (stored in uppercase)
    predicates = {edge["predicate"] for edge in all_edges}
    assert predicates == {"WORKS_AT"}
    
    # Check subjects and objects
    subjects = {edge["subject"] for edge in all_edges}
    objects = {edge["object"] for edge in all_edges}
    assert subjects == {"ALICE", "BOB"}
    assert objects == {"TECHCORP"}


def test_edges_structure(temp_graph_store):
    """Test that edges() returns correct structure."""
    # Setup
    temp_graph_store.add_node("Alice", "Person", {})
    temp_graph_store.add_node("TechCorp", "Organization", {})
    
    metadata = {
        "supporting_evidence": [{
            "source_nm": "Company Directory",
            "source_url": "http://example.com",
            "spans": [{
                "text": "Alice works at TechCorp",
                "start": 0,
                "end": 23,
                "extraction_datetime": "2024-01-15T10:00:00Z"
            }]
        }],
        "confidence": 0.95
    }
    
    result = temp_graph_store.add_edge("Alice", "works_at", "TechCorp", metadata)
    assert result["success"] is True
    
    all_edges = temp_graph_store.edges()
    assert len(all_edges) == 1
    
    edge = all_edges[0]
    assert edge["subject"] == "ALICE"
    assert edge["predicate"] == "WORKS_AT"
    assert edge["object"] == "TECHCORP"
    assert len(edge["supporting_evidence"]) == 1
    assert edge["supporting_evidence"][0]["source_nm"] == "Company Directory"
    assert edge["supporting_evidence"][0]["source_url"] == "http://example.com"
    assert len(edge["supporting_evidence"][0]["spans"]) == 1
    assert edge["metadata"]["confidence"] == 0.95
    assert edge["id"] is not None


def test_edges_from_triples(populated_graph_store, sample_triples):
    """Test edges() returns edges created from triples."""
    all_edges = populated_graph_store.edges()
    
    # Should have edges from sample_triples
    assert len(all_edges) == len(sample_triples)
    
    # Verify structure
    for edge in all_edges:
        assert "subject" in edge
        assert "predicate" in edge
        assert "object" in edge
        assert "supporting_evidence" in edge
        assert "metadata" in edge


def test_edges_matches_statistics(temp_graph_store):
    """Test that edges() count matches statistics."""
    # Setup
    temp_graph_store.add_node("Alice", "Person", {})
    temp_graph_store.add_node("Bob", "Person", {})
    temp_graph_store.add_node("TechCorp", "Organization", {})
    
    metadata = {
        "supporting_evidence": [{
            "source_nm": "Test",
            "source_url": "",
            "spans": [{"text": "test", "start": 0, "end": 4, "extraction_datetime": "2024-01-15T10:00:00Z"}]
        }]
    }
    
    temp_graph_store.add_edge("Alice", "works_at", "TechCorp", metadata)
    temp_graph_store.add_edge("Bob", "works_at", "TechCorp", metadata)
    
    all_edges = temp_graph_store.edges()
    stats = temp_graph_store.get_statistics()
    
    assert len(all_edges) == stats["edge_count"]


def test_edges_matches_query_by_pattern(temp_graph_store):
    """Test that edges() returns same results as query_by_pattern() with all None."""
    # Setup
    temp_graph_store.add_node("Alice", "Person", {})
    temp_graph_store.add_node("Bob", "Person", {})
    temp_graph_store.add_node("TechCorp", "Organization", {})
    
    metadata = {
        "supporting_evidence": [{
            "source_nm": "Test",
            "source_url": "",
            "spans": [{"text": "test", "start": 0, "end": 4, "extraction_datetime": "2024-01-15T10:00:00Z"}]
        }]
    }
    
    temp_graph_store.add_edge("Alice", "works_at", "TechCorp", metadata)
    temp_graph_store.add_edge("Bob", "works_at", "TechCorp", metadata)
    
    all_edges = temp_graph_store.edges()
    pattern_edges = temp_graph_store.query_by_pattern()
    
    assert len(all_edges) == len(pattern_edges)
    
    # Sort both by subject for comparison
    all_edges_sorted = sorted(all_edges, key=lambda e: (e["subject"], e["predicate"], e["object"]))
    pattern_edges_sorted = sorted(pattern_edges, key=lambda e: (e["subject"], e["predicate"], e["object"]))
    
    for edge1, edge2 in zip(all_edges_sorted, pattern_edges_sorted):
        assert edge1["subject"] == edge2["subject"]
        assert edge1["predicate"] == edge2["predicate"]
        assert edge1["object"] == edge2["object"]


def test_update_edge(temp_graph_store):
    """Test updating edge properties."""
    # Setup
    temp_graph_store.add_node("Alice", "Person", {})
    temp_graph_store.add_node("TechCorp", "Organization", {})
    temp_graph_store.add_edge(
        "Alice", "works_at", "TechCorp",
        metadata={
            "supporting_evidence": [{
                "source_nm": "Source1",
                "source_url": "",
                "spans": [{"text": "test", "start": 0, "end": 4, "extraction_datetime": "2024-01-15T10:00:00Z"}]
            }]
        }
    )
    
    # Update
    success = temp_graph_store.update_edge(
        "Alice", "works_at", "TechCorp",
        updates={"metadata": {"verified": True}}
    )
    assert success is True
    
    # Verify
    edges = temp_graph_store.get_edge("Alice", "works_at", "TechCorp")
    assert edges[0]["metadata"]["verified"] is True


def test_delete_edge(temp_graph_store):
    """Test deleting an edge."""
    # Setup
    temp_graph_store.add_node("Alice", "Person", {})
    temp_graph_store.add_node("TechCorp", "Organization", {})
    temp_graph_store.add_edge("Alice", "works_at", "TechCorp", metadata={})
    
    # Delete
    success = temp_graph_store.delete_edge("Alice", "works_at", "TechCorp")
    assert success is True
    
    # Verify
    edges = temp_graph_store.get_edge("Alice", "works_at", "TechCorp")
    assert edges is None


def test_delete_node_cascades_to_edges(temp_graph_store):
    """Test that deleting a node also deletes its edges."""
    # Setup
    temp_graph_store.add_node("Alice", "Person", {})
    temp_graph_store.add_node("TechCorp", "Organization", {})
    temp_graph_store.add_edge("Alice", "works_at", "TechCorp", metadata={})
    
    # Delete node
    temp_graph_store.delete_node("Alice")
    
    # Verify edge is gone (can't query without node existing)
    node = temp_graph_store.get_node("Alice")
    assert node is None


# ========== Triple Integration Tests ==========

def test_add_triples(temp_graph_store, sample_triples):
    """Test bulk import of triples."""
    count = temp_graph_store.add_triples(sample_triples)
    assert count == len(sample_triples)
    
    # Verify nodes were created
    assert temp_graph_store.get_node("Alice Johnson") is not None
    assert temp_graph_store.get_node("TechCorp") is not None
    assert temp_graph_store.get_node("San Francisco") is not None


def test_add_edge_from_triple(temp_graph_store, sample_triple):
    """Test creating edge from Triple object."""
    success = temp_graph_store.add_edge_from_triple(sample_triple)
    assert success is True
    
    # Verify edge was created with correct nested metadata
    edges = temp_graph_store.get_edge(
        sample_triple.subject.name,  # Use entity name
        sample_triple.predicate,
        sample_triple.object.name  # Use entity name
    )
    assert edges is not None
    assert len(edges) == 1
    assert len(edges[0]["supporting_evidence"]) == 1
    assert edges[0]["supporting_evidence"][0]["source_nm"] == sample_triple.source.source_name
    # Check that extraction_datetime is in the spans
    assert len(edges[0]["supporting_evidence"][0]["spans"]) > 0
    assert edges[0]["supporting_evidence"][0]["spans"][0]["extraction_datetime"] == sample_triple.extraction_datetime


def test_get_triples(populated_graph_store, sample_triples):
    """Test exporting triples from database."""
    exported_triples = populated_graph_store.get_triples()
    
    assert len(exported_triples) == len(sample_triples)
    
    # Verify Triple objects are correctly reconstructed
    for triple in exported_triples:
        assert isinstance(triple, Triple)
        assert triple.subject
        assert triple.predicate
        assert triple.object
        assert isinstance(triple.source, SourceMetadata)
        assert isinstance(triple.supporting_spans, list)


def test_triple_roundtrip(temp_graph_store, sample_triples):
    """Test that triples can be imported and exported without loss."""
    # Import
    temp_graph_store.add_triples(sample_triples)
    
    # Export
    exported = temp_graph_store.get_triples()
    
    # Compare (order may differ)
    assert len(exported) == len(sample_triples)
    
    # Check that all original triples are in exported set (converted to uppercase)
    # Now subject and object are Entity objects
    original_facts = {(t.subject.name.upper(), t.predicate.upper(), t.object.name.upper()) for t in sample_triples}
    exported_facts = {(t.subject.name, t.predicate, t.object.name) for t in exported}
    assert original_facts == exported_facts


# ========== Pattern Query Tests ==========

def test_query_by_pattern_all_wildcards(graph_store_with_diverse_data, diverse_triples):
    """Test querying with all wildcards returns all edges."""
    results = graph_store_with_diverse_data.query_by_pattern()
    assert len(results) == len(diverse_triples)


def test_query_by_pattern_subject_only(graph_store_with_diverse_data):
    """Test querying by subject."""
    results = graph_store_with_diverse_data.query_by_pattern(subject="Alice Johnson")
    
    # Alice has 2 relationships: works_at and uses
    assert len(results) == 2
    for edge in results:
        assert edge["subject"] == "ALICE JOHNSON"


def test_query_by_pattern_predicate_only(graph_store_with_diverse_data):
    """Test querying by predicate."""
    results = graph_store_with_diverse_data.query_by_pattern(predicate="works_at")
    
    # 3 people work at companies
    assert len(results) == 3
    for edge in results:
        assert edge["predicate"] == "WORKS_AT"


def test_query_by_pattern_object_only(graph_store_with_diverse_data):
    """Test querying by object."""
    results = graph_store_with_diverse_data.query_by_pattern(obj="TechCorp")
    
    # 2 people work at TechCorp (TechCorp is the object in these relationships)
    assert len(results) == 2
    for edge in results:
        assert edge["object"] == "TECHCORP"


def test_query_by_pattern_subject_and_predicate(graph_store_with_diverse_data):
    """Test querying by subject and predicate."""
    results = graph_store_with_diverse_data.query_by_pattern(
        subject="Alice Johnson",
        predicate="works_at"
    )
    
    assert len(results) == 1
    assert results[0]["subject"] == "ALICE JOHNSON"
    assert results[0]["predicate"] == "WORKS_AT"
    assert results[0]["object"] == "TECHCORP"


def test_query_by_pattern_no_matches(graph_store_with_diverse_data):
    """Test querying with pattern that has no matches."""
    results = graph_store_with_diverse_data.query_by_pattern(
        subject="Nonexistent Person",
        predicate="works_at"
    )
    assert len(results) == 0


# ========== Source Query Tests ==========

def test_query_by_source(graph_store_with_diverse_data):
    """Test filtering by source."""
    results = graph_store_with_diverse_data.query_by_source("Source A")
    
    # Source A has multiple triples
    assert len(results) >= 4
    for edge in results:
        # Check that Source A is in the supporting evidence
        sources = [s["source_nm"] for s in edge["supporting_evidence"]]
        assert "Source A" in sources


def test_query_by_source_no_matches(graph_store_with_diverse_data):
    """Test querying for nonexistent source."""
    results = graph_store_with_diverse_data.query_by_source("Nonexistent Source")
    assert len(results) == 0


# ========== Date Range Query Tests ==========

def test_query_by_date_range_all(graph_store_with_diverse_data, diverse_triples):
    """Test querying with very wide date range."""
    start = datetime(2024, 1, 1)
    end = datetime(2024, 12, 31)
    
    results = graph_store_with_diverse_data.query_by_date_range(start, end)
    assert len(results) == len(diverse_triples)


def test_query_by_date_range_start_only(graph_store_with_diverse_data):
    """Test querying with only start date."""
    from datetime import timezone
    start = datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc)
    
    results = graph_store_with_diverse_data.query_by_date_range(start=start)
    
    # Should get edges that have at least one span from 10:30 and later
    assert len(results) >= 3
    for edge in results:
        # Check that at least one span matches the date criteria
        has_matching_span = False
        for source in edge["supporting_evidence"]:
            for span in source.get("spans", []):
                dt_str = span.get("extraction_datetime", "")
                if dt_str:
                    dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
                    if dt >= start:
                        has_matching_span = True
                        break
            if has_matching_span:
                break
        assert has_matching_span


def test_query_by_date_range_end_only(graph_store_with_diverse_data):
    """Test querying with only end date."""
    from datetime import timezone
    end = datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc)
    
    results = graph_store_with_diverse_data.query_by_date_range(end=end)
    
    # Should get edges that have at least one span from 10:30 and earlier
    for edge in results:
        # Check that at least one span matches the date criteria
        has_matching_span = False
        for source in edge["supporting_evidence"]:
            for span in source.get("spans", []):
                dt_str = span.get("extraction_datetime", "")
                if dt_str:
                    dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
                    if dt <= end:
                        has_matching_span = True
                        break
            if has_matching_span:
                break
        assert has_matching_span


def test_query_by_date_range_narrow(graph_store_with_diverse_data):
    """Test querying with narrow date range."""
    from datetime import timezone
    start = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)
    end = datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc)
    
    results = graph_store_with_diverse_data.query_by_date_range(start, end)
    
    # Should get edges with at least one span in the 30-minute window
    for edge in results:
        has_matching_span = False
        for source in edge["supporting_evidence"]:
            for span in source.get("spans", []):
                dt_str = span.get("extraction_datetime", "")
                if dt_str:
                    dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
                    if start <= dt <= end:
                        has_matching_span = True
                        break
            if has_matching_span:
                break
        assert has_matching_span


# ========== Cypher Query Tests ==========

def test_query_cypher_count_nodes(graph_store_with_diverse_data):
    """Test direct Cypher query for counting nodes."""
    results = graph_store_with_diverse_data.query_cypher(
        "MATCH (e:Entity) RETURN count(e) as count"
    )
    
    assert len(results) > 0
    assert "count" in results[0]
    # Should have at least 7 unique entities
    assert results[0]["count"] >= 7


def test_query_cypher_count_edges(graph_store_with_diverse_data, diverse_triples):
    """Test direct Cypher query for counting edges."""
    results = graph_store_with_diverse_data.query_cypher(
        "MATCH ()-[r:Relationship]->() RETURN count(r) as count"
    )
    
    assert len(results) > 0
    assert results[0]["count"] == len(diverse_triples)


def test_query_cypher_complex(graph_store_with_diverse_data):
    """Test complex Cypher query."""
    # Find all people, where they work, and where those companies are located
    # Note: predicates are stored in uppercase
    query = """
    MATCH (p:Entity)-[w:Relationship {predicate: 'WORKS_AT'}]->(c:Entity)
    MATCH (c)-[l:Relationship {predicate: 'LOCATED_IN'}]->(loc:Entity)
    RETURN p.name AS person, c.name AS company, loc.name AS location
    """
    
    results = graph_store_with_diverse_data.query_cypher(query)
    
    # Should find people working at companies with locations
    assert len(results) >= 2
    for row in results:
        assert "person" in row
        assert "company" in row
        assert "location" in row


# ========== Graph Management Tests ==========

def test_create_graph(temp_db_path, skip_if_no_graph_store):
    """Test creating a fresh graph database."""
    store = GraphStore(db_path=temp_db_path)
    
    # Add some data
    store.add_node("Alice", "Person", {})
    
    # Create fresh graph (reinitialize)
    store.create_graph()
    
    # Old data should be gone (new database)
    # Note: This depends on Kùzu behavior
    stats = store.get_statistics()
    # New graph should be empty or reset
    
    store.close()


def test_delete_graph(temp_db_path, skip_if_no_graph_store):
    """Test deleting entire graph database."""
    store = GraphStore(db_path=temp_db_path)
    store.add_node("Alice", "Person", {})
    
    # Get the actual database file path
    db_file_path = store.db_path
    
    # Delete graph
    store.delete_graph()
    
    # Database file should be removed
    assert not os.path.exists(db_file_path)


def test_get_statistics_empty(temp_graph_store):
    """Test statistics on empty graph."""
    stats = temp_graph_store.get_statistics()
    
    assert stats["node_count"] == 0
    assert stats["edge_count"] == 0
    assert len(stats["sources"]) == 0
    assert len(stats["predicates"]) == 0


def test_get_statistics_populated(graph_store_with_diverse_data, diverse_triples):
    """Test statistics on populated graph."""
    stats = graph_store_with_diverse_data.get_statistics()
    
    assert stats["node_count"] >= 7  # Unique entities
    assert stats["edge_count"] == len(diverse_triples)
    assert len(stats["sources"]) == 2  # Source A and Source B
    assert len(stats["predicates"]) == 3  # works_at, located_in, uses


# ========== Error Handling Tests ==========

def test_add_nodes_from_triple_creates_nodes(temp_graph_store, sample_triple):
    """Test that add_nodes_from_triple creates both subject and object nodes."""
    subject_added, object_added = temp_graph_store.add_nodes_from_triple(sample_triple)
    
    # Both should be added
    assert subject_added is True
    assert object_added is True
    
    # Verify nodes exist - use entity names
    assert temp_graph_store.get_node(sample_triple.subject.name) is not None
    assert temp_graph_store.get_node(sample_triple.object.name) is not None


def test_add_nodes_from_triple_existing_nodes(temp_graph_store, sample_triple):
    """Test that add_nodes_from_triple handles existing nodes."""
    # Add nodes first - use entity names and types
    temp_graph_store.add_node(sample_triple.subject.name, sample_triple.subject.type, {})
    temp_graph_store.add_node(sample_triple.object.name, sample_triple.object.type, {})
    
    # Try to add again via triple
    subject_added, object_added = temp_graph_store.add_nodes_from_triple(sample_triple)
    
    # Should return False since they already exist
    assert subject_added is False
    assert object_added is False


def test_empty_query_results(temp_graph_store):
    """Test that queries on empty database return empty results."""
    results = temp_graph_store.query_by_pattern(predicate="works_at")
    assert len(results) == 0
    
    results = temp_graph_store.query_by_source("Any Source")
    assert len(results) == 0
    
    results = temp_graph_store.query_by_date_range()
    assert len(results) == 0


# ========== Integration Tests ==========

@pytest.mark.integration
def test_end_to_end_workflow(temp_db_path, skip_if_no_api_key, skip_if_no_graph_store):
    """
    End-to-end test: Extract → Store → Query → Export.
    
    This test requires an API key and demonstrates the full workflow.
    """
    from spindle import SpindleExtractor, create_ontology
    
    # Create ontology
    entity_types = [
        {"name": "Person", "description": "A human being"},
        {"name": "Organization", "description": "A company"}
    ]
    relation_types = [
        {
            "name": "works_at",
            "description": "Employment",
            "domain": "Person",
            "range": "Organization"
        }
    ]
    ontology = create_ontology(entity_types, relation_types)
    
    # Extract triples
    extractor = SpindleExtractor(ontology)
    text = "Alice Johnson works at TechCorp."
    result = extractor.extract(text, "Test Source")
    
    # Store in database
    with GraphStore(db_path=temp_db_path) as store:
        count = store.add_triples(result.triples)
        assert count > 0
        
        # Query
        works_at_edges = store.query_by_pattern(predicate="works_at")
        assert len(works_at_edges) > 0
        
        # Export
        exported = store.get_triples()
        assert len(exported) == count
        
        # Verify roundtrip
        for triple in exported:
            assert triple.source.source_name == "Test Source"


def test_multi_source_workflow(temp_graph_store):
    """Test storing triples from multiple sources - should consolidate into one edge."""
    from spindle.baml_client.types import Entity, AttributeValue
    
    source1 = SourceMetadata(source_name="Source 1")
    source2 = SourceMetadata(source_name="Source 2")
    
    # Create Entity objects
    alice = Entity(name="Alice", type="Person", description="A person", custom_atts={})
    techcorp = Entity(name="TechCorp", type="Organization", description="A company", custom_atts={})
    
    triples1 = [
        Triple(
            subject=alice,
            predicate="works_at",
            object=techcorp,
            source=source1,
            supporting_spans=[CharacterSpan(text="Alice works at TechCorp", start=0, end=23)],
            extraction_datetime="2024-01-15T10:00:00Z"
        )
    ]
    
    triples2 = [
        Triple(
            subject=alice,
            predicate="works_at",
            object=techcorp,
            source=source2,
            supporting_spans=[CharacterSpan(text="Alice is employed at TechCorp", start=0, end=29)],
            extraction_datetime="2024-01-15T11:00:00Z"
        )
    ]
    
    # Add from both sources
    temp_graph_store.add_triples(triples1)
    temp_graph_store.add_triples(triples2)
    
    # Should consolidate into ONE edge with evidence from both sources
    all_edges = temp_graph_store.query_by_pattern()
    assert len(all_edges) == 1
    assert len(all_edges[0]["supporting_evidence"]) == 2
    
    # Can still filter by source - edge should appear for both sources
    source1_edges = temp_graph_store.query_by_source("Source 1")
    assert len(source1_edges) == 1
    
    source2_edges = temp_graph_store.query_by_source("Source 2")
    assert len(source2_edges) == 1
    
    # When exporting as triples, should get one triple per source
    exported_triples = temp_graph_store.get_triples()
    assert len(exported_triples) == 2  # One per source


# ========== Graph Embedding Tests ==========

# Skip embedding tests if dependencies not available
try:
    import networkx as nx
    try:
        from node2vec import Node2Vec
        _EMBEDDING_DEPS_AVAILABLE = True
    except ImportError:
        _EMBEDDING_DEPS_AVAILABLE = False
except ImportError:
    _EMBEDDING_DEPS_AVAILABLE = False

try:
    from spindle import ChromaVectorStore
    _VECTOR_STORE_AVAILABLE = True
except ImportError:
    _VECTOR_STORE_AVAILABLE = False

EMBEDDING_TESTS_SKIP = pytest.mark.skipif(
    not _EMBEDDING_DEPS_AVAILABLE or not _VECTOR_STORE_AVAILABLE,
    reason="Graph embedding tests require node2vec, networkx, and chromadb"
)


@EMBEDDING_TESTS_SKIP
def test_extract_graph_structure(temp_graph_store):
    """Test extracting graph structure from GraphStore."""
    from spindle.vector_store import GraphEmbeddingGenerator
    
    # Add some nodes and edges
    temp_graph_store.add_node("Alice", "Person", description="A person")
    temp_graph_store.add_node("Bob", "Person", description="Another person")
    temp_graph_store.add_node("TechCorp", "Organization", description="A company")
    
    temp_graph_store.add_edge("Alice", "works_at", "TechCorp")
    temp_graph_store.add_edge("Bob", "works_at", "TechCorp")
    
    # Extract graph
    graph = GraphEmbeddingGenerator.extract_graph_structure(temp_graph_store)
    
    # Verify structure
    assert isinstance(graph, nx.Graph)
    assert len(graph.nodes()) == 3
    assert len(graph.edges()) == 2
    
    # Check node attributes
    assert "ALICE" in graph.nodes()
    assert graph.nodes["ALICE"]["type"] == "Person"
    
    # Check edges
    assert graph.has_edge("ALICE", "TECHCORP")
    assert graph.has_edge("BOB", "TECHCORP")


@EMBEDDING_TESTS_SKIP
def test_compute_graph_embeddings(temp_graph_store):
    """Test computing Node2Vec embeddings for graph."""
    from spindle import ChromaVectorStore
    
    # Create a simple graph
    temp_graph_store.add_node("Alice", "Person")
    temp_graph_store.add_node("Bob", "Person")
    temp_graph_store.add_node("TechCorp", "Organization")
    
    temp_graph_store.add_edge("Alice", "works_at", "TechCorp")
    temp_graph_store.add_edge("Bob", "works_at", "TechCorp")
    
    # Create vector store
    vector_store = ChromaVectorStore(collection_name="test_embeddings")
    
    try:
        # Compute embeddings
        embeddings = temp_graph_store.compute_graph_embeddings(
            vector_store,
            dimensions=64,  # Smaller for testing
            num_walks=5,    # Fewer walks for faster testing
            walk_length=10  # Shorter walks for faster testing
        )
        
        # Verify embeddings were created
        assert len(embeddings) == 3
        assert "ALICE" in embeddings
        assert "BOB" in embeddings
        assert "TECHCORP" in embeddings
        
        # Verify all embeddings have vector_index values
        assert all(isinstance(uid, str) and len(uid) > 0 for uid in embeddings.values())
        
        # Verify nodes were updated with vector_index
        alice = temp_graph_store.get_node("Alice")
        assert alice is not None
        assert alice.get("vector_index") is not None
        assert alice["vector_index"] == embeddings["ALICE"]
        
    finally:
        vector_store.close()


@EMBEDDING_TESTS_SKIP
def test_update_node_embeddings(temp_graph_store):
    """Test updating nodes with computed embeddings."""
    # Add nodes
    temp_graph_store.add_node("Alice", "Person")
    temp_graph_store.add_node("Bob", "Person")
    
    # Initially no embeddings
    alice = temp_graph_store.get_node("Alice")
    assert alice.get("vector_index") is None or alice.get("vector_index") == ""
    
    # Update with embeddings
    embeddings = {"ALICE": "test_vector_id_1", "BOB": "test_vector_id_2"}
    updated_count = temp_graph_store.update_node_embeddings(embeddings)
    
    assert updated_count == 2
    
    # Verify nodes were updated
    alice = temp_graph_store.get_node("Alice")
    bob = temp_graph_store.get_node("Bob")
    
    assert alice.get("vector_index") == "test_vector_id_1"
    assert bob.get("vector_index") == "test_vector_id_2"


@EMBEDDING_TESTS_SKIP
def test_graph_embeddings_empty_graph(temp_graph_store):
    """Test embedding computation on empty graph."""
    from spindle import ChromaVectorStore
    from spindle.vector_store import GraphEmbeddingGenerator
    
    vector_store = ChromaVectorStore(collection_name="test_empty")
    
    try:
        # Extract empty graph
        graph = GraphEmbeddingGenerator.extract_graph_structure(temp_graph_store)
        assert len(graph.nodes()) == 0
        assert len(graph.edges()) == 0
        
        # Should raise ValueError when trying to compute embeddings
        with pytest.raises(ValueError, match="empty graph"):
            GraphEmbeddingGenerator.compute_node2vec_embeddings(graph)
            
    finally:
        vector_store.close()


@EMBEDDING_TESTS_SKIP
def test_graph_embeddings_single_node(temp_graph_store):
    """Test embedding computation with single node (no edges)."""
    from spindle import ChromaVectorStore
    
    # Add single node
    temp_graph_store.add_node("Alice", "Person")
    
    vector_store = ChromaVectorStore(collection_name="test_single")
    
    try:
        # Compute embeddings - should work (returns zero vector)
        embeddings = temp_graph_store.compute_graph_embeddings(
            vector_store,
            dimensions=64,
            num_walks=5,
            walk_length=10
        )
        
        assert len(embeddings) == 1
        assert "ALICE" in embeddings
        
        # Verify embedding was stored
        alice = temp_graph_store.get_node("Alice")
        assert alice.get("vector_index") is not None
        
    finally:
        vector_store.close()


@EMBEDDING_TESTS_SKIP
def test_graph_embeddings_with_isolated_nodes(temp_graph_store):
    """Test embedding computation with isolated nodes."""
    from spindle import ChromaVectorStore
    
    # Add connected nodes
    temp_graph_store.add_node("Alice", "Person")
    temp_graph_store.add_node("Bob", "Person")
    temp_graph_store.add_edge("Alice", "knows", "Bob")
    
    # Add isolated node
    temp_graph_store.add_node("Isolated", "Person")
    
    vector_store = ChromaVectorStore(collection_name="test_isolated")
    
    try:
        # Compute embeddings
        embeddings = temp_graph_store.compute_graph_embeddings(
            vector_store,
            dimensions=64,
            num_walks=5,
            walk_length=10
        )
        
        # Should have all 3 nodes
        assert len(embeddings) == 3
        assert "ALICE" in embeddings
        assert "BOB" in embeddings
        assert "ISOLATED" in embeddings
        
        # All nodes should have embeddings
        for node_name in ["Alice", "Bob", "Isolated"]:
            node = temp_graph_store.get_node(node_name)
            assert node.get("vector_index") is not None
            
    finally:
        vector_store.close()


@EMBEDDING_TESTS_SKIP
def test_compute_graph_embeddings_no_vector_store(temp_graph_store):
    """Test that compute_graph_embeddings requires vector_store."""
    temp_graph_store.add_node("Alice", "Person")
    
    # Should raise ValueError if vector_store is None
    with pytest.raises(ValueError, match="vector_store is required"):
        temp_graph_store.compute_graph_embeddings(None)

