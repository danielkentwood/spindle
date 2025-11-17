"""Tests for the Spindle REST API."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from spindle.api.main import app
from spindle.api.session import session_manager


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_sessions():
    """Clear sessions before each test."""
    session_manager._sessions.clear()
    yield
    session_manager._sessions.clear()


# ============================================================================
# Health Check Tests
# ============================================================================


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data


# ============================================================================
# Session Management Tests
# ============================================================================


def test_create_session(client):
    """Test creating a new session."""
    response = client.post(
        "/api/sessions",
        json={"name": "test-session"}
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "test-session"
    assert "session_id" in data
    assert "created_at" in data


def test_list_sessions(client):
    """Test listing sessions."""
    # Create a few sessions
    client.post("/api/sessions", json={"name": "session1"})
    client.post("/api/sessions", json={"name": "session2"})
    
    response = client.get("/api/sessions")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert any(s["name"] == "session1" for s in data)
    assert any(s["name"] == "session2" for s in data)


def test_get_session(client):
    """Test getting a specific session."""
    # Create session
    create_response = client.post(
        "/api/sessions",
        json={"name": "test-session"}
    )
    session_id = create_response.json()["session_id"]
    
    # Get session
    response = client.get(f"/api/sessions/{session_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == session_id
    assert data["name"] == "test-session"


def test_get_nonexistent_session(client):
    """Test getting a session that doesn't exist."""
    response = client.get("/api/sessions/nonexistent-id")
    assert response.status_code == 404


def test_delete_session(client):
    """Test deleting a session."""
    # Create session
    create_response = client.post("/api/sessions", json={})
    session_id = create_response.json()["session_id"]
    
    # Delete session
    response = client.delete(f"/api/sessions/{session_id}")
    assert response.status_code == 204
    
    # Verify deleted
    get_response = client.get(f"/api/sessions/{session_id}")
    assert get_response.status_code == 404


def test_update_session_ontology(client):
    """Test updating session ontology."""
    # Create session
    create_response = client.post("/api/sessions", json={})
    session_id = create_response.json()["session_id"]
    
    # Update ontology
    ontology = {
        "entity_types": [{"name": "Person", "description": "A person", "attributes": []}],
        "relation_types": [{"name": "KNOWS", "description": "Knows", "domain": "Person", "range": "Person"}]
    }
    response = client.put(
        f"/api/sessions/{session_id}/ontology",
        json={"ontology": ontology}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["ontology"] == ontology


def test_update_session_config(client):
    """Test updating session configuration."""
    # Create session
    create_response = client.post("/api/sessions", json={})
    session_id = create_response.json()["session_id"]
    
    # Update config
    response = client.put(
        f"/api/sessions/{session_id}/config",
        json={"name": "updated-name", "config": {"key": "value"}}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "updated-name"
    assert data["config"]["key"] == "value"


# ============================================================================
# Extraction Tests
# ============================================================================


def test_extract_triples_without_ontology(client):
    """Test extracting triples with auto-recommendation."""
    request = {
        "text": "Alice works at TechCorp as an engineer.",
        "source_name": "test-doc",
        "ontology_scope": "minimal"
    }
    
    response = client.post("/api/extraction/extract", json=request)
    assert response.status_code == 200
    data = response.json()
    assert "triples" in data
    assert "reasoning" in data
    assert data["source_name"] == "test-doc"
    assert "triple_count" in data


def test_extract_triples_with_ontology(client):
    """Test extracting triples with provided ontology."""
    ontology = {
        "entity_types": [
            {"name": "Person", "description": "A person", "attributes": []},
            {"name": "Organization", "description": "An organization", "attributes": []}
        ],
        "relation_types": [
            {"name": "WORKS_AT", "description": "Works at", "domain": "Person", "range": "Organization"}
        ]
    }
    
    request = {
        "text": "Alice works at TechCorp.",
        "source_name": "test-doc",
        "ontology": ontology
    }
    
    response = client.post("/api/extraction/extract", json=request)
    assert response.status_code == 200
    data = response.json()
    assert "triples" in data


def test_extract_batch(client):
    """Test batch extraction."""
    request = {
        "texts": [
            {"text": "Alice works at TechCorp.", "source_name": "doc1"},
            {"text": "Bob manages Alice.", "source_name": "doc2"}
        ],
        "ontology_scope": "minimal"
    }
    
    response = client.post("/api/extraction/extract/batch", json=request)
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert "total_triples" in data
    assert len(data["results"]) == 2


def test_extract_session(client):
    """Test session-based extraction."""
    # Create session with ontology
    ontology = {
        "entity_types": [{"name": "Person", "description": "A person", "attributes": []}],
        "relation_types": [{"name": "KNOWS", "description": "Knows", "domain": "Person", "range": "Person"}]
    }
    create_response = client.post("/api/sessions", json={})
    session_id = create_response.json()["session_id"]
    client.put(
        f"/api/sessions/{session_id}/ontology",
        json={"ontology": ontology}
    )
    
    # Extract with session
    request = {
        "text": "Alice knows Bob.",
        "source_name": "test-doc"
    }
    
    response = client.post(
        f"/api/extraction/session/{session_id}/extract",
        json=request
    )
    assert response.status_code == 200
    data = response.json()
    assert "triples" in data


# ============================================================================
# Ontology Tests
# ============================================================================


def test_recommend_ontology(client):
    """Test ontology recommendation."""
    request = {
        "text": "Alice works at TechCorp as an engineer in San Francisco.",
        "scope": "balanced"
    }
    
    response = client.post("/api/ontology/recommend", json=request)
    assert response.status_code == 200
    data = response.json()
    assert "ontology" in data
    assert "text_purpose" in data
    assert "reasoning" in data
    assert data["entity_type_count"] > 0
    assert data["relation_type_count"] > 0


def test_analyze_ontology_extension(client):
    """Test ontology extension analysis."""
    ontology = {
        "entity_types": [{"name": "Person", "description": "A person", "attributes": []}],
        "relation_types": [{"name": "KNOWS", "description": "Knows", "domain": "Person", "range": "Person"}]
    }
    
    request = {
        "text": "The hospital treated 50 patients yesterday.",
        "current_ontology": ontology,
        "scope": "balanced"
    }
    
    response = client.post("/api/ontology/extend/analyze", json=request)
    assert response.status_code == 200
    data = response.json()
    assert "needs_extension" in data
    assert "reasoning" in data


def test_apply_ontology_extension(client):
    """Test applying ontology extension."""
    ontology = {
        "entity_types": [{"name": "Person", "description": "A person", "attributes": []}],
        "relation_types": []
    }
    
    extension = {
        "needs_extension": True,
        "new_entity_types": [{"name": "Location", "description": "A place", "attributes": []}],
        "new_relation_types": [{"name": "LOCATED_IN", "description": "Located in", "domain": "Person", "range": "Location"}],
        "reasoning": "Need location types"
    }
    
    request = {
        "current_ontology": ontology,
        "extension": extension
    }
    
    response = client.post("/api/ontology/extend/apply", json=request)
    assert response.status_code == 200
    data = response.json()
    assert "extended_ontology" in data
    assert data["entity_type_count"] == 2
    assert data["relation_type_count"] == 1


def test_recommend_and_extract(client):
    """Test combined recommendation and extraction."""
    request = {
        "text": "Alice works at TechCorp.",
        "source_name": "test-doc",
        "scope": "minimal"
    }
    
    response = client.post("/api/ontology/recommend-and-extract", json=request)
    assert response.status_code == 200
    data = response.json()
    assert "ontology" in data
    assert "text_purpose" in data
    assert "triples" in data
    assert "triple_count" in data


# ============================================================================
# Process Extraction Tests
# ============================================================================


def test_extract_process(client):
    """Test process extraction."""
    request = {
        "text": "First, the customer submits an order. Then, the order is validated. Finally, it is shipped.",
        "process_hint": "order fulfillment"
    }
    
    response = client.post("/api/process/extract", json=request)
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "reasoning" in data
    assert "issues" in data


# ============================================================================
# Error Handling Tests
# ============================================================================


def test_invalid_session_id(client):
    """Test error handling for invalid session ID."""
    response = client.get("/api/sessions/invalid-id-12345")
    assert response.status_code == 404
    data = response.json()
    assert "code" in data
    assert "message" in data


def test_invalid_extraction_request(client):
    """Test error handling for invalid extraction request."""
    # Missing required fields
    response = client.post("/api/extraction/extract", json={})
    assert response.status_code == 422  # Validation error


def test_invalid_ontology_format(client):
    """Test error handling for invalid ontology format."""
    request = {
        "text": "Test text",
        "source_name": "test",
        "ontology": {"invalid": "format"}
    }
    
    response = client.post("/api/extraction/extract", json=request)
    assert response.status_code == 400
    data = response.json()
    assert "Invalid ontology format" in data["message"]


# ============================================================================
# Integration Tests
# ============================================================================


def test_full_session_workflow(client):
    """Test a complete session workflow."""
    # 1. Create session
    session_response = client.post(
        "/api/sessions",
        json={"name": "integration-test"}
    )
    assert session_response.status_code == 201
    session_id = session_response.json()["session_id"]
    
    # 2. Recommend ontology
    ontology_response = client.post(
        "/api/ontology/recommend",
        json={"text": "Alice works at TechCorp.", "scope": "minimal"}
    )
    assert ontology_response.status_code == 200
    ontology = ontology_response.json()["ontology"]
    
    # 3. Update session ontology
    update_response = client.put(
        f"/api/sessions/{session_id}/ontology",
        json={"ontology": ontology}
    )
    assert update_response.status_code == 200
    
    # 4. Extract triples
    extract_response = client.post(
        f"/api/extraction/session/{session_id}/extract",
        json={
            "text": "Alice works at TechCorp as an engineer.",
            "source_name": "doc1"
        }
    )
    assert extract_response.status_code == 200
    
    # 5. Verify session state
    session_info = client.get(f"/api/sessions/{session_id}").json()
    assert session_info["triple_count"] > 0
    assert session_info["ontology"] is not None
    
    # 6. Clean up
    delete_response = client.delete(f"/api/sessions/{session_id}")
    assert delete_response.status_code == 204

