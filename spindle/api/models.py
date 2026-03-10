"""Pydantic models for API request/response schemas."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ============================================================================
# Common Models
# ============================================================================


class ErrorResponse(BaseModel):
    """Standard error response."""

    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Check timestamp")


# ============================================================================
# Session Models
# ============================================================================


class SessionCreate(BaseModel):
    """Request to create a new session."""

    name: Optional[str] = Field(None, description="Optional session name")
    graph_store_path: Optional[str] = Field(None, description="Path to graph store database")
    vector_store_uri: Optional[str] = Field(None, description="Vector store URI or path")
    config: Optional[Dict[str, Any]] = Field(None, description="Additional configuration")


class SessionInfo(BaseModel):
    """Session information."""

    session_id: str = Field(..., description="Unique session identifier")
    name: Optional[str] = Field(None, description="Session name")
    created_at: datetime = Field(..., description="Creation timestamp")
    graph_store_path: Optional[str] = Field(None, description="Graph store path")
    vector_store_uri: Optional[str] = Field(None, description="Vector store URI")
    ontology: Optional[Dict[str, Any]] = Field(None, description="Session ontology")
    triple_count: int = Field(0, description="Number of triples in session")
    config: Dict[str, Any] = Field(default_factory=dict, description="Session config")


class SessionUpdate(BaseModel):
    """Update session configuration."""

    name: Optional[str] = Field(None, description="New session name")
    config: Optional[Dict[str, Any]] = Field(None, description="Updated configuration")


class OntologyUpdate(BaseModel):
    """Update session ontology."""

    ontology: Dict[str, Any] = Field(..., description="New ontology definition")


# ============================================================================
# Extraction Models
# ============================================================================


class ExtractionRequest(BaseModel):
    """Request to extract triples from text."""

    text: str = Field(..., description="Text to extract triples from")
    source_name: str = Field(..., description="Source document name")
    source_url: Optional[str] = Field(None, description="Source document URL")
    ontology: Dict[str, Any] = Field(..., description="Ontology definition (required)")
    existing_triples: Optional[List[Dict[str, Any]]] = Field(None, description="Existing triples for consistency")


class BatchExtractionRequest(BaseModel):
    """Request to extract triples from multiple texts."""

    texts: List[Dict[str, str]] = Field(
        ...,
        description="List of texts with keys: 'text', 'source_name', 'source_url' (optional)"
    )
    ontology: Dict[str, Any] = Field(..., description="Ontology definition (required)")
    existing_triples: Optional[List[Dict[str, Any]]] = Field(None, description="Existing triples")
    max_concurrent: int = Field(20, description="Max concurrent extractions")


class SessionExtractionRequest(BaseModel):
    """Request to extract triples within a session."""

    text: str = Field(..., description="Text to extract from")
    source_name: str = Field(..., description="Source name")
    source_url: Optional[str] = Field(None, description="Source URL")


class ExtractionResponse(BaseModel):
    """Response from triple extraction."""

    triples: List[Dict[str, Any]]
    reasoning: str
    source_name: str
    triple_count: int


class BatchExtractionResponse(BaseModel):
    """Response from batch extraction."""

    results: List[ExtractionResponse]
    total_triples: int


# ============================================================================
# Resolution Models
# ============================================================================


class ResolutionRequest(BaseModel):
    """Request to run entity resolution (stateless mode)."""

    graph_store_path: str = Field(..., description="Path to graph store database")
    vector_store_uri: str = Field(..., description="Vector store URI")
    apply_to_nodes: bool = Field(True, description="Resolve node duplicates")
    apply_to_edges: bool = Field(True, description="Resolve edge duplicates")
    context: str = Field("", description="Domain/ontology context")
    config: Optional[Dict[str, Any]] = Field(None, description="Resolution config overrides")


class ResolutionSessionRequest(BaseModel):
    """Request to run resolution within a session."""

    apply_to_nodes: bool = Field(True, description="Resolve node duplicates")
    apply_to_edges: bool = Field(True, description="Resolve edge duplicates")
    context: str = Field("", description="Domain/ontology context")
    config: Optional[Dict[str, Any]] = Field(None, description="Resolution config overrides")


class ResolutionResponse(BaseModel):
    """Response from entity resolution."""

    total_nodes_processed: int
    total_edges_processed: int
    blocks_created: int
    node_match_count: int
    edge_match_count: int
    same_as_edges_created: int
    duplicate_clusters: int
    execution_time_seconds: float
    config: Dict[str, Any]
