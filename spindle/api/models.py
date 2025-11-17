"""Pydantic models for API request/response schemas."""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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
    catalog_url: Optional[str] = Field(None, description="Document catalog URL")
    config: Optional[Dict[str, Any]] = Field(None, description="Additional configuration")


class SessionInfo(BaseModel):
    """Session information."""

    session_id: str = Field(..., description="Unique session identifier")
    name: Optional[str] = Field(None, description="Session name")
    created_at: datetime = Field(..., description="Creation timestamp")
    graph_store_path: Optional[str] = Field(None, description="Graph store path")
    vector_store_uri: Optional[str] = Field(None, description="Vector store URI")
    catalog_url: Optional[str] = Field(None, description="Catalog URL")
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
# Ingestion Models
# ============================================================================


class IngestionRequest(BaseModel):
    """Request to ingest documents (stateless mode)."""

    file_paths: List[str] = Field(..., description="List of file paths to ingest")
    catalog_url: Optional[str] = Field(None, description="Document catalog URL")
    vector_store_uri: Optional[str] = Field(None, description="Vector store URI")
    template_paths: Optional[List[str]] = Field(None, description="Template search paths")
    cache_dir: Optional[str] = Field(None, description="Cache directory path")
    allow_network_requests: bool = Field(False, description="Allow network requests")


class IngestionSessionRequest(BaseModel):
    """Request to ingest documents into a session."""

    file_paths: List[str] = Field(..., description="List of file paths to ingest")


class DocumentInfo(BaseModel):
    """Information about an ingested document."""

    document_id: str
    source_path: str
    checksum: str
    loader_name: str
    template_name: str
    metadata: Dict[str, Any]
    created_at: datetime


class ChunkInfo(BaseModel):
    """Information about a document chunk."""

    chunk_id: str
    document_id: str
    text: str
    metadata: Dict[str, Any]


class IngestionMetrics(BaseModel):
    """Ingestion run metrics."""

    processed_documents: int = 0
    processed_chunks: int = 0
    bytes_read: int = 0
    errors: List[str] = Field(default_factory=list)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class IngestionResponse(BaseModel):
    """Response from document ingestion."""

    documents: List[DocumentInfo]
    chunks: List[ChunkInfo]
    metrics: IngestionMetrics
    message: str = Field("Ingestion completed successfully")


class IngestionStreamChunk(BaseModel):
    """Streaming chunk for ingestion progress."""

    event: str = Field(..., description="Event type (stage_start, stage_complete, document_complete, etc.)")
    data: Dict[str, Any] = Field(..., description="Event data")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# Extraction Models
# ============================================================================


class ExtractionRequest(BaseModel):
    """Request to extract triples from text."""

    text: str = Field(..., description="Text to extract triples from")
    source_name: str = Field(..., description="Source document name")
    source_url: Optional[str] = Field(None, description="Source document URL")
    ontology: Optional[Dict[str, Any]] = Field(None, description="Ontology definition (optional)")
    ontology_scope: Optional[str] = Field("balanced", description="Ontology scope if auto-recommending")
    existing_triples: Optional[List[Dict[str, Any]]] = Field(None, description="Existing triples for consistency")


class BatchExtractionRequest(BaseModel):
    """Request to extract triples from multiple texts."""

    texts: List[Dict[str, str]] = Field(
        ...,
        description="List of texts with keys: 'text', 'source_name', 'source_url' (optional)"
    )
    ontology: Optional[Dict[str, Any]] = Field(None, description="Ontology definition")
    ontology_scope: Optional[str] = Field("balanced", description="Ontology scope")
    existing_triples: Optional[List[Dict[str, Any]]] = Field(None, description="Existing triples")
    max_concurrent: int = Field(20, description="Max concurrent extractions")


class SessionExtractionRequest(BaseModel):
    """Request to extract triples within a session."""

    text: str = Field(..., description="Text to extract from")
    source_name: str = Field(..., description="Source name")
    source_url: Optional[str] = Field(None, description="Source URL")
    ontology_scope: Optional[str] = Field(None, description="Override ontology scope")


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
# Ontology Models
# ============================================================================


class OntologyScope(str, Enum):
    """Ontology scope levels."""

    MINIMAL = "minimal"
    BALANCED = "balanced"
    COMPREHENSIVE = "comprehensive"


class OntologyRecommendationRequest(BaseModel):
    """Request to recommend an ontology."""

    text: str = Field(..., description="Text to analyze")
    scope: OntologyScope = Field(OntologyScope.BALANCED, description="Desired ontology scope")


class OntologyRecommendationResponse(BaseModel):
    """Response from ontology recommendation."""

    ontology: Dict[str, Any]
    text_purpose: str
    reasoning: str
    entity_type_count: int
    relation_type_count: int


class OntologyExtensionAnalysisRequest(BaseModel):
    """Request to analyze ontology extension needs."""

    text: str = Field(..., description="New text to analyze")
    current_ontology: Dict[str, Any] = Field(..., description="Current ontology")
    scope: OntologyScope = Field(OntologyScope.BALANCED, description="Analysis scope")


class OntologyExtensionAnalysisResponse(BaseModel):
    """Response from ontology extension analysis."""

    needs_extension: bool
    new_entity_types: List[Dict[str, Any]]
    new_relation_types: List[Dict[str, Any]]
    critical_information_at_risk: Optional[str]
    reasoning: str


class OntologyExtensionApplyRequest(BaseModel):
    """Request to apply an ontology extension."""

    current_ontology: Dict[str, Any] = Field(..., description="Current ontology")
    extension: Dict[str, Any] = Field(..., description="Extension result from analysis")


class OntologyExtensionApplyResponse(BaseModel):
    """Response from applying ontology extension."""

    extended_ontology: Dict[str, Any]
    entity_type_count: int
    relation_type_count: int


class RecommendAndExtractRequest(BaseModel):
    """Request to recommend ontology and extract in one step."""

    text: str = Field(..., description="Text to analyze and extract from")
    source_name: str = Field(..., description="Source name")
    source_url: Optional[str] = Field(None, description="Source URL")
    scope: OntologyScope = Field(OntologyScope.BALANCED, description="Ontology scope")
    existing_triples: Optional[List[Dict[str, Any]]] = Field(None, description="Existing triples")


class RecommendAndExtractResponse(BaseModel):
    """Response from recommend and extract."""

    ontology: Dict[str, Any]
    text_purpose: str
    ontology_reasoning: str
    triples: List[Dict[str, Any]]
    extraction_reasoning: str
    triple_count: int


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


# ============================================================================
# Process Models
# ============================================================================


class ProcessExtractionRequest(BaseModel):
    """Request to extract a process graph."""

    text: str = Field(..., description="Text containing process description")
    process_hint: Optional[str] = Field(None, description="Hint about the process type")
    existing_graph: Optional[Dict[str, Any]] = Field(None, description="Existing process graph to extend")


class ProcessStepInfo(BaseModel):
    """Information about a process step."""

    step_id: str
    title: str
    summary: str
    step_type: str
    actors: List[str]
    inputs: List[str]
    outputs: List[str]
    duration: Optional[str]
    prerequisites: List[str]


class ProcessDependencyInfo(BaseModel):
    """Information about a process dependency."""

    from_step: str
    to_step: str
    relation: str
    condition: Optional[str]


class ProcessGraphInfo(BaseModel):
    """Information about a process graph."""

    process_name: Optional[str]
    scope: Optional[str]
    primary_goal: str
    start_step_ids: List[str]
    end_step_ids: List[str]
    steps: List[ProcessStepInfo]
    dependencies: List[ProcessDependencyInfo]
    notes: List[str]


class ProcessExtractionResponse(BaseModel):
    """Response from process extraction."""

    status: str
    graph: Optional[ProcessGraphInfo]
    reasoning: str
    issues: List[Dict[str, Any]]

