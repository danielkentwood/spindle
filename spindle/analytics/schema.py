"""Structured models describing ingestion analytics observations."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from pydantic import BaseModel, Field, computed_field

from spindle.observability.events import ServiceEvent


class RiskLevel(str, Enum):
    """Qualitative assessment for risk-oriented metrics."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ContextStrategy(str, Enum):
    """Recommended context granularity for downstream processing."""

    CHUNK = "chunk"
    WINDOW = "window"
    SEGMENT = "segment"
    DOCUMENT = "document"


class SourceType(str, Enum):
    """Normalized source origins for documents."""

    FILE = "file"
    URL = "url"
    API = "api"
    STREAM = "stream"
    OTHER = "other"


class QuantileSummary(BaseModel):
    """Simple descriptive statistics for numeric distributions."""

    minimum: float
    maximum: float
    median: float | None = None
    mean: float | None = None
    p95: float | None = Field(default=None, description="95th percentile value.")


class DocumentMetadata(BaseModel):
    """Core identifiers describing the ingested document."""

    document_id: str
    source_uri: str | None = None
    source_type: SourceType = SourceType.OTHER
    content_type: str | None = None
    language: str | None = Field(
        default=None, description="ISO-639 language code when available."
    )
    ingested_at: datetime
    hash_signature: str | None = None


class StructuralMetrics(BaseModel):
    """Aggregate structural data gathered during ingestion."""

    token_count: int
    character_count: int | None = None
    page_count: int | None = None
    section_count: int | None = None
    average_tokens_per_section: float | None = None
    chunk_count: int
    chunk_token_summary: QuantileSummary


class ChunkWindowSummary(BaseModel):
    """Statistics computed over sliding windows of chunks."""

    window_size: int = Field(gt=0)
    token_summary: QuantileSummary
    overlap_tokens: int | None = None
    overlap_ratio: float | None = Field(
        default=None, description="Overlap as a decimal percentage (0-1)."
    )
    cross_chunk_link_rate: float | None = Field(
        default=None,
        description="Fraction of extracted triples referencing more than one chunk within the window.",
    )
    context_limit_risk: RiskLevel = RiskLevel.MEDIUM


class SemanticSegmentSummary(BaseModel):
    """Information about larger semantic groupings within a document."""

    segment_boundaries: list[int] = Field(
        default_factory=list, description="Token offsets marking segment starts."
    )
    segment_token_summary: QuantileSummary | None = None
    embedding_dispersion: float | None = Field(
        default=None,
        description="Aggregate variance or dispersion metric across chunk embeddings.",
    )
    topic_transition_score: float | None = Field(
        default=None,
        description="Normalized score describing topic drift between adjacent segments.",
    )


class OntologySignal(BaseModel):
    """Signals feeding ontology recommendation and extension."""

    ontology_candidate_terms: list[str] = Field(default_factory=list)
    coverage_estimate: float | None = Field(
        default=None,
        description="Estimated proportion of ontology classes covered by the document.",
    )
    graph_density_estimate: float | None = Field(
        default=None,
        description="Approximate density (edges per node) predicted from the document.",
    )


class ContextWindowAssessment(BaseModel):
    """Assessment for choosing the downstream processing window size."""

    recommended_strategy: ContextStrategy = ContextStrategy.WINDOW
    supporting_risk: RiskLevel = RiskLevel.MEDIUM
    estimated_token_usage: int | None = Field(
        default=None,
        description="Estimated tokens required when using the recommended strategy.",
    )
    target_token_budget: int | None = None


class ServiceEventRecord(BaseModel):
    """Pydantic wrapper around `ServiceEvent` for persistence compatibility."""

    timestamp: datetime
    service: str
    name: str
    payload: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_service_event(cls, event: ServiceEvent) -> "ServiceEventRecord":
        """Create a record from a native `ServiceEvent`."""

        return cls(
            timestamp=event.timestamp,
            service=event.service,
            name=event.name,
            payload=dict(event.payload),
        )

    def to_service_event(self) -> ServiceEvent:
        """Convert the record back into a `ServiceEvent`."""

        return ServiceEvent(
            timestamp=self.timestamp,
            service=self.service,
            name=self.name,
            payload=dict(self.payload),
        )


class ObservabilitySignals(BaseModel):
    """Observability-aligned analytics payload."""

    service_events: list[ServiceEventRecord] = Field(default_factory=list)
    error_signals: list[str] = Field(default_factory=list)
    latency_breakdown: Mapping[str, float] = Field(default_factory=dict)

    @computed_field  # type: ignore[misc]
    def has_errors(self) -> bool:
        """Whether any error signals were captured."""

        return bool(self.error_signals)


class DocumentObservation(BaseModel):
    """Top-level ingestion analytics record."""

    schema_version: str = "1.0.0"
    metadata: DocumentMetadata
    structural: StructuralMetrics
    chunk_windows: list[ChunkWindowSummary] = Field(default_factory=list)
    segments: SemanticSegmentSummary | None = None
    ontology: OntologySignal | None = None
    context: ContextWindowAssessment | None = None
    observability: ObservabilitySignals = Field(default_factory=ObservabilitySignals)

    def extend_service_events(
        self,
        events: Sequence[ServiceEvent] | Iterable[ServiceEvent],
    ) -> None:
        """Append additional service events to the observability payload."""

        self.observability.service_events.extend(
            ServiceEventRecord.from_service_event(event) for event in events
        )


__all__ = [
    "ChunkWindowSummary",
    "ContextStrategy",
    "ContextWindowAssessment",
    "DocumentMetadata",
    "DocumentObservation",
    "OntologySignal",
    "ObservabilitySignals",
    "QuantileSummary",
    "RiskLevel",
    "SemanticSegmentSummary",
    "ServiceEventRecord",
    "SourceType",
    "StructuralMetrics",
]

