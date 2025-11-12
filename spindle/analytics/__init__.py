"""Analytics helpers for Spindle document ingestion."""

from spindle.analytics.recorder import IngestionAnalyticsEmitter
from spindle.analytics.store import AnalyticsStore
from spindle.analytics.views import (
    chunk_window_risk,
    corpus_overview,
    document_size_table,
    entity_resolution_metrics,
    observability_events,
    ontology_recommendation_metrics,
    triple_extraction_metrics,
)
from spindle.analytics.schema import (
    ChunkWindowSummary,
    ContextWindowAssessment,
    DocumentMetadata,
    DocumentObservation,
    OntologySignal,
    SemanticSegmentSummary,
    ServiceEventRecord,
    StructuralMetrics,
)

__all__ = [
    "AnalyticsStore",
    "ChunkWindowSummary",
    "ContextWindowAssessment",
    "DocumentMetadata",
    "DocumentObservation",
    "IngestionAnalyticsEmitter",
    "OntologySignal",
    "SemanticSegmentSummary",
    "ServiceEventRecord",
    "StructuralMetrics",
    "chunk_window_risk",
    "corpus_overview",
    "document_size_table",
    "entity_resolution_metrics",
    "observability_events",
    "ontology_recommendation_metrics",
    "triple_extraction_metrics",
]

