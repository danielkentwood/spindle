"""Ingestion subsystem public interface."""

from .pipeline import LangChainIngestionPipeline, build_ingestion_pipeline
from .templates import (
    DEFAULT_TEMPLATE_SPECS,
    TemplateRegistry,
    load_templates_from_paths,
    merge_template_sequences,
)
from .types import (
    ChunkArtifact,
    DocumentArtifact,
    DocumentGraph,
    DocumentGraphEdge,
    DocumentGraphNode,
    IngestionConfig,
    IngestionContext,
    IngestionEvent,
    IngestionResult,
    IngestionRunMetrics,
    TemplateSpec,
)

__all__ = [
    "LangChainIngestionPipeline",
    "build_ingestion_pipeline",
    "TemplateRegistry",
    "DEFAULT_TEMPLATE_SPECS",
    "load_templates_from_paths",
    "merge_template_sequences",
    "ChunkArtifact",
    "DocumentArtifact",
    "DocumentGraph",
    "DocumentGraphEdge",
    "DocumentGraphNode",
    "IngestionConfig",
    "IngestionContext",
    "IngestionEvent",
    "IngestionResult",
    "IngestionRunMetrics",
    "TemplateSpec",
]

