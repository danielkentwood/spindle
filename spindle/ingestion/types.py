"""Shared ingestion dataclasses and type definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from spindle.configuration import SpindleConfig


Metadata = Dict[str, Any]


@dataclass(slots=True)
class TemplateSelector:
    """Criteria that determine whether a template applies to a document."""

    mime_types: Sequence[str] = field(default_factory=tuple)
    path_globs: Sequence[str] = field(default_factory=tuple)
    file_extensions: Sequence[str] = field(default_factory=tuple)


@dataclass(slots=True)
class TemplateSpec:
    """Configuration describing a document template pipeline."""

    name: str
    selector: TemplateSelector = field(default_factory=TemplateSelector)
    loader: str | Callable[..., Any] = ""
    preprocessors: Sequence[str | Callable[..., Any]] = field(default_factory=tuple)
    splitter: str | Mapping[str, Any] | Callable[..., Any] = ""
    metadata_extractors: Sequence[str | Callable[..., Any]] = field(default_factory=tuple)
    postprocessors: Sequence[str | Callable[..., Any]] = field(default_factory=tuple)
    graph_hooks: Sequence[str | Callable[..., Any]] = field(default_factory=tuple)
    description: str | None = None


@dataclass(slots=True)
class IngestionConfig:
    """User provided configuration for an ingestion run."""

    template_specs: Sequence[TemplateSpec]
    template_search_paths: Sequence[Path] = field(default_factory=tuple)
    catalog_url: Optional[str] = None
    vector_store_uri: Optional[str] = None
    cache_dir: Optional[Path] = None
    allow_network_requests: bool = False
    spindle_config: Optional[SpindleConfig] = None


@dataclass(slots=True)
class DocumentArtifact:
    """Represents a source document and its extracted metadata."""

    document_id: str
    source_path: Path
    checksum: str
    loader_name: str
    template_name: str
    metadata: Metadata = field(default_factory=dict)
    raw_bytes: Optional[bytes] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True)
class ChunkArtifact:
    """Represents a text chunk ready for vector storage."""

    chunk_id: str
    document_id: str
    text: str
    metadata: Metadata = field(default_factory=dict)
    embedding: Optional[Sequence[float]] = None


@dataclass(slots=True)
class DocumentGraphNode:
    """Node in the ingestion-time document graph."""

    node_id: str
    document_id: str
    label: str
    attributes: Metadata = field(default_factory=dict)


@dataclass(slots=True)
class DocumentGraphEdge:
    """Edge connecting chunks/documents/metadata within the graph."""

    edge_id: str
    source_id: str
    target_id: str
    relation: str
    attributes: Metadata = field(default_factory=dict)


@dataclass(slots=True)
class DocumentGraph:
    """Graph produced during ingestion for downstream reasoning."""

    nodes: List[DocumentGraphNode] = field(default_factory=list)
    edges: List[DocumentGraphEdge] = field(default_factory=list)

    def add_node(self, node: DocumentGraphNode) -> None:
        self.nodes.append(node)

    def add_edge(self, edge: DocumentGraphEdge) -> None:
        self.edges.append(edge)


@dataclass(slots=True)
class Corpus:
    """A collection of documents for ontology pipeline processing.

    A corpus groups related documents together and tracks the state of
    ontology pipeline stages (vocabulary, metadata, taxonomy, thesaurus,
    ontology, knowledge graph) that have been executed against it.
    """

    corpus_id: str
    name: str
    description: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    pipeline_state: Metadata = field(default_factory=dict)


@dataclass(slots=True)
class CorpusDocument:
    """Links a document to a corpus.

    References an existing DocumentArtifact by document_id, allowing
    the same ingested document to belong to multiple corpora.
    """

    corpus_id: str
    document_id: str
    added_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True)
class IngestionRunMetrics:
    """Aggregated metrics for an ingestion run."""

    started_at: datetime = field(default_factory=datetime.utcnow)
    finished_at: Optional[datetime] = None
    processed_documents: int = 0
    processed_chunks: int = 0
    bytes_read: int = 0
    errors: List[str] = field(default_factory=list)
    extra: Metadata = field(default_factory=dict)


@dataclass(slots=True)
class IngestionEvent:
    """Event emitted from the pipeline for observability hooks."""

    timestamp: datetime
    name: str
    payload: Metadata


@dataclass(slots=True)
class IngestionContext:
    """Context object passed between pipeline stages."""

    config: IngestionConfig
    active_template: Optional[TemplateSpec] = None
    run_metadata: Metadata = field(default_factory=dict)


@dataclass(slots=True)
class IngestionResult:
    """Final outcome of an ingestion run."""

    documents: List[DocumentArtifact]
    chunks: List[ChunkArtifact]
    document_graph: DocumentGraph
    metrics: IngestionRunMetrics
    events: Iterable[IngestionEvent] = field(default_factory=list)

