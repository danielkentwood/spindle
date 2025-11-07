"""LangChain-backed ingestion pipeline implementation."""

from __future__ import annotations

import hashlib
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, List, Sequence

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from spindle.ingestion.errors import PipelineExecutionError
from spindle.ingestion.graph import DocumentGraphBuilder
from spindle.ingestion.templates import TemplateRegistry
from spindle.ingestion.types import (
    ChunkArtifact,
    DocumentArtifact,
    DocumentGraph,
    IngestionConfig,
    IngestionContext,
    IngestionEvent,
    IngestionResult,
    IngestionRunMetrics,
    TemplateSpec,
)
from spindle.ingestion.utils import ensure_callable, import_from_string

if TYPE_CHECKING:
    from spindle.ingestion.storage import ChromaVectorStoreAdapter, DocumentCatalog


StageCallable = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(slots=True)
class PipelineStage:
    name: str
    func: StageCallable


class LangChainIngestionPipeline:
    """Execute document ingestion using LangChain runnables."""

    def __init__(
        self,
        config: IngestionConfig,
        registry: TemplateRegistry,
        observers: Sequence[Callable[[IngestionEvent], None]] | None = None,
        document_catalog: "DocumentCatalog" | None = None,
        vector_store: "ChromaVectorStoreAdapter" | None = None,
    ) -> None:
        self._config = config
        self._registry = registry
        self._observers = list(observers or [])
        self._catalog = document_catalog
        self._vector_store = vector_store

    def ingest(self, paths: Sequence[Path]) -> IngestionResult:
        metrics = IngestionRunMetrics(started_at=datetime.utcnow())
        documents: List[DocumentArtifact] = []
        chunks: List[ChunkArtifact] = []
        events: list[IngestionEvent] = []
        graph_builder = DocumentGraphBuilder()
        run_id = uuid.uuid4().hex
        self._stage_totals: dict[str, float] = defaultdict(float)
        self._stage_counts: dict[str, int] = defaultdict(int)

        for path in paths:
            try:
                template = self._registry.resolve(path=path)
            except Exception as exc:  # pragma: no cover - failures recorded
                metrics.errors.append(str(exc))
                continue
            context = IngestionContext(config=self._config, active_template=template)
            payload = {
                "path": path,
                "template": template,
                "context": context,
                "graph_builder": graph_builder,
            }
            sequence = self._build_sequence(template)
            try:
                result_payload = sequence.invoke(payload)
            except Exception as exc:
                metrics.errors.append(str(exc))
                raise PipelineExecutionError(str(exc)) from exc

            artifact = result_payload["document_artifact"]
            chunk_artifacts = result_payload["chunk_artifacts"]
            documents.append(artifact)
            chunks.extend(chunk_artifacts)
            metrics.processed_documents += 1
            metrics.processed_chunks += len(chunk_artifacts)
            metrics.bytes_read += result_payload.get("bytes_read", 0)
            for event in result_payload.get("events", []):
                events.append(event)
                self._emit_event(event)

        metrics.finished_at = datetime.utcnow()
        metrics.extra["run_id"] = run_id
        metrics.extra["stage_durations_ms"] = dict(self._stage_totals)
        metrics.extra["stage_calls"] = dict(self._stage_counts)
        result = IngestionResult(
            documents=documents,
            chunks=chunks,
            document_graph=graph_builder.graph,
            metrics=metrics,
            events=events,
        )
        if self._catalog:
            self._catalog.persist_result(result, run_id=run_id)
        if self._vector_store:
            self._vector_store.upsert_chunks(result.chunks)
        return result

    def _build_sequence(self, template: TemplateSpec) -> RunnableLambda:
        stages = [
            PipelineStage("checksum", self._compute_checksum),
            PipelineStage("load", self._load_documents),
            PipelineStage("preprocess", self._apply_preprocessors),
            PipelineStage("split", self._split_documents),
            PipelineStage("metadata", self._enrich_metadata),
            PipelineStage("chunks", self._create_chunk_artifacts),
            PipelineStage("graph", self._build_graph),
        ]

        runnable: RunnableLambda | None = None
        for stage in stages:
            stage_runnable = RunnableLambda(self._wrap_stage(stage))
            runnable = stage_runnable if runnable is None else runnable | stage_runnable
        if runnable is None:  # pragma: no cover - defensive
            raise PipelineExecutionError("No stages configured for ingestion pipeline")
        return runnable

    @staticmethod
    def _compute_checksum(payload: dict[str, Any]) -> dict[str, Any]:
        path: Path = payload["path"]
        digest = hashlib.sha256()
        raw_bytes = path.read_bytes()
        digest.update(raw_bytes)
        payload["checksum"] = digest.hexdigest()
        payload["bytes_read"] = len(raw_bytes)
        payload["raw_bytes"] = raw_bytes
        return payload

    @staticmethod
    def _load_documents(payload: dict[str, Any]) -> dict[str, Any]:
        spec: TemplateSpec = payload["template"]
        loader_cls = ensure_callable(spec.loader)
        loader = loader_cls(str(payload["path"]))
        documents: list[Document] | Any = loader.load()
        if not isinstance(documents, list):
            documents = list(documents)
        payload["documents"] = documents
        return payload

    @staticmethod
    def _apply_preprocessors(payload: dict[str, Any]) -> dict[str, Any]:
        spec: TemplateSpec = payload["template"]
        documents: list[Document] = payload.get("documents", [])
        for preprocessor in spec.preprocessors:
            func = ensure_callable(preprocessor)
            documents = func(documents)
        payload["documents"] = documents
        return payload

    @staticmethod
    def _split_documents(payload: dict[str, Any]) -> dict[str, Any]:
        spec: TemplateSpec = payload["template"]
        splitter_config = spec.splitter
        if isinstance(splitter_config, dict):
            path = splitter_config.get("name")
            params = splitter_config.get("params", {})
            splitter_cls = import_from_string(path)
            splitter = splitter_cls(**params)
        else:
            splitter_cls = ensure_callable(splitter_config)
            splitter = splitter_cls()
        documents: list[Document] = payload.get("documents", [])
        chunks = splitter.split_documents(documents)
        payload["chunks"] = chunks
        return payload

    @staticmethod
    def _enrich_metadata(payload: dict[str, Any]) -> dict[str, Any]:
        spec: TemplateSpec = payload["template"]
        documents: list[Document] = payload.get("documents", [])
        metadata = payload.setdefault("metadata", {})
        for extractor in spec.metadata_extractors:
            func = ensure_callable(extractor)
            extracted = func(documents)
            if isinstance(extracted, dict):
                metadata.update(extracted)
        payload["metadata"] = metadata
        return payload

    @staticmethod
    def _create_chunk_artifacts(payload: dict[str, Any]) -> dict[str, Any]:
        path: Path = payload["path"]
        spec: TemplateSpec = payload["template"]
        documents: list[Document] = payload.get("documents", [])
        metadata = payload.get("metadata", {}).copy()

        document_id = uuid.uuid4().hex
        artifact = DocumentArtifact(
            document_id=document_id,
            source_path=path,
            checksum=payload.get("checksum", ""),
            loader_name=str(spec.loader),
            template_name=spec.name,
            metadata=metadata,
            raw_bytes=payload.get("raw_bytes"),
        )

        chunk_artifacts: list[ChunkArtifact] = []
        for chunk in payload.get("chunks", []):
            chunk_id = uuid.uuid4().hex
            chunk_metadata = dict(chunk.metadata)
            chunk_metadata.setdefault("document_id", document_id)
            chunk_artifacts.append(
                ChunkArtifact(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    text=chunk.page_content,
                    metadata=chunk_metadata,
                )
            )

        payload["document_artifact"] = artifact
        payload["chunk_artifacts"] = chunk_artifacts
        payload["documents"] = documents
        return payload

    @staticmethod
    def _build_graph(payload: dict[str, Any]) -> dict[str, Any]:
        builder: DocumentGraphBuilder = payload["graph_builder"]
        artifact: DocumentArtifact = payload["document_artifact"]
        chunks: list[ChunkArtifact] = payload.get("chunk_artifacts", [])
        spec: TemplateSpec = payload["template"]
        document_node = builder.add_document(artifact)
        chunk_nodes = builder.add_chunks(document_node, chunks)
        for hook in spec.graph_hooks:
            func = ensure_callable(hook)
            func(
                document_artifact=artifact,
                chunk_artifacts=chunks,
                document_node=document_node,
                chunk_nodes=chunk_nodes,
                builder=builder,
            )
        payload.setdefault("events", []).append(
            IngestionEvent(
                timestamp=datetime.utcnow(),
                name="graph_built",
                payload={
                    "document_id": artifact.document_id,
                    "chunk_count": len(chunks),
                },
            )
        )
        return payload

    def _emit_event(self, event: IngestionEvent) -> None:
        for observer in self._observers:
            observer(event)

    def _wrap_stage(self, stage: PipelineStage) -> StageCallable:
        def _wrapped(payload: dict[str, Any]) -> dict[str, Any]:
            start = time.perf_counter()
            start_event = IngestionEvent(
                timestamp=datetime.utcnow(),
                name="stage_start",
                payload={"stage": stage.name},
            )
            payload.setdefault("events", []).append(start_event)
            self._emit_event(start_event)

            result = stage.func(payload)

            duration_ms = (time.perf_counter() - start) * 1000
            finish_event = IngestionEvent(
                timestamp=datetime.utcnow(),
                name="stage_complete",
                payload={
                    "stage": stage.name,
                    "duration_ms": duration_ms,
                },
            )
            result.setdefault("events", []).append(finish_event)
            self._emit_event(finish_event)
            self._stage_totals[stage.name] += duration_ms
            self._stage_counts[stage.name] += 1
            return result

        return _wrapped


def build_ingestion_pipeline(
    config: IngestionConfig,
    registry: TemplateRegistry,
    observers: Sequence[Callable[[IngestionEvent], None]] | None = None,
    document_catalog: "DocumentCatalog" | None = None,
    vector_store: "ChromaVectorStoreAdapter" | None = None,
) -> LangChainIngestionPipeline:
    return LangChainIngestionPipeline(
        config=config,
        registry=registry,
        observers=observers,
        document_catalog=document_catalog,
        vector_store=vector_store,
    )

