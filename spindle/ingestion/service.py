"""High-level service orchestration for ingestion runs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from spindle.configuration import SpindleConfig
from spindle.ingestion.observers import (
    PerformanceTracker,
    logging_observer,
    observability_observer,
)
from spindle.ingestion.pipeline import build_ingestion_pipeline
from spindle.ingestion.storage import create_storage_backends
from spindle.ingestion.templates import (
    DEFAULT_TEMPLATE_SPECS,
    TemplateRegistry,
    load_templates_from_paths,
    merge_template_sequences,
)
from spindle.ingestion.types import IngestionConfig, IngestionResult
from spindle.observability import get_event_recorder

RECORDER = get_event_recorder("ingestion.service")


def build_config(
    *,
    template_paths: Sequence[Path] | None = None,
    catalog_url: str | None = None,
    vector_store_uri: str | None = None,
    spindle_config: SpindleConfig | None = None,
) -> IngestionConfig:
    template_paths = tuple(template_paths or ())
    if spindle_config:
        default_template_paths = spindle_config.templates.search_paths
        if not template_paths and default_template_paths:
            template_paths = tuple(default_template_paths)
        if catalog_url is None:
            catalog_url = spindle_config.storage.catalog_url
        if vector_store_uri is None:
            vector_store_uri = str(spindle_config.storage.vector_store_dir)
    user_templates = load_templates_from_paths(template_paths)
    templates = merge_template_sequences(DEFAULT_TEMPLATE_SPECS, user_templates)
    return IngestionConfig(
        template_specs=templates,
        template_search_paths=template_paths,
        catalog_url=catalog_url,
        vector_store_uri=vector_store_uri,
        spindle_config=spindle_config,
    )


def run_ingestion(paths: Iterable[Path], config: IngestionConfig) -> IngestionResult:
    registry = TemplateRegistry(config.template_specs)
    catalog, vector = create_storage_backends(config)
    tracker = PerformanceTracker()
    resolved_paths = tuple(Path(p).resolve() for p in paths)
    RECORDER.record(
        name="run.start",
        payload={
            "path_count": len(resolved_paths),
            "catalog_url": config.catalog_url,
            "vector_store_uri": config.vector_store_uri,
        },
    )
    pipeline = build_ingestion_pipeline(
        config=config,
        registry=registry,
        observers=[logging_observer, tracker, observability_observer],
        document_catalog=catalog,
        vector_store=vector,
    )
    try:
        result = pipeline.ingest(resolved_paths)
    except Exception as exc:
        RECORDER.record(
            name="run.error",
            payload={
                "error": str(exc),
                "path_count": len(resolved_paths),
            },
        )
        raise
    result.metrics.extra.setdefault("stage_summary", tracker.summary())
    RECORDER.record(
        name="run.complete",
        payload={
            "path_count": len(resolved_paths),
            "processed_documents": result.metrics.processed_documents,
            "processed_chunks": result.metrics.processed_chunks,
            "errors": list(result.metrics.errors),
        },
    )
    return result

