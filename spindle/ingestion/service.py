"""High-level service orchestration for ingestion runs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from spindle.ingestion.observers import PerformanceTracker, logging_observer
from spindle.ingestion.pipeline import build_ingestion_pipeline
from spindle.ingestion.storage import create_storage_backends
from spindle.ingestion.templates import (
    DEFAULT_TEMPLATE_SPECS,
    TemplateRegistry,
    load_templates_from_paths,
    merge_template_sequences,
)
from spindle.ingestion.types import IngestionConfig, IngestionResult


def build_config(
    *,
    template_paths: Sequence[Path] | None = None,
    catalog_url: str | None = None,
    vector_store_uri: str | None = None,
) -> IngestionConfig:
    template_paths = tuple(template_paths or ())
    user_templates = load_templates_from_paths(template_paths)
    templates = merge_template_sequences(DEFAULT_TEMPLATE_SPECS, user_templates)
    return IngestionConfig(
        template_specs=templates,
        template_search_paths=template_paths,
        catalog_url=catalog_url,
        vector_store_uri=vector_store_uri,
    )


def run_ingestion(paths: Iterable[Path], config: IngestionConfig) -> IngestionResult:
    registry = TemplateRegistry(config.template_specs)
    catalog, vector = create_storage_backends(config)
    tracker = PerformanceTracker()
    pipeline = build_ingestion_pipeline(
        config=config,
        registry=registry,
        observers=[logging_observer, tracker],
        document_catalog=catalog,
        vector_store=vector,
    )
    result = pipeline.ingest(tuple(Path(p).resolve() for p in paths))
    result.metrics.extra.setdefault("stage_summary", tracker.summary())
    return result

