"""High-level service orchestration for ingestion runs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from spindle.analytics import IngestionAnalyticsEmitter
from spindle.analytics.store import AnalyticsStore
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
ANALYTICS_EMITTER = IngestionAnalyticsEmitter()


def build_config(
    *,
    template_paths: Sequence[Path] | None = None,
    catalog_url: str | None = None,
    vector_store_uri: str | None = None,
    cache_dir: Path | None = None,
    allow_network_requests: bool | None = None,
    spindle_config: SpindleConfig | None = None,
) -> IngestionConfig:
    template_paths = tuple(template_paths or ())
    cache_dir_path: Path | None = cache_dir
    allow_network = allow_network_requests if allow_network_requests is not None else False
    if spindle_config:
        default_template_paths = spindle_config.templates.search_paths
        if not template_paths and default_template_paths:
            template_paths = tuple(default_template_paths)
        if catalog_url is None:
            catalog_url = (
                spindle_config.ingestion.catalog_url
                or spindle_config.storage.catalog_url
            )
        if vector_store_uri is None:
            if spindle_config.ingestion.vector_store_uri:
                vector_store_uri = spindle_config.ingestion.vector_store_uri
            else:
                vector_store_uri = str(spindle_config.storage.vector_store_dir)
        if cache_dir_path is None:
            cache_dir_path = spindle_config.ingestion.cache_dir
        if allow_network_requests is None:
            allow_network = spindle_config.ingestion.allow_network_requests
    user_templates = load_templates_from_paths(template_paths)
    templates = merge_template_sequences(DEFAULT_TEMPLATE_SPECS, user_templates)
    return IngestionConfig(
        template_specs=templates,
        template_search_paths=template_paths,
        catalog_url=catalog_url,
        vector_store_uri=vector_store_uri,
        cache_dir=cache_dir_path,
        allow_network_requests=allow_network,
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
        observations = ANALYTICS_EMITTER.emit_run(result)
        store = _resolve_analytics_store(config)
        if store and observations:
            store.persist_observations(observations)
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


def _resolve_analytics_store(config: IngestionConfig) -> AnalyticsStore | None:
    spindle_config = config.spindle_config
    if spindle_config is None:
        return None

    extras = dict(spindle_config.extras)
    database_url = extras.get("analytics_database")
    if not database_url:
        analytics_path = spindle_config.storage.log_dir / "analytics.db"
        analytics_path.parent.mkdir(parents=True, exist_ok=True)
        database_url = f"sqlite:///{analytics_path}"

    try:
        return AnalyticsStore(database_url)
    except Exception:  # pragma: no cover - defensive guard for invalid configs
        return None

