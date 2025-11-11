"""Tests for ingestion analytics instrumentation and persistence."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from spindle.analytics.recorder import IngestionAnalyticsEmitter
from spindle.analytics.store import AnalyticsStore
from spindle.analytics.views import chunk_window_risk, corpus_overview, document_size_table
from spindle.ingestion.types import (
    ChunkArtifact,
    DocumentArtifact,
    DocumentGraph,
    IngestionEvent,
    IngestionResult,
    IngestionRunMetrics,
)


def _build_ingestion_result() -> IngestionResult:
    document = DocumentArtifact(
        document_id="doc-1",
        source_path=Path("/tmp/example.txt"),
        checksum="abc123",
        loader_name="loader",
        template_name="template",
        metadata={"content_type": "text/plain"},
        raw_bytes=None,
        created_at=datetime.utcnow(),
    )
    chunks = [
        ChunkArtifact(
            chunk_id="chunk-1",
            document_id=document.document_id,
            text="First chunk content for testing analytics.",
            metadata={},
        ),
        ChunkArtifact(
            chunk_id="chunk-2",
            document_id=document.document_id,
            text="Second chunk with additional content to increase token counts.",
            metadata={},
        ),
    ]
    metrics = IngestionRunMetrics()
    metrics.extra["stage_summary"] = {
        "split": {"duration_ms": 12.0, "count": 1, "avg_ms": 12.0},
        "chunks": {"duration_ms": 8.0, "count": 1, "avg_ms": 8.0},
    }
    events = [
        IngestionEvent(
            timestamp=datetime.utcnow(),
            name="graph_built",
            payload={
                "document_id": document.document_id,
                "chunk_count": len(chunks),
            },
        )
    ]
    return IngestionResult(
        documents=[document],
        chunks=chunks,
        document_graph=DocumentGraph(),
        metrics=metrics,
        events=events,
    )


def test_emitter_produces_observation(tmp_path) -> None:
    emitter = IngestionAnalyticsEmitter(window_sizes=(2,), token_budget=250)
    result = _build_ingestion_result()
    observations = emitter.emit_run(result)

    assert len(observations) == 1
    observation = observations[0]
    assert observation.metadata.document_id == "doc-1"
    assert observation.structural.chunk_count == 2
    assert observation.structural.token_count > 0
    assert observation.context is not None
    assert "analytics" in result.metrics.extra

    db_path = tmp_path / "analytics.db"
    store = AnalyticsStore(f"sqlite:///{db_path}")
    store.persist_observations(observations)

    fetched = store.fetch_observations()
    assert len(fetched) == 1
    assert fetched[0].metadata.document_id == "doc-1"

    events = store.fetch_service_events(document_id="doc-1")
    assert len(events) == 1
    assert events[0].name == "graph_built"


def test_views_provide_dashboard_rows(tmp_path) -> None:
    emitter = IngestionAnalyticsEmitter(window_sizes=(2,), token_budget=250)
    result = _build_ingestion_result()
    observations = emitter.emit_run(result)

    db_path = tmp_path / "analytics.db"
    store = AnalyticsStore(f"sqlite:///{db_path}")
    store.persist_observations(observations)

    overview = corpus_overview(store)
    assert overview["documents"] == 1
    assert overview["total_tokens"] == observations[0].structural.token_count

    table = document_size_table(store)
    assert table[0]["document_id"] == "doc-1"

    windows = chunk_window_risk(store)
    assert any(row["window_size"] == 2 for row in windows)

