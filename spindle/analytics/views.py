"""Curated views over persisted ingestion analytics observations."""

from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any, Iterable

from spindle.analytics.schema import ChunkWindowSummary, DocumentObservation, RiskLevel
from spindle.analytics.store import AnalyticsStore


def _collect_observations(
    store: AnalyticsStore,
    *,
    limit: int | None = None,
) -> list[DocumentObservation]:
    return store.fetch_observations(limit=limit)


def corpus_overview(
    store: AnalyticsStore,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    """Return aggregate corpus-level statistics."""

    observations = _collect_observations(store, limit=limit)
    if not observations:
        return {
            "documents": 0,
            "avg_tokens": 0,
            "avg_chunks": 0,
            "total_tokens": 0,
            "context_strategy_counts": {},
            "risk_counts": {},
        }

    token_counts = [obs.structural.token_count for obs in observations]
    chunk_counts = [obs.structural.chunk_count for obs in observations]
    strategy_counts: Counter[str] = Counter()
    risk_counts: Counter[str] = Counter()

    for observation in observations:
        if observation.context:
            strategy_counts[observation.context.recommended_strategy.value] += 1
            risk_counts[observation.context.supporting_risk.value] += 1

    return {
        "documents": len(observations),
        "avg_tokens": mean(token_counts),
        "avg_chunks": mean(chunk_counts),
        "total_tokens": sum(token_counts),
        "context_strategy_counts": dict(strategy_counts),
        "risk_counts": dict(risk_counts),
    }


def document_size_table(
    store: AnalyticsStore,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Return per-document structural metrics suitable for tabular dashboards."""

    observations = _collect_observations(store, limit=limit)
    table: list[dict[str, Any]] = []
    for observation in observations:
        table.append(
            {
                "document_id": observation.metadata.document_id,
                "source_uri": observation.metadata.source_uri,
                "token_count": observation.structural.token_count,
                "chunk_count": observation.structural.chunk_count,
                "schema_version": observation.schema_version,
                "context_strategy": (
                    observation.context.recommended_strategy.value
                    if observation.context
                    else None
                ),
                "risk_level": (
                    observation.context.supporting_risk.value
                    if observation.context
                    else RiskLevel.MEDIUM.value
                ),
            }
        )
    return table


def chunk_window_risk(
    store: AnalyticsStore,
    *,
    window_size: int | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Materialize per-window token and risk statistics."""

    observations = _collect_observations(store, limit=limit)
    rows: list[dict[str, Any]] = []
    for observation in observations:
        for summary in observation.chunk_windows:
            if window_size and summary.window_size != window_size:
                continue
            rows.append(_window_summary_row(observation, summary))
    return rows


def _window_summary_row(
    observation: DocumentObservation,
    summary: ChunkWindowSummary,
) -> dict[str, Any]:
    return {
        "document_id": observation.metadata.document_id,
        "window_size": summary.window_size,
        "max_tokens": summary.token_summary.maximum,
        "median_tokens": summary.token_summary.median,
        "risk": summary.context_limit_risk.value,
        "cross_chunk_link_rate": summary.cross_chunk_link_rate,
    }


def observability_events(
    store: AnalyticsStore,
    *,
    document_id: str | None = None,
    limit: int | None = None,
) -> Iterable[dict[str, Any]]:
    """Expose associated service events for downstream visualization."""

    records = store.fetch_service_events(document_id=document_id, limit=limit)
    for record in records:
        yield {
            "timestamp": record.timestamp.isoformat(),
            "service": record.service,
            "name": record.name,
            "payload": record.payload,
        }


__all__ = [
    "corpus_overview",
    "document_size_table",
    "chunk_window_risk",
    "observability_events",
]

