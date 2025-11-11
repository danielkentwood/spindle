"""Emit ingestion analytics observations into the observability pipeline."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from spindle.analytics.schema import (
    ChunkWindowSummary,
    ContextStrategy,
    ContextWindowAssessment,
    DocumentMetadata,
    DocumentObservation,
    ObservabilitySignals,
    QuantileSummary,
    RiskLevel,
    SemanticSegmentSummary,
    ServiceEventRecord,
    StructuralMetrics,
    SourceType,
)
from spindle.ingestion.types import (
    ChunkArtifact,
    DocumentArtifact,
    IngestionEvent,
    IngestionResult,
)
from spindle.observability import get_event_recorder
from spindle.observability.events import ServiceEvent


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _quantile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    sorted_values = sorted(values)
    idx = (len(sorted_values) - 1) * fraction
    lower_index = math.floor(idx)
    upper_index = math.ceil(idx)
    lower = sorted_values[lower_index]
    upper = sorted_values[upper_index]
    if lower_index == upper_index:
        return float(lower)
    weight = idx - lower_index
    return float(lower + (upper - lower) * weight)


def _build_quantile_summary(values: Sequence[int]) -> QuantileSummary:
    float_values = [float(value) for value in values]
    return QuantileSummary(
        minimum=float(min(values)) if values else 0.0,
        maximum=float(max(values)) if values else 0.0,
        median=_quantile(float_values, 0.5),
        mean=_mean(float_values),
        p95=_quantile(float_values, 0.95),
    )


def _window_token_counts(tokens: Sequence[int], window_size: int) -> list[int]:
    if window_size <= 0 or not tokens:
        return []
    if len(tokens) < window_size:
        return []
    return [
        sum(tokens[idx : idx + window_size])
        for idx in range(0, len(tokens) - window_size + 1)
    ]


def _count_tokens(text: str) -> int:
    return len(text.split())


def _infer_source_type(path: Path) -> SourceType:
    if str(path).startswith(("http://", "https://")):
        return SourceType.URL
    if str(path).startswith(("s3://", "gs://")):
        return SourceType.API
    return SourceType.FILE


def _determine_risk_level(total_tokens: int, token_budget: int) -> RiskLevel:
    if token_budget <= 0:
        return RiskLevel.MEDIUM
    usage_ratio = total_tokens / token_budget
    if usage_ratio <= 0.6:
        return RiskLevel.LOW
    if usage_ratio <= 1.1:
        return RiskLevel.MEDIUM
    return RiskLevel.HIGH


def _determine_context_strategy(
    total_tokens: int,
    max_window_tokens: int | None,
    *,
    token_budget: int,
) -> ContextStrategy:
    if total_tokens <= token_budget:
        return ContextStrategy.DOCUMENT
    if max_window_tokens is not None and max_window_tokens <= token_budget:
        return ContextStrategy.WINDOW
    return ContextStrategy.SEGMENT


def _group_chunks_by_document(
    chunks: Sequence[ChunkArtifact],
) -> dict[str, list[ChunkArtifact]]:
    grouped: dict[str, list[ChunkArtifact]] = defaultdict(list)
    for chunk in chunks:
        grouped[chunk.document_id].append(chunk)
    return grouped


def _group_events_by_document(
    events: Iterable[IngestionEvent],
) -> dict[str, list[IngestionEvent]]:
    grouped: dict[str, list[IngestionEvent]] = defaultdict(list)
    for event in events:
        document_id = event.payload.get("document_id")
        if document_id:
            grouped[document_id].append(event)
    return grouped


class IngestionAnalyticsEmitter:
    """Compute analytics for ingestion results and emit observability events."""

    def __init__(
        self,
        *,
        service: str = "ingestion.analytics",
        window_sizes: Sequence[int] = (2, 3, 5),
        token_budget: int = 12000,
    ) -> None:
        self._recorder = get_event_recorder(service)
        self._window_sizes = tuple(sorted(set(window_sizes)))
        self._token_budget = token_budget

    def emit_run(self, result: IngestionResult) -> list[DocumentObservation]:
        """Emit analytics observations for a completed ingestion run."""

        chunks_by_document = _group_chunks_by_document(result.chunks)
        events_by_document = _group_events_by_document(result.events)
        stage_summary = result.metrics.extra.get("stage_summary", {})
        observations: list[DocumentObservation] = []

        for document in result.documents:
            chunk_group = chunks_by_document.get(document.document_id, [])
            event_group = events_by_document.get(document.document_id, [])
            observation = self._build_observation(
                document=document,
                chunks=chunk_group,
                stage_summary=stage_summary,
                run_errors=list(result.metrics.errors),
                events=event_group,
            )
            self._recorder.record(
                name="document.observed",
                timestamp=datetime.utcnow(),
                payload={
                    "document_id": document.document_id,
                    "observation": observation.model_dump(mode="json"),
                },
            )
            observations.append(observation)

        result.metrics.extra.setdefault("analytics", {})
        result.metrics.extra["analytics"]["documents"] = [
            observation.model_dump(mode="json") for observation in observations
        ]
        result.metrics.extra["analytics"]["schema_version"] = (
            observations[0].schema_version if observations else "1.0.0"
        )
        return observations

    def _build_observation(
        self,
        *,
        document: DocumentArtifact,
        chunks: Sequence[ChunkArtifact],
        stage_summary: Mapping[str, Mapping[str, float]] | None,
        run_errors: list[str],
        events: Sequence[IngestionEvent],
    ) -> DocumentObservation:
        tokens_by_chunk = [max(_count_tokens(chunk.text), 1) for chunk in chunks]
        total_tokens = sum(tokens_by_chunk)
        section_count = document.metadata.get("section_count")
        average_tokens_per_section = (
            total_tokens / section_count if section_count else None
        )
        structural = StructuralMetrics(
            token_count=total_tokens,
            character_count=sum(len(chunk.text) for chunk in chunks) or None,
            page_count=document.metadata.get("page_count"),
            section_count=document.metadata.get("section_count"),
            average_tokens_per_section=average_tokens_per_section,
            chunk_count=len(chunks),
            chunk_token_summary=_build_quantile_summary(tokens_by_chunk or [0]),
        )

        chunk_windows = self._build_chunk_windows(tokens_by_chunk)
        context_assessment = self._build_context_assessment(
            total_tokens=total_tokens,
            chunk_windows=chunk_windows,
        )

        observation = DocumentObservation(
            metadata=DocumentMetadata(
                document_id=document.document_id,
                source_uri=str(document.source_path),
                source_type=_infer_source_type(document.source_path),
                content_type=document.metadata.get("content_type"),
                language=document.metadata.get("language"),
                ingested_at=document.created_at,
                hash_signature=document.checksum,
            ),
            structural=structural,
            chunk_windows=chunk_windows,
            segments=SemanticSegmentSummary(segment_boundaries=[]),
            ontology=None,
            context=context_assessment,
            observability=self._build_observability_signals(
                stage_summary=stage_summary,
                errors=run_errors,
                events=events,
            ),
        )

        return observation

    def _build_chunk_windows(
        self,
        tokens_by_chunk: Sequence[int],
    ) -> list[ChunkWindowSummary]:
        summaries: list[ChunkWindowSummary] = []
        for window_size in self._window_sizes:
            window_tokens = _window_token_counts(tokens_by_chunk, window_size)
            if not window_tokens:
                continue
            token_summary = _build_quantile_summary(window_tokens)
            summaries.append(
                ChunkWindowSummary(
                    window_size=window_size,
                    token_summary=token_summary,
                    overlap_tokens=None,
                    overlap_ratio=None,
                    cross_chunk_link_rate=0.0,
                    context_limit_risk=_determine_risk_level(
                        int(token_summary.maximum),
                        self._token_budget,
                    ),
                )
            )
        return summaries

    def _build_context_assessment(
        self,
        *,
        total_tokens: int,
        chunk_windows: Sequence[ChunkWindowSummary],
    ) -> ContextWindowAssessment:
        max_window_tokens = None
        if chunk_windows:
            max_window_tokens = max(
                int(summary.token_summary.maximum or 0) for summary in chunk_windows
            )
        recommended = _determine_context_strategy(
            total_tokens,
            max_window_tokens,
            token_budget=self._token_budget,
        )
        risk = _determine_risk_level(total_tokens, self._token_budget)
        estimated_usage = (
            total_tokens if recommended is ContextStrategy.DOCUMENT else max_window_tokens
        )

        return ContextWindowAssessment(
            recommended_strategy=recommended,
            supporting_risk=risk,
            estimated_token_usage=estimated_usage,
            target_token_budget=self._token_budget,
        )

    def _build_observability_signals(
        self,
        *,
        stage_summary: Mapping[str, Mapping[str, float]] | None,
        errors: Sequence[str],
        events: Sequence[IngestionEvent],
    ) -> ObservabilitySignals:
        latency_breakdown = {
            stage: details.get("avg_ms", details.get("duration_ms", 0.0))
            for stage, details in (stage_summary or {}).items()
        }
        service_events = [
            ServiceEventRecord.from_service_event(
                ServiceEvent(
                    timestamp=event.timestamp,
                    service="ingestion.pipeline",
                    name=event.name,
                    payload=dict(event.payload),
                )
            )
            for event in events
        ]
        return ObservabilitySignals(
            service_events=service_events,
            error_signals=list(errors),
            latency_breakdown=latency_breakdown,
        )


__all__ = ["IngestionAnalyticsEmitter"]

