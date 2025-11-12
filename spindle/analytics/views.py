"""Curated views over persisted ingestion analytics observations."""

from __future__ import annotations

from collections import Counter, defaultdict
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


def ontology_recommendation_metrics(
    store: AnalyticsStore,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    """Aggregate LLM metrics from ontology recommendation service events, grouped by scope and model."""
    
    events = store.fetch_service_events(service="ontology.recommender", limit=limit)
    complete_events = [e for e in events if e.name == "recommend.complete"]
    
    if not complete_events:
        return {
            "total_calls": 0,
            "by_scope": {},
            "by_model": {},
            "total_tokens": 0,
            "total_cost": 0.0,
            "avg_latency_ms": 0.0,
        }
    
    by_scope: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "calls": 0,
        "total_tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_cost": 0.0,
        "latencies_ms": [],
    })
    
    by_model: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "calls": 0,
        "total_tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_cost": 0.0,
        "latencies_ms": [],
    })
    
    total_tokens = 0
    total_cost = 0.0
    all_latencies: list[float] = []
    
    for event in complete_events:
        payload = event.payload
        scope = payload.get("scope", "unknown")
        model = payload.get("model") or "unknown"
        
        tokens = payload.get("total_tokens", 0)
        cost = payload.get("cost", 0.0)
        latency = payload.get("latency_ms", 0.0)
        
        # Aggregate by scope
        by_scope[scope]["calls"] += 1
        by_scope[scope]["total_tokens"] += tokens
        by_scope[scope]["total_cost"] += cost
        by_scope[scope]["latencies_ms"].append(latency)
        
        if payload.get("input_tokens") is not None:
            by_scope[scope]["input_tokens"] += payload.get("input_tokens", 0)
        if payload.get("output_tokens") is not None:
            by_scope[scope]["output_tokens"] += payload.get("output_tokens", 0)
        
        # Aggregate by model
        by_model[model]["calls"] += 1
        by_model[model]["total_tokens"] += tokens
        by_model[model]["total_cost"] += cost
        by_model[model]["latencies_ms"].append(latency)
        
        if payload.get("input_tokens") is not None:
            by_model[model]["input_tokens"] += payload.get("input_tokens", 0)
        if payload.get("output_tokens") is not None:
            by_model[model]["output_tokens"] += payload.get("output_tokens", 0)
        
        total_tokens += tokens
        total_cost += cost
        all_latencies.append(latency)
    
    # Calculate averages
    for scope_data in by_scope.values():
        if scope_data["latencies_ms"]:
            scope_data["avg_latency_ms"] = mean(scope_data["latencies_ms"])
        else:
            scope_data["avg_latency_ms"] = 0.0
        scope_data.pop("latencies_ms")
    
    for model_data in by_model.values():
        if model_data["latencies_ms"]:
            model_data["avg_latency_ms"] = mean(model_data["latencies_ms"])
        else:
            model_data["avg_latency_ms"] = 0.0
        model_data.pop("latencies_ms")
    
    return {
        "total_calls": len(complete_events),
        "by_scope": dict(by_scope),
        "by_model": dict(by_model),
        "total_tokens": total_tokens,
        "total_cost": total_cost,
        "avg_latency_ms": mean(all_latencies) if all_latencies else 0.0,
    }


def triple_extraction_metrics(
    store: AnalyticsStore,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    """Aggregate LLM metrics from triple extraction service events, grouped by ontology_scope and model."""
    
    events = store.fetch_service_events(service="extractor", limit=limit)
    complete_events = [e for e in events if e.name in ("extract.complete", "extract_async.complete")]
    
    if not complete_events:
        return {
            "total_calls": 0,
            "by_scope": {},
            "by_model": {},
            "total_tokens": 0,
            "total_cost": 0.0,
            "avg_latency_ms": 0.0,
            "total_triples": 0,
            "total_entities": 0,
        }
    
    by_scope: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "calls": 0,
        "total_tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_cost": 0.0,
        "latencies_ms": [],
        "triples": 0,
        "entities": 0,
    })
    
    by_model: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "calls": 0,
        "total_tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_cost": 0.0,
        "latencies_ms": [],
        "triples": 0,
        "entities": 0,
    })
    
    total_tokens = 0
    total_cost = 0.0
    all_latencies: list[float] = []
    total_triples = 0
    
    for event in complete_events:
        payload = event.payload
        scope = payload.get("ontology_scope", "unknown")
        model = payload.get("model") or "unknown"
        
        tokens = payload.get("total_tokens", 0)
        cost = payload.get("cost", 0.0)
        latency = payload.get("latency_ms", 0.0)
        triple_count = payload.get("triple_count", 0)
        
        # Aggregate by scope
        by_scope[scope]["calls"] += 1
        by_scope[scope]["total_tokens"] += tokens
        by_scope[scope]["total_cost"] += cost
        by_scope[scope]["latencies_ms"].append(latency)
        by_scope[scope]["triples"] += triple_count
        
        if payload.get("input_tokens") is not None:
            by_scope[scope]["input_tokens"] += payload.get("input_tokens", 0)
        if payload.get("output_tokens") is not None:
            by_scope[scope]["output_tokens"] += payload.get("output_tokens", 0)
        
        # Aggregate by model
        by_model[model]["calls"] += 1
        by_model[model]["total_tokens"] += tokens
        by_model[model]["total_cost"] += cost
        by_model[model]["latencies_ms"].append(latency)
        by_model[model]["triples"] += triple_count
        
        if payload.get("input_tokens") is not None:
            by_model[model]["input_tokens"] += payload.get("input_tokens", 0)
        if payload.get("output_tokens") is not None:
            by_model[model]["output_tokens"] += payload.get("output_tokens", 0)
        
        total_tokens += tokens
        total_cost += cost
        all_latencies.append(latency)
        total_triples += triple_count
    
    # Calculate averages and extract unique entities
    for scope_data in by_scope.values():
        if scope_data["latencies_ms"]:
            scope_data["avg_latency_ms"] = mean(scope_data["latencies_ms"])
        else:
            scope_data["avg_latency_ms"] = 0.0
        scope_data.pop("latencies_ms")
        
        # Calculate triples per call
        if scope_data["calls"] > 0:
            scope_data["avg_triples_per_call"] = scope_data["triples"] / scope_data["calls"]
        else:
            scope_data["avg_triples_per_call"] = 0.0
    
    for model_data in by_model.values():
        if model_data["latencies_ms"]:
            model_data["avg_latency_ms"] = mean(model_data["latencies_ms"])
        else:
            model_data["avg_latency_ms"] = 0.0
        model_data.pop("latencies_ms")
        
        # Calculate triples per call
        if model_data["calls"] > 0:
            model_data["avg_triples_per_call"] = model_data["triples"] / model_data["calls"]
        else:
            model_data["avg_triples_per_call"] = 0.0
    
    return {
        "total_calls": len(complete_events),
        "by_scope": dict(by_scope),
        "by_model": dict(by_model),
        "total_tokens": total_tokens,
        "total_cost": total_cost,
        "avg_latency_ms": mean(all_latencies) if all_latencies else 0.0,
        "total_triples": total_triples,
    }


def entity_resolution_metrics(
    store: AnalyticsStore,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    """Aggregate LLM metrics from entity resolution service events, grouped by step and model."""
    
    events = store.fetch_service_events(service="entity_resolution", limit=limit)
    
    # Group by step type
    entity_matching_events = [e for e in events if e.name == "matching.entities.complete"]
    edge_matching_events = [e for e in events if e.name == "matching.edges.complete"]
    
    def aggregate_step_metrics(step_events: list, step_name: str) -> dict[str, Any]:
        if not step_events:
            return {
                "calls": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "avg_latency_ms": 0.0,
                "by_model": {},
            }
        
        by_model: dict[str, dict[str, Any]] = defaultdict(lambda: {
            "calls": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "latencies_ms": [],
        })
        
        total_tokens = 0
        total_cost = 0.0
        latencies: list[float] = []
        
        for event in step_events:
            payload = event.payload
            model = payload.get("model") or "unknown"
            
            tokens = payload.get("total_tokens", 0)
            cost = payload.get("cost", 0.0)
            latency = payload.get("latency_ms", 0.0)
            
            by_model[model]["calls"] += 1
            by_model[model]["total_tokens"] += tokens
            by_model[model]["total_cost"] += cost
            by_model[model]["latencies_ms"].append(latency)
            
            total_tokens += tokens
            total_cost += cost
            latencies.append(latency)
        
        # Calculate averages for each model
        for model_data in by_model.values():
            if model_data["latencies_ms"]:
                model_data["avg_latency_ms"] = mean(model_data["latencies_ms"])
            else:
                model_data["avg_latency_ms"] = 0.0
            model_data.pop("latencies_ms")
        
        return {
            "calls": len(step_events),
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "avg_latency_ms": mean(latencies) if latencies else 0.0,
            "by_model": dict(by_model),
        }
    
    return {
        "entity_matching": aggregate_step_metrics(entity_matching_events, "entity_matching"),
        "edge_matching": aggregate_step_metrics(edge_matching_events, "edge_matching"),
        "total_calls": len(entity_matching_events) + len(edge_matching_events),
        "total_tokens": sum(
            e.payload.get("total_tokens", 0)
            for e in entity_matching_events + edge_matching_events
        ),
        "total_cost": sum(
            e.payload.get("cost", 0.0)
            for e in entity_matching_events + edge_matching_events
        ),
    }


__all__ = [
    "corpus_overview",
    "document_size_table",
    "chunk_window_risk",
    "observability_events",
    "ontology_recommendation_metrics",
    "triple_extraction_metrics",
    "entity_resolution_metrics",
]

