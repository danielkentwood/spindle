"""
Internal helper functions for extraction operations.

This module contains internal utilities for:
- Event recording
- Span processing and index computation
- Process graph manipulation and validation
- String utilities
"""

from typing import List, Dict, Any, Optional, Tuple
import re
import baml_py
from spindle.baml_client.types import (
    CharacterSpan,
    EvidenceSpan,
    ProcessExtractionIssue,
    ProcessGraph,
    ProcessDependency,
    ProcessStep,
)
from spindle.observability import get_event_recorder

EXTRACTOR_RECORDER = get_event_recorder("extractor")
ONTOLOGY_RECORDER = get_event_recorder("ontology.recommender")


def _record_extractor_event(name: str, payload: Dict[str, Any]) -> None:
    EXTRACTOR_RECORDER.record(name=name, payload=payload)


def _record_ontology_event(name: str, payload: Dict[str, Any]) -> None:
    ONTOLOGY_RECORDER.record(name=name, payload=payload)


def _extract_model_from_collector(collector: baml_py.baml_py.Collector) -> Optional[str]:
    """Extract model/client name from BAML collector logs.
    
    Returns the model identifier if available, otherwise None.
    """
    if not hasattr(collector, 'logs') or not collector.logs:
        return None
    
    # Try to get model from logs
    for log in collector.logs:
        # Check for common model/client attributes
        if hasattr(log, 'client_name'):
            return getattr(log, 'client_name')
        if hasattr(log, 'model'):
            return getattr(log, 'model')
        if hasattr(log, 'client'):
            client = getattr(log, 'client')
            if isinstance(client, str):
                return client
            if hasattr(client, 'name'):
                return client.name
        # Check for model in metadata or options
        if hasattr(log, 'metadata') and isinstance(log.metadata, dict):
            model = log.metadata.get('model') or log.metadata.get('client_name')
            if model:
                return model
        if hasattr(log, 'options') and isinstance(log.options, dict):
            model = log.options.get('model') or log.options.get('client_name')
            if model:
                return model
    
    return None


def _normalize_ws(text: str) -> str:
    """Normalize whitespace: collapse all whitespace to single spaces."""
    return re.sub(r'\s+', ' ', text.strip())


def _find_span_indices(source_text: str, span_text: str, normalized_source: Optional[str] = None) -> Optional[Tuple[int, int]]:
    """
    Find the start and end indices of span_text within source_text.
    
    Uses multiple strategies in order:
    1. Exact substring match
    2. Whitespace-normalized fuzzy match (handles newlines, extra spaces)
    3. Case-insensitive match
    
    Note: The LLM often provides clean span text (without extra whitespace/newlines),
    while the source may contain formatting. The returned indices point to the
    location in the original source, which may have different whitespace but the
    same content when normalized.
    
    Args:
        source_text: The full source text
        span_text: The text span to find
        normalized_source: Pre-computed normalized source text (optional, for performance)
    
    Returns:
        Tuple of (start, end) indices, or None if not found
    """
    # Strategy 1: Try exact match first (fast path)
    start = source_text.find(span_text)
    if start != -1:
        return (start, start + len(span_text))
    
    # Strategy 2: Fuzzy match with whitespace normalization
    # The LLM often provides clean text without newlines/extra spaces,
    # but the source may have them. Create a pattern that matches
    # any amount of whitespace where the span has whitespace.
    
    # Split span into words and create pattern
    words = span_text.split()
    if len(words) > 0:
        # Escape each word and join with flexible whitespace
        pattern_parts = [re.escape(word) for word in words]
        pattern = r'\s+'.join(pattern_parts)
        
        try:
            match = re.search(pattern, source_text, re.DOTALL)
            if match:
                return (match.start(), match.end())
        except re.error:
            pass  # If regex fails, continue to next strategy
    
    # Strategy 3: Case-insensitive fuzzy match
    if len(words) > 0:
        pattern_parts = [re.escape(word) for word in words]
        pattern = r'\s+'.join(pattern_parts)
        try:
            match = re.search(pattern, source_text, re.DOTALL | re.IGNORECASE)
            if match:
                return (match.start(), match.end())
        except re.error:
            pass
    
    # Strategy 4: Case-insensitive exact match
    lower_source = source_text.lower()
    lower_span = span_text.lower()
    start = lower_source.find(lower_span)
    if start != -1:
        return (start, start + len(span_text))
    
    # Could not find the span
    return None


def _compute_all_span_indices(source_text: str, spans: List[CharacterSpan]) -> List[CharacterSpan]:
    """
    Compute character indices for all spans efficiently using batch processing.
    
    This function optimizes span index computation by:
    - Pre-computing normalized source text once
    - Processing all spans that need indices in a single pass
    - Reusing regex patterns where possible
    
    Args:
        source_text: The full source text
        spans: List of CharacterSpan objects that need index computation
    
    Returns:
        List of CharacterSpan objects with computed indices
    """
    # Pre-compute normalized source text once for all spans
    normalized_source = _normalize_ws(source_text)
    
    updated_spans = []
    for span in spans:
        if span.start is not None and span.end is not None:
            # Span already has indices, keep it as-is
            updated_spans.append(span)
        else:
            # Find the span text in the source text
            indices = _find_span_indices(source_text, span.text, normalized_source)
            if indices:
                # Create a new span with computed indices
                updated_span = CharacterSpan(
                    text=span.text,
                    start=indices[0],
                    end=indices[1]
                )
                updated_spans.append(updated_span)
            else:
                # If exact match not found, set to -1 to indicate failure
                updated_span = CharacterSpan(
                    text=span.text,
                    start=-1,
                    end=-1
                )
                updated_spans.append(updated_span)
    
    return updated_spans


def _dedupe_str_sequence(values: list[str]) -> list[str]:
    """Return a list with duplicates removed while preserving order."""

    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def _compute_evidence_span_indices(
    source_text: str,
    spans: list[EvidenceSpan],
) -> list[EvidenceSpan]:
    """
    Compute indices for EvidenceSpan objects while preserving existing metadata.
    """

    updated: list[EvidenceSpan] = []
    for span in spans:
        if span.start is not None and span.end is not None:
            updated.append(span)
            continue
        indices = _find_span_indices(source_text, span.text)
        if indices:
            updated.append(
                span.model_copy(update={"start": indices[0], "end": indices[1]})
            )
        else:
            updated.append(span.model_copy(update={"start": -1, "end": -1}))
    return updated


def _rehydrate_process_graph(source_text: str, graph: ProcessGraph) -> ProcessGraph:
    """
    Fill in evidence indices for a process graph and normalise basic fields.
    """

    normalised_steps: list[ProcessStep] = []
    for step in graph.steps:
        normalised_steps.append(
            step.model_copy(
                update={
                    "actors": _dedupe_str_sequence(list(step.actors)),
                    "inputs": _dedupe_str_sequence(list(step.inputs)),
                    "outputs": _dedupe_str_sequence(list(step.outputs)),
                    "prerequisites": _dedupe_str_sequence(list(step.prerequisites)),
                    "evidence": _compute_evidence_span_indices(
                        source_text, list(step.evidence)
                    ),
                }
            )
        )

    normalised_dependencies: list[ProcessDependency] = []
    for dependency in graph.dependencies:
        normalised_dependencies.append(
            dependency.model_copy(
                update={
                    "evidence": _compute_evidence_span_indices(
                        source_text, list(dependency.evidence)
                    )
                }
            )
        )

    return graph.model_copy(
        update={
            "steps": normalised_steps,
            "dependencies": normalised_dependencies,
            "notes": _dedupe_str_sequence(list(graph.notes)),
        }
    )


def _merge_evidence(
    primary: list[EvidenceSpan],
    secondary: list[EvidenceSpan],
) -> list[EvidenceSpan]:
    """Combine evidence spans, favouring those with resolved indices."""

    combined: list[EvidenceSpan] = []
    seen_keys: set[tuple[str, int | None, int | None]] = set()
    for span in [*primary, *secondary]:
        key = (span.text, span.start, span.end)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        combined.append(span)
    return combined


def _merge_process_step(existing: ProcessStep, incoming: ProcessStep) -> ProcessStep:
    """Merge two steps that represent the same identifier."""

    return existing.model_copy(
        update={
            "title": existing.title or incoming.title,
            "summary": existing.summary or incoming.summary,
            "step_type": existing.step_type or incoming.step_type,
            "actors": _dedupe_str_sequence([*existing.actors, *incoming.actors]),
            "inputs": _dedupe_str_sequence([*existing.inputs, *incoming.inputs]),
            "outputs": _dedupe_str_sequence([*existing.outputs, *incoming.outputs]),
            "duration": existing.duration or incoming.duration,
            "prerequisites": _dedupe_str_sequence(
                [*existing.prerequisites, *incoming.prerequisites]
            ),
            "evidence": _merge_evidence(list(existing.evidence), list(incoming.evidence)),
        }
    )


def _merge_process_dependency(
    existing: ProcessDependency,
    incoming: ProcessDependency,
) -> ProcessDependency:
    """Merge dependency metadata while preserving evidence."""

    condition = existing.condition or incoming.condition
    relation = existing.relation or incoming.relation
    merged_evidence = _merge_evidence(
        list(existing.evidence),
        list(incoming.evidence),
    )
    return existing.model_copy(
        update={
            "relation": relation,
            "condition": condition,
            "evidence": merged_evidence,
        }
    )


def _merge_process_graphs(
    base: ProcessGraph | None,
    update: ProcessGraph,
) -> ProcessGraph:
    """Combine two process graphs, preserving identifiers and merging metadata."""

    if base is None:
        return update

    step_index: dict[str, ProcessStep] = {
        step.step_id: step for step in base.steps
    }
    for step in update.steps:
        if step.step_id in step_index:
            step_index[step.step_id] = _merge_process_step(
                step_index[step.step_id], step
            )
        else:
            step_index[step.step_id] = step

    dependency_index: dict[tuple[str, str, str | None], ProcessDependency] = {}
    for dependency in base.dependencies:
        key = (dependency.from_step, dependency.to_step, dependency.relation)
        dependency_index[key] = dependency

    for dependency in update.dependencies:
        key = (dependency.from_step, dependency.to_step, dependency.relation)
        if key in dependency_index:
            dependency_index[key] = _merge_process_dependency(
                dependency_index[key], dependency
            )
        else:
            dependency_index[key] = dependency

    combined_notes = _dedupe_str_sequence([*base.notes, *update.notes])
    combined_scope = base.scope or update.scope
    combined_goal = base.primary_goal or update.primary_goal
    combined_name = base.process_name or update.process_name

    merged_graph = base.model_copy(
        update={
            "process_name": combined_name,
            "scope": combined_scope,
            "primary_goal": combined_goal,
            "notes": combined_notes,
            "steps": list(step_index.values()),
            "dependencies": list(dependency_index.values()),
        }
    )
    return merged_graph


def _recalculate_process_boundaries(graph: ProcessGraph) -> ProcessGraph:
    """Derive start and end step identifiers from dependencies."""

    step_ids = {step.step_id for step in graph.steps}
    incoming: dict[str, int] = {step_id: 0 for step_id in step_ids}
    outgoing: dict[str, int] = {step_id: 0 for step_id in step_ids}

    for dependency in graph.dependencies:
        if dependency.from_step in outgoing:
            outgoing[dependency.from_step] += 1
        if dependency.to_step in incoming:
            incoming[dependency.to_step] += 1

    start_ids = [step_id for step_id, count in incoming.items() if count == 0]
    end_ids = [step_id for step_id, count in outgoing.items() if count == 0]

    return graph.model_copy(
        update={
            "start_step_ids": _dedupe_str_sequence(start_ids),
            "end_step_ids": _dedupe_str_sequence(end_ids),
        }
    )


def _detect_cycles(graph: ProcessGraph) -> list[list[str]]:
    """Detect cycles using depth-first search; return list of cycles if found."""

    adjacency: dict[str, list[str]] = {step.step_id: [] for step in graph.steps}
    for dependency in graph.dependencies:
        adjacency.setdefault(dependency.from_step, []).append(dependency.to_step)

    visited: set[str] = set()
    stack: set[str] = set()
    cycles: list[list[str]] = []
    path: list[str] = []

    def visit(node: str) -> None:
        if node in stack:
            try:
                idx = path.index(node)
                cycles.append(path[idx:] + [node])
            except ValueError:
                cycles.append([node])
            return
        if node in visited:
            return
        visited.add(node)
        stack.add(node)
        path.append(node)
        for neighbour in adjacency.get(node, []):
            visit(neighbour)
        stack.remove(node)
        path.pop()

    for node in adjacency:
        if node not in visited:
            visit(node)

    return cycles


def _validate_process_graph(graph: ProcessGraph) -> list[ProcessExtractionIssue]:
    """Produce validation issues discovered during graph post-processing."""

    issues: list[ProcessExtractionIssue] = []
    step_ids = {step.step_id for step in graph.steps}
    orphan_edges: list[ProcessDependency] = []

    for dependency in graph.dependencies:
        if (
            dependency.from_step not in step_ids
            or dependency.to_step not in step_ids
        ):
            orphan_edges.append(dependency)

    if orphan_edges:
        issues.append(
            ProcessExtractionIssue(
                code="missing_step_reference",
                message="Some dependencies reference step identifiers that are not present in the graph.",
                related_step_ids=_dedupe_str_sequence(
                    [
                        *[edge.from_step for edge in orphan_edges],
                        *[edge.to_step for edge in orphan_edges],
                    ]
                ),
            )
        )

    cycles = _detect_cycles(graph)
    if cycles:
        issues.append(
            ProcessExtractionIssue(
                code="cycle_detected",
                message="Detected cycles within the process dependencies; review relations or step ordering.",
                related_step_ids=_dedupe_str_sequence([node for cycle in cycles for node in cycle]),
            )
        )

    return issues


def _merge_issues(
    original: list[ProcessExtractionIssue],
    additional: list[ProcessExtractionIssue],
) -> list[ProcessExtractionIssue]:
    """Combine issues, deduplicating by code and related step identifiers."""

    combined: list[ProcessExtractionIssue] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for issue in [*original, *additional]:
        key = (issue.code, tuple(sorted(issue.related_step_ids)))
        if key in seen:
            continue
        seen.add(key)
        combined.append(issue)
    return combined

