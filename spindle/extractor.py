"""
Spindle Extractor Module

This module provides the core triple extraction functionality for Spindle.

Key Components:
- SpindleExtractor: Extract triples from text using a predefined ontology
- OntologyRecommender: Automatically recommend ontologies by analyzing text,
  and conservatively extend existing ontologies when processing new sources
- Helper functions for ontology creation, serialization, and filtering
"""

from typing import List, Dict, Any, Optional, Tuple, AsyncIterator
from datetime import datetime
import asyncio
import re
from spindle.baml_client import b
from spindle.baml_client.async_client import b as async_b
from spindle.baml_client.types import (
    AttributeDefinition,
    AttributeValue,
    CharacterSpan,
    Entity,
    EntityType,
    ExtractionResult,
    Ontology,
    OntologyExtension,
    OntologyRecommendation,
    ProcessExtractionIssue,
    ProcessExtractionResult,
    ProcessGraph,
    ProcessDependency,
    EvidenceSpan,
    ProcessStep,
    RelationType,
    SourceMetadata,
    Triple,
)
from spindle.observability import get_event_recorder

try:
    from spindle.llm_config import (
        LLMConfig,
        detect_available_auth,
        create_baml_env_overrides,
    )

    LLM_CONFIG_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    LLMConfig = None
    detect_available_auth = None
    create_baml_env_overrides = None
    LLM_CONFIG_AVAILABLE = False


EXTRACTOR_RECORDER = get_event_recorder("extractor")
ONTOLOGY_RECORDER = get_event_recorder("ontology.recommender")


def _record_extractor_event(name: str, payload: Dict[str, Any]) -> None:
    EXTRACTOR_RECORDER.record(name=name, payload=payload)


def _record_ontology_event(name: str, payload: Dict[str, Any]) -> None:
    ONTOLOGY_RECORDER.record(name=name, payload=payload)


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


def extract_process_graph(
    text: str,
    process_hint: Optional[str] = None,
    existing_graph: Optional[ProcessGraph] = None,
) -> ProcessExtractionResult:
    """
    Extract or extend a process DAG from text using the ProcessGraph BAML function.
    """

    result = b.ExtractProcessGraph(
        text=text,
        process_hint=process_hint,
        existing_graph=existing_graph,
    )

    if result.graph is None:
        return result

    enriched_graph = _rehydrate_process_graph(text, result.graph)
    merged_graph = _merge_process_graphs(existing_graph, enriched_graph)
    final_graph = _recalculate_process_boundaries(merged_graph)
    validation_issues = _validate_process_graph(final_graph)

    return result.model_copy(
        update={
            "graph": final_graph,
            "issues": _merge_issues(result.issues, validation_issues),
        }
    )


class SpindleExtractor:
    """
    Main interface for extracting knowledge graph triples from text.
    
    This class wraps the BAML extraction function and provides a simple
    interface for incremental triple extraction with entity consistency.
    Each extracted triple includes source metadata, supporting text spans,
    and an extraction datetime (set automatically in post-processing).
    
    Supports multiple authentication methods for LLM access via optional LLM
    configuration or automatic detection of available credentials.

    If no ontology is provided at initialization, the extractor will
    automatically recommend one based on the text when extract() is called.
    """
    
    def __init__(
        self,
        ontology: Optional[Ontology] = None,
        ontology_scope: str = "balanced",
        llm_config: Optional["LLMConfig"] = None,
        auto_detect_auth: bool = True,
    ):
        """
        Initialize the extractor with an ontology.
        
        Args:
            ontology: Optional Ontology object defining valid entity and relation types.
                     If None, an ontology will be automatically recommended from the
                     text when extract() is first called.
            ontology_scope: Scope for auto-recommended ontologies. One of:
                          - "minimal": Essential concepts only (3-8 entity types, 4-10 relations)
                          - "balanced": Standard analysis (6-12 entity types, 8-15 relations)
                          - "comprehensive": Detailed ontology (10-20 entity types, 12-25 relations)
                          Only used if ontology is None.
            llm_config: Optional LLMConfig providing explicit authentication details
                for LLM access. If None and auto_detect_auth is True, credentials
                will be auto-detected from the environment.
            auto_detect_auth: Whether to auto-detect authentication credentials when
                llm_config is not provided. Defaults to True.
        """
        self.ontology = ontology
        self.ontology_scope = ontology_scope
        self.llm_config = llm_config
        self.auto_detect_auth = auto_detect_auth
        self._baml_env_overrides: Optional[Dict[str, str]] = None

        self._configure_baml_client(self.llm_config, self.auto_detect_auth)
        self._ontology_recommender = (
            None
            if ontology is not None
            else OntologyRecommender(
                llm_config=self.llm_config,
                auto_detect_auth=self.auto_detect_auth,
            )
        )
    
    def _configure_baml_client(
        self,
        llm_config: Optional["LLMConfig"],
        auto_detect_auth: bool,
    ) -> None:
        """Configure BAML client environment overrides based on LLM config."""
        if not LLM_CONFIG_AVAILABLE:
            self._baml_env_overrides = None
            return

        config = llm_config
        if config is None and auto_detect_auth and detect_available_auth is not None:
            config = detect_available_auth()

        if config is None or create_baml_env_overrides is None:
            self._baml_env_overrides = None
            return

        self.llm_config = config
        self._baml_env_overrides = create_baml_env_overrides(config)

    def _get_baml_client(self):
        """Return the configured BAML sync client."""
        if self._baml_env_overrides:
            return b.with_options(env=self._baml_env_overrides)
        return b

    def _get_async_baml_client(self):
        """Return the configured BAML async client."""
        if self._baml_env_overrides:
            return async_b.with_options(env=self._baml_env_overrides)
        return async_b
    
    def extract(
        self,
        text: str,
        source_name: str,
        source_url: Optional[str] = None,
        existing_triples: List[Triple] = None,
        ontology_scope: Optional[str] = None
    ) -> ExtractionResult:
        """
        Extract triples from text using the configured or auto-recommended ontology.
        
        If no ontology was provided at initialization, one will be automatically
        recommended based on the text before extraction.
        
        Args:
            text: The text to extract triples from
            source_name: Name or identifier of the source document
            source_url: Optional URL of the source document
            existing_triples: Optional list of previously extracted triples
                            to maintain entity consistency. Duplicate triples
                            from different sources are allowed.
            ontology_scope: Override the default ontology scope for this extraction.
                          One of "minimal", "balanced", or "comprehensive".
                          Only used if no ontology was provided at init.
        
        Returns:
            ExtractionResult containing the extracted triples (with source
            metadata, supporting spans, and extraction datetime) and reasoning
        """
        if existing_triples is None:
            existing_triples = []

        _record_extractor_event(
            "extract.start",
            {
                "source_name": source_name,
                "source_url": source_url,
                "existing_triples": len(existing_triples),
            },
        )

        try:
            # Auto-recommend ontology if not provided
            if self.ontology is None:
                scope = ontology_scope or self.ontology_scope
                recommendation = self._ontology_recommender.recommend(
                    text=text,
                    scope=scope
                )
                self.ontology = recommendation.ontology

            # Create source metadata
            source_metadata = SourceMetadata(
                source_name=source_name,
                source_url=source_url
            )

            # Call the BAML extraction function
            baml_client = self._get_baml_client()
            result = baml_client.ExtractTriples(
                text=text,
                ontology=self.ontology,
                source_metadata=source_metadata,
                existing_triples=existing_triples
            )

            # Post-processing: Set extraction datetime for all triples
            extraction_time = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            for triple in result.triples:
                triple.extraction_datetime = extraction_time

                # Post-processing: Compute character indices for supporting spans using batch processing
                triple.supporting_spans = _compute_all_span_indices(text, triple.supporting_spans)
        except Exception as exc:
            _record_extractor_event(
                "extract.error",
                {
                    "source_name": source_name,
                    "error": str(exc),
                },
            )
            raise

        _record_extractor_event(
            "extract.complete",
            {
                "source_name": source_name,
                "triple_count": len(result.triples),
            },
        )
        return result
    
    async def extract_async(
        self,
        text: str,
        source_name: str,
        source_url: Optional[str] = None,
        existing_triples: List[Triple] = None,
        ontology_scope: Optional[str] = None
    ) -> ExtractionResult:
        """
        Extract triples from text using the configured or auto-recommended ontology (async version).
        
        This is the async version of extract(). It uses the async BAML client for
        non-blocking extraction, which is useful for batch processing and streaming.
        
        If no ontology was provided at initialization, one will be automatically
        recommended based on the text before extraction.
        
        Args:
            text: The text to extract triples from
            source_name: Name or identifier of the source document
            source_url: Optional URL of the source document
            existing_triples: Optional list of previously extracted triples
                            to maintain entity consistency. Duplicate triples
                            from different sources are allowed.
            ontology_scope: Override the default ontology scope for this extraction.
                          One of "minimal", "balanced", or "comprehensive".
                          Only used if no ontology was provided at init.
        
        Returns:
            ExtractionResult containing the extracted triples (with source
            metadata, supporting spans, and extraction datetime) and reasoning
        """
        if existing_triples is None:
            existing_triples = []

        _record_extractor_event(
            "extract_async.start",
            {
                "source_name": source_name,
                "source_url": source_url,
                "existing_triples": len(existing_triples),
            },
        )

        try:
            # Auto-recommend ontology if not provided
            if self.ontology is None:
                scope = ontology_scope or self.ontology_scope
                recommendation = self._ontology_recommender.recommend(
                    text=text,
                    scope=scope
                )
                self.ontology = recommendation.ontology

            # Create source metadata
            source_metadata = SourceMetadata(
                source_name=source_name,
                source_url=source_url
            )

            # Call the async BAML extraction function
            async_baml_client = self._get_async_baml_client()
            result = await async_baml_client.ExtractTriples(
                text=text,
                ontology=self.ontology,
                source_metadata=source_metadata,
                existing_triples=existing_triples
            )

            # Post-processing: Set extraction datetime for all triples
            extraction_time = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            for triple in result.triples:
                triple.extraction_datetime = extraction_time

                # Post-processing: Compute character indices for supporting spans using batch processing
                triple.supporting_spans = _compute_all_span_indices(text, triple.supporting_spans)
        except Exception as exc:
            _record_extractor_event(
                "extract_async.error",
                {
                    "source_name": source_name,
                    "error": str(exc),
                },
            )
            raise

        _record_extractor_event(
            "extract_async.complete",
            {
                "source_name": source_name,
                "triple_count": len(result.triples),
            },
        )
        return result
    
    async def extract_batch(
        self,
        texts: List[Tuple[str, str, Optional[str]]],
        existing_triples: List[Triple] = None,
        max_concurrent: int = 20,
        ontology_scope: Optional[str] = None
    ) -> List[ExtractionResult]:
        """
        Extract triples from multiple texts sequentially with entity consistency.
        
        This method processes texts one at a time (asynchronously) to maintain
        entity consistency. Triples extracted from earlier texts are automatically
        included in the existing_triples for later texts, ensuring consistent
        entity naming across the batch.
        
        Args:
            texts: List of tuples (text, source_name, source_url) to extract from.
                  source_url can be None.
            existing_triples: Optional initial list of previously extracted triples
                            to maintain entity consistency across the batch.
            max_concurrent: Maximum concurrent extractions (default: 20).
                          Currently used for internal async operations.
            ontology_scope: Override the default ontology scope for extractions.
                          One of "minimal", "balanced", or "comprehensive".
                          Only used if no ontology was provided at init.
        
        Returns:
            List of ExtractionResult objects, one per input text, in the same order
        """
        if existing_triples is None:
            existing_triples = []

        _record_extractor_event(
            "extract_batch.start",
            {
                "item_count": len(texts),
                "existing_triples": len(existing_triples),
            },
        )

        results = []
        accumulated_triples = list(existing_triples)

        try:
            # Process texts sequentially but asynchronously to maintain consistency
            for text, source_name, source_url in texts:
                # Pass a copy of accumulated_triples to avoid reference issues
                result = await self.extract_async(
                    text=text,
                    source_name=source_name,
                    source_url=source_url,
                    existing_triples=list(accumulated_triples),
                    ontology_scope=ontology_scope
                )
                results.append(result)
                # Accumulate triples from this extraction for next texts
                accumulated_triples.extend(result.triples)
        except Exception as exc:
            _record_extractor_event(
                "extract_batch.error",
                {
                    "item_count": len(texts),
                    "error": str(exc),
                },
            )
            raise

        _record_extractor_event(
            "extract_batch.complete",
            {
                "item_count": len(results),
                "total_triples": sum(len(res.triples) for res in results),
            },
        )
        return results
    
    async def extract_batch_stream(
        self,
        texts: List[Tuple[str, str, Optional[str]]],
        existing_triples: List[Triple] = None,
        max_concurrent: int = 20,
        ontology_scope: Optional[str] = None
    ) -> AsyncIterator[ExtractionResult]:
        """
        Extract triples from multiple texts, streaming results as they complete.
        
        This is an async generator that yields ExtractionResult objects as soon
        as each extraction completes. Results are yielded in the same order as
        the input texts, maintaining sequential consistency where triples from
        earlier texts are available to later texts.
        
        Args:
            texts: List of tuples (text, source_name, source_url) to extract from.
                  source_url can be None.
            existing_triples: Optional initial list of previously extracted triples
                            to maintain entity consistency across the batch.
            max_concurrent: Maximum concurrent extractions (default: 20).
                          Currently used for internal async operations.
            ontology_scope: Override the default ontology scope for extractions.
                          One of "minimal", "balanced", or "comprehensive".
                          Only used if no ontology was provided at init.
        
        Yields:
            ExtractionResult objects as they complete, in input order
        
        Example:
            >>> extractor = SpindleExtractor(ontology)
            >>> texts = [
            ...     ("Alice works at TechCorp", "doc1", None),
            ...     ("Bob manages Alice", "doc2", None)
            ... ]
            >>> async for result in extractor.extract_batch_stream(texts):
            ...     print(f"Extracted {len(result.triples)} triples from {result.triples[0].source.source_name}")
        """
        if existing_triples is None:
            existing_triples = []
        
        accumulated_triples = list(existing_triples)
        _record_extractor_event(
            "extract_batch_stream.start",
            {
                "item_count": len(texts),
                "existing_triples": len(existing_triples),
            },
        )

        try:
            # Process texts sequentially but asynchronously to maintain consistency
            for text, source_name, source_url in texts:
                # Pass a copy of accumulated_triples to avoid reference issues
                result = await self.extract_async(
                    text=text,
                    source_name=source_name,
                    source_url=source_url,
                    existing_triples=list(accumulated_triples),
                    ontology_scope=ontology_scope
                )
                _record_extractor_event(
                    "extract_batch_stream.item",
                    {
                        "source_name": source_name,
                        "triple_count": len(result.triples),
                    },
                )
                # Yield result as soon as it completes
                yield result
                # Accumulate triples from this extraction for next texts
                accumulated_triples.extend(result.triples)
        except Exception as exc:
            _record_extractor_event(
                "extract_batch_stream.error",
                {
                    "item_count": len(texts),
                    "error": str(exc),
                },
            )
            raise

        _record_extractor_event(
            "extract_batch_stream.complete",
            {
                "item_count": len(texts),
            },
        )


def create_ontology(
    entity_types: List[Dict[str, Any]],
    relation_types: List[Dict[str, str]]
) -> Ontology:
    """
    Factory function to create an Ontology object from dictionaries.
    
    Args:
        entity_types: List of dicts with 'name', 'description', and optional 'attributes' keys
                     Each attribute should have 'name', 'type', and 'description'
        relation_types: List of dicts with 'name', 'description', 'domain',
                       and 'range' keys
    
    Returns:
        Ontology object
    
    Example:
        >>> entity_types = [
        ...     {
        ...         "name": "Campaign",
        ...         "description": "A marketing campaign",
        ...         "attributes": [
        ...             {
        ...                 "name": "campaign_launch_dt",
        ...                 "type": "date",
        ...                 "description": "The date the campaign launched"
        ...             },
        ...             {
        ...                 "name": "campaign_completion_dt",
        ...                 "type": "date",
        ...                 "description": "The date the campaign completed"
        ...             }
        ...         ]
        ...     },
        ...     {"name": "Person", "description": "A human being"}
        ... ]
        >>> relation_types = [
        ...     {
        ...         "name": "manages",
        ...         "description": "Management relationship",
        ...         "domain": "Person",
        ...         "range": "Campaign"
        ...     }
        ... ]
        >>> ontology = create_ontology(entity_types, relation_types)
    """
    entity_objs = []
    for et in entity_types:
        # Handle attributes if present
        attributes = []
        if "attributes" in et and et["attributes"]:
            attributes = [
                AttributeDefinition(
                    name=attr["name"],
                    type=attr["type"],
                    description=attr["description"]
                )
                for attr in et["attributes"]
            ]
        
        entity_objs.append(
            EntityType(
                name=et["name"],
                description=et["description"],
                attributes=attributes
            )
        )
    
    relation_objs = [
        RelationType(
            name=rt["name"],
            description=rt["description"],
            domain=rt["domain"],
            range=rt["range"]
        )
        for rt in relation_types
    ]
    
    return Ontology(entity_types=entity_objs, relation_types=relation_objs)


def create_source_metadata(
    source_name: str,
    source_url: Optional[str] = None
) -> SourceMetadata:
    """
    Create a SourceMetadata object.
    
    Args:
        source_name: Name or identifier of the source
        source_url: Optional URL of the source
    
    Returns:
        SourceMetadata object
    """
    return SourceMetadata(source_name=source_name, source_url=source_url)


def triples_to_dict(triples: List[Triple]) -> List[Dict[str, Any]]:
    """
    Convert Triple objects to dictionaries for serialization.
    
    Args:
        triples: List of Triple objects with Entity subjects and objects
    
    Returns:
        List of dictionaries with all triple fields including structured entities
        
    Note:
        Entities are serialized with name, type, description, and custom_atts.
        Custom attributes include type metadata: {"value": "...", "type": "..."}
    """
    return [
        {
            "subject": {
                "name": triple.subject.name,
                "type": triple.subject.type,
                "description": triple.subject.description,
                "custom_atts": {
                    attr_name: {
                        "value": attr_val.value,
                        "type": attr_val.type
                    }
                    for attr_name, attr_val in triple.subject.custom_atts.items()
                }
            },
            "predicate": triple.predicate,
            "object": {
                "name": triple.object.name,
                "type": triple.object.type,
                "description": triple.object.description,
                "custom_atts": {
                    attr_name: {
                        "value": attr_val.value,
                        "type": attr_val.type
                    }
                    for attr_name, attr_val in triple.object.custom_atts.items()
                }
            },
            "source": {
                "source_name": triple.source.source_name,
                "source_url": triple.source.source_url
            },
            "supporting_spans": [
                {
                    "text": span.text,
                    "start": span.start if span.start is not None else -1,
                    "end": span.end if span.end is not None else -1
                }
                for span in triple.supporting_spans
            ],
            "extraction_datetime": triple.extraction_datetime if triple.extraction_datetime else ""
        }
        for triple in triples
    ]


def dict_to_triples(dicts: List[Dict[str, Any]]) -> List[Triple]:
    """
    Convert dictionaries back to Triple objects.
    
    Args:
        dicts: List of dictionaries with triple fields including structured entities
    
    Returns:
        List of Triple objects with Entity subjects and objects
        
    Note:
        Handles both old format (string entities) and new format (Entity objects)
        for backward compatibility during migration.
    """
    triples = []
    for d in dicts:
        # Handle subject - support both old string format and new Entity format
        if isinstance(d["subject"], str):
            # Old format: convert string to minimal Entity
            subject = Entity(
                name=d["subject"],
                type="Unknown",
                description="",
                custom_atts={}
            )
        else:
            # New format: reconstruct Entity from dict
            subject = Entity(
                name=d["subject"]["name"],
                type=d["subject"]["type"],
                description=d["subject"]["description"],
                custom_atts={
                    attr_name: AttributeValue(
                        value=attr_val["value"],
                        type=attr_val["type"]
                    )
                    for attr_name, attr_val in d["subject"].get("custom_atts", {}).items()
                }
            )
        
        # Handle object - support both old string format and new Entity format
        if isinstance(d["object"], str):
            # Old format: convert string to minimal Entity
            obj = Entity(
                name=d["object"],
                type="Unknown",
                description="",
                custom_atts={}
            )
        else:
            # New format: reconstruct Entity from dict
            obj = Entity(
                name=d["object"]["name"],
                type=d["object"]["type"],
                description=d["object"]["description"],
                custom_atts={
                    attr_name: AttributeValue(
                        value=attr_val["value"],
                        type=attr_val["type"]
                    )
                    for attr_name, attr_val in d["object"].get("custom_atts", {}).items()
                }
            )
        
        triples.append(
            Triple(
                subject=subject,
                predicate=d["predicate"],
                object=obj,
                source=SourceMetadata(
                    source_name=d["source"]["source_name"],
                    source_url=d["source"].get("source_url")
                ),
                supporting_spans=[
                    CharacterSpan(
                        text=span["text"],
                        start=span.get("start") if span.get("start", -1) >= 0 else None,
                        end=span.get("end") if span.get("end", -1) >= 0 else None
                    )
                    for span in d.get("supporting_spans", [])
                ],
                extraction_datetime=d.get("extraction_datetime", "")
            )
        )
    
    return triples


def get_supporting_text(triple: Triple) -> List[str]:
    """
    Extract the supporting text snippets from a triple's character spans.
    
    Args:
        triple: A Triple object with supporting_spans
    
    Returns:
        List of text strings from the supporting spans
    """
    return [span.text for span in triple.supporting_spans]


def filter_triples_by_source(
    triples: List[Triple],
    source_name: str
) -> List[Triple]:
    """
    Filter triples to only those from a specific source.
    
    Args:
        triples: List of Triple objects
        source_name: Name of the source to filter by
    
    Returns:
        List of triples from the specified source
    """
    return [t for t in triples if t.source.source_name == source_name]


def parse_extraction_datetime(triple: Triple) -> Optional[datetime]:
    """
    Parse the extraction datetime string into a datetime object.
    
    Args:
        triple: A Triple object with extraction_datetime
    
    Returns:
        datetime object, or None if parsing fails
    """
    try:
        # Handle various ISO 8601 formats
        dt_str = triple.extraction_datetime.strip()
        # Try parsing with common formats
        for fmt in [
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S.%f%z",
        ]:
            try:
                return datetime.strptime(dt_str, fmt)
            except ValueError:
                continue
        # Try ISO format parser
        return datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
    except (ValueError, AttributeError):
        return None


def filter_triples_by_date_range(
    triples: List[Triple],
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> List[Triple]:
    """
    Filter triples by extraction date range.
    
    Args:
        triples: List of Triple objects
        start_date: Optional start datetime (inclusive)
        end_date: Optional end datetime (inclusive)
    
    Returns:
        List of triples extracted within the date range
    """
    filtered = []
    for triple in triples:
        dt = parse_extraction_datetime(triple)
        if dt is None:
            continue
        if start_date and dt < start_date:
            continue
        if end_date and dt > end_date:
            continue
        filtered.append(triple)
    return filtered


class OntologyRecommender:
    """
    Service for recommending ontologies based on text analysis.
    
    This class wraps the BAML ontology recommendation function and provides
    a simple interface for automatically generating ontologies suitable for
    knowledge graph extraction from text.
    
    Uses principle-based ontology design with configurable scope levels
    instead of hard numerical limits.

    Supports optional LLM configuration for authenticated access, with automatic
    credential detection when explicit configuration is not supplied.
    """

    def __init__(
        self,
        llm_config: Optional["LLMConfig"] = None,
        auto_detect_auth: bool = True,
    ):
        """
        Initialize the ontology recommender with optional LLM configuration.

        Args:
            llm_config: Optional LLMConfig providing authentication details. If None
                and auto_detect_auth is True, credentials will be auto-detected.
            auto_detect_auth: Whether to auto-detect authentication when llm_config
                is not provided.
        """
        self.llm_config = llm_config
        self.auto_detect_auth = auto_detect_auth
        self._baml_env_overrides: Optional[Dict[str, str]] = None
        self._configure_baml_client(self.llm_config, self.auto_detect_auth)

    def _configure_baml_client(
        self,
        llm_config: Optional["LLMConfig"],
        auto_detect_auth: bool,
    ) -> None:
        """Configure BAML client environment overrides based on LLM config."""
        if not LLM_CONFIG_AVAILABLE:
            self._baml_env_overrides = None
            return

        config = llm_config
        if config is None and auto_detect_auth and detect_available_auth is not None:
            config = detect_available_auth()

        if config is None or create_baml_env_overrides is None:
            self._baml_env_overrides = None
            return

        self.llm_config = config
        self._baml_env_overrides = create_baml_env_overrides(config)

    def _get_baml_client(self):
        """Return the configured BAML client."""
        if self._baml_env_overrides:
            return b.with_options(env=self._baml_env_overrides)
        return b
    
    def recommend(
        self,
        text: str,
        scope: str = "balanced"
    ) -> OntologyRecommendation:
        """
        Analyze text and recommend an appropriate ontology.
        
        This method examines the provided text to infer its overarching
        purpose/goal and domain, then recommends entity types and relation
        types that would be most suitable for extracting knowledge from
        this and similar texts.
        
        The ontology is designed using principled guidelines about granularity
        and abstraction level, rather than hard numerical limits. The LLM
        determines the appropriate number of types based on the text's
        complexity and domain.
        
        Args:
            text: The text to analyze for ontology recommendation
            scope: Desired ontology scope, one of:
                  - "minimal": Essential concepts only (3-8 entity types, 4-10 relations)
                    Use for quick extraction, simple queries, broad patterns
                  - "balanced": Standard analysis (6-12 entity types, 8-15 relations)
                    Use for general-purpose extraction (DEFAULT)
                  - "comprehensive": Detailed ontology (10-20 entity types, 12-25 relations)
                    Use for detailed analysis, domain expertise, research
        
        Returns:
            OntologyRecommendation containing:
                - ontology: The recommended Ontology object (ready for use
                           with SpindleExtractor)
                - text_purpose: Analysis of the text's overarching purpose
                - reasoning: Explanation of the recommendation choices
        
        Example:
            >>> recommender = OntologyRecommender()
            >>> text = "Alice works at TechCorp in San Francisco..."
            >>> 
            >>> # Get a balanced ontology (default)
            >>> recommendation = recommender.recommend(text)
            >>> 
            >>> # Or request a specific scope
            >>> minimal_rec = recommender.recommend(text, scope="minimal")
            >>> comprehensive_rec = recommender.recommend(text, scope="comprehensive")
            >>> 
            >>> print(recommendation.text_purpose)
            >>> extractor = SpindleExtractor(recommendation.ontology)
        """
        _record_ontology_event(
            "recommend.start",
            {
                "scope": scope,
                "text_length": len(text),
            },
        )
        try:
            client = self._get_baml_client()
            result = client.RecommendOntology(
                text=text,
                scope=scope
            )
        except Exception as exc:
            _record_ontology_event(
                "recommend.error",
                {
                    "scope": scope,
                    "error": str(exc),
                },
            )
            raise

        _record_ontology_event(
            "recommend.complete",
            {
                "scope": scope,
                "entity_type_count": len(result.ontology.entity_types),
                "relation_type_count": len(result.ontology.relation_types),
            },
        )
        return result
    
    def recommend_and_extract(
        self,
        text: str,
        source_name: str,
        source_url: Optional[str] = None,
        scope: str = "balanced",
        existing_triples: List[Triple] = None
    ) -> Tuple[OntologyRecommendation, ExtractionResult]:
        """
        Recommend an ontology and immediately use it to extract triples.
        
        This is a convenience method that combines ontology recommendation
        and triple extraction in one step. Useful when you don't have a
        pre-defined ontology and want to let the system automatically
        determine the best structure for your text.
        
        Args:
            text: The text to analyze and extract from
            source_name: Name or identifier of the source document
            source_url: Optional URL of the source document
            scope: Desired ontology scope ("minimal", "balanced", or "comprehensive")
            existing_triples: Optional list of previously extracted triples
                             for entity consistency
        
        Returns:
            Tuple of (OntologyRecommendation, ExtractionResult):
                - OntologyRecommendation with the generated ontology
                - ExtractionResult with extracted triples using that ontology
        
        Example:
            >>> recommender = OntologyRecommender()
            >>> text = "Alice works at TechCorp..."
            >>> 
            >>> # Use default balanced scope
            >>> recommendation, extraction = recommender.recommend_and_extract(
            ...     text=text,
            ...     source_name="example.txt"
            ... )
            >>> 
            >>> # Or request comprehensive scope
            >>> recommendation, extraction = recommender.recommend_and_extract(
            ...     text=complex_paper,
            ...     source_name="research.pdf",
            ...     scope="comprehensive"
            ... )
            >>> 
            >>> print(f"Purpose: {recommendation.text_purpose}")
            >>> print(f"Extracted {len(extraction.triples)} triples")
        """
        _record_ontology_event(
            "recommend_and_extract.start",
            {
                "scope": scope,
                "text_length": len(text),
            },
        )
        try:
            # First, recommend the ontology
            recommendation = self.recommend(
                text=text,
                scope=scope
            )

            # Then, use the recommended ontology to extract triples
            extractor = SpindleExtractor(
                recommendation.ontology,
                llm_config=self.llm_config,
                auto_detect_auth=self.auto_detect_auth,
            )
            extraction_result = extractor.extract(
                text=text,
                source_name=source_name,
                source_url=source_url,
                existing_triples=existing_triples
            )
        except Exception as exc:
            _record_ontology_event(
                "recommend_and_extract.error",
                {
                    "scope": scope,
                    "source_name": source_name,
                    "error": str(exc),
                },
            )
            raise

        _record_ontology_event(
            "recommend_and_extract.complete",
            {
                "scope": scope,
                "source_name": source_name,
                "triple_count": len(extraction_result.triples),
            },
        )
        return recommendation, extraction_result
    
    def analyze_extension(
        self,
        text: str,
        current_ontology: Ontology,
        scope: str = "balanced"
    ) -> OntologyExtension:
        """
        Analyze whether an existing ontology needs to be extended for new text.
        
        This method conservatively analyzes if the current ontology is sufficient
        to extract critical information from new text, or if it needs extension.
        
        CONSERVATIVE APPROACH: The analysis defaults to NO extension. Extensions
        are only recommended if failing to extend would result in losing critical
        information that cannot be captured with existing types.
        
        Args:
            text: The new text to analyze
            current_ontology: The existing Ontology to potentially extend
            scope: The scope level to consider ("minimal", "balanced", "comprehensive")
                  This affects how conservative the extension analysis is.
        
        Returns:
            OntologyExtension containing:
                - needs_extension: Boolean indicating if extension is needed
                - new_entity_types: List of new entity types to add (empty if none)
                - new_relation_types: List of new relation types to add (empty if none)
                - critical_information_at_risk: Description of what would be lost
                - reasoning: Detailed explanation of the decision
        
        Example:
            >>> recommender = OntologyRecommender()
            >>> # Existing ontology for business domain
            >>> ontology = create_ontology(business_entities, business_relations)
            >>> 
            >>> # Analyze new text from medical domain
            >>> medical_text = "Dr. Chen prescribed Medication A for the patient..."
            >>> extension = recommender.analyze_extension(
            ...     text=medical_text,
            ...     current_ontology=ontology,
            ...     scope="balanced"
            ... )
            >>> 
            >>> if extension.needs_extension:
            ...     print(f"Extension needed: {extension.critical_information_at_risk}")
            ...     print(f"New types: {[et.name for et in extension.new_entity_types]}")
            ... else:
            ...     print(f"No extension needed: {extension.reasoning}")
        """
        client = self._get_baml_client()
        result = client.AnalyzeOntologyExtension(
            text=text,
            current_ontology=current_ontology,
            scope=scope
        )
        
        return result
    
    def extend_ontology(
        self,
        current_ontology: Ontology,
        extension: OntologyExtension
    ) -> Ontology:
        """
        Apply an ontology extension to create an extended ontology.
        
        This creates a new Ontology object by combining the current ontology
        with the new types from the extension analysis.
        
        Args:
            current_ontology: The existing Ontology
            extension: The OntologyExtension from analyze_extension()
        
        Returns:
            A new Ontology object with the extensions applied
        
        Example:
            >>> extension = recommender.analyze_extension(text, ontology)
            >>> if extension.needs_extension:
            ...     extended_ontology = recommender.extend_ontology(ontology, extension)
            ...     print(f"Original: {len(ontology.entity_types)} types")
            ...     print(f"Extended: {len(extended_ontology.entity_types)} types")
        """
        # Combine existing and new entity types
        all_entity_types = list(current_ontology.entity_types) + list(extension.new_entity_types)
        
        # Combine existing and new relation types
        all_relation_types = list(current_ontology.relation_types) + list(extension.new_relation_types)
        
        # Create and return the extended ontology
        return Ontology(
            entity_types=all_entity_types,
            relation_types=all_relation_types
        )
    
    def analyze_and_extend(
        self,
        text: str,
        current_ontology: Ontology,
        scope: str = "balanced",
        auto_apply: bool = True
    ) -> Tuple[OntologyExtension, Optional[Ontology]]:
        """
        Analyze extension needs and optionally apply them in one step.
        
        This is a convenience method that combines analyze_extension() and
        extend_ontology() with optional automatic application.
        
        Args:
            text: The new text to analyze
            current_ontology: The existing Ontology
            scope: The scope level for analysis
            auto_apply: If True, automatically apply extensions and return new ontology.
                       If False, return None as second element (user must apply manually).
        
        Returns:
            Tuple of (OntologyExtension, Optional[Ontology]):
                - OntologyExtension with the analysis results
                - Extended Ontology if auto_apply=True and needs_extension=True, else None
        
        Example:
            >>> # Automatic application
            >>> extension, new_ontology = recommender.analyze_and_extend(
            ...     text=new_text,
            ...     current_ontology=ontology,
            ...     auto_apply=True
            ... )
            >>> 
            >>> if new_ontology:
            ...     # Extension was needed and applied
            ...     extractor = SpindleExtractor(new_ontology)
            ... else:
            ...     # No extension needed, use original
            ...     extractor = SpindleExtractor(ontology)
        """
        extension = self.analyze_extension(text, current_ontology, scope)
        
        if extension.needs_extension and auto_apply:
            extended_ontology = self.extend_ontology(current_ontology, extension)
            return extension, extended_ontology
        
        return extension, None


def ontology_to_dict(ontology: Ontology) -> Dict[str, List[Dict[str, Any]]]:
    """
    Convert an Ontology object to a dictionary for serialization.
    
    Args:
        ontology: An Ontology object
    
    Returns:
        Dictionary with 'entity_types' and 'relation_types' keys,
        including attributes for entity types
    
    Example:
        >>> ontology = create_ontology(entity_types, relation_types)
        >>> ontology_dict = ontology_to_dict(ontology)
        >>> json.dumps(ontology_dict, indent=2)
    """
    return {
        "entity_types": [
            {
                "name": et.name,
                "description": et.description,
                "attributes": [
                    {
                        "name": attr.name,
                        "type": attr.type,
                        "description": attr.description
                    }
                    for attr in et.attributes
                ]
            }
            for et in ontology.entity_types
        ],
        "relation_types": [
            {
                "name": rt.name,
                "description": rt.description,
                "domain": rt.domain,
                "range": rt.range
            }
            for rt in ontology.relation_types
        ]
    }


def recommendation_to_dict(
    recommendation: OntologyRecommendation
) -> Dict[str, Any]:
    """
    Convert an OntologyRecommendation to a dictionary for serialization.
    
    Args:
        recommendation: An OntologyRecommendation object
    
    Returns:
        Dictionary with ontology, text_purpose, and reasoning
    """
    return {
        "ontology": ontology_to_dict(recommendation.ontology),
        "text_purpose": recommendation.text_purpose,
        "reasoning": recommendation.reasoning
    }


def extension_to_dict(
    extension: OntologyExtension
) -> Dict[str, Any]:
    """
    Convert an OntologyExtension to a dictionary for serialization.
    
    Args:
        extension: An OntologyExtension object
    
    Returns:
        Dictionary with extension analysis results including attributes
    """
    return {
        "needs_extension": extension.needs_extension,
        "new_entity_types": [
            {
                "name": et.name,
                "description": et.description,
                "attributes": [
                    {
                        "name": attr.name,
                        "type": attr.type,
                        "description": attr.description
                    }
                    for attr in et.attributes
                ]
            }
            for et in extension.new_entity_types
        ],
        "new_relation_types": [
            {
                "name": rt.name,
                "description": rt.description,
                "domain": rt.domain,
                "range": rt.range
            }
            for rt in extension.new_relation_types
        ],
        "critical_information_at_risk": extension.critical_information_at_risk,
        "reasoning": extension.reasoning
    }

