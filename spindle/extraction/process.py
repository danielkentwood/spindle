"""
Process graph extraction functionality.

This module provides functions for extracting process DAGs from text.
"""

from typing import Optional
from spindle.baml_client import b
from spindle.baml_client.types import (
    ProcessExtractionResult,
    ProcessGraph,
)
from spindle.extraction.helpers import (
    _rehydrate_process_graph,
    _merge_process_graphs,
    _recalculate_process_boundaries,
    _validate_process_graph,
    _merge_issues,
)


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

