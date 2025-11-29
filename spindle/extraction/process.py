"""
Process graph extraction functionality.

This module provides functions for extracting process DAGs from text.
"""

from typing import Optional
from langfuse import observe, get_client as get_langfuse_client
import baml_py
from spindle.baml_client import b
from spindle.baml_client.types import (
    ProcessExtractionResult,
    ProcessGraph,
)
from spindle.extraction.helpers import (
    _extract_model_from_collector,
    _rehydrate_process_graph,
    _merge_process_graphs,
    _recalculate_process_boundaries,
    _validate_process_graph,
    _merge_issues,
)


@observe(as_type="generation", capture_input=False, capture_output=False)
def extract_process_graph(
    text: str,
    process_hint: Optional[str] = None,
    existing_graph: Optional[ProcessGraph] = None,
) -> ProcessExtractionResult:
    """
    Extract or extend a process DAG from text using the ProcessGraph BAML function.
    """

    # Call BAML extraction with collector
    collector = baml_py.baml_py.Collector("process-graph-extraction-collector")
    result = b.with_options(collector=collector).ExtractProcessGraph(
        text=text,
        process_hint=process_hint,
        existing_graph=existing_graph,
    )

    # Extract model from collector
    model = _extract_model_from_collector(collector) or "CustomFast"

    # Update Langfuse generation
    langfuse = get_langfuse_client()
    langfuse.update_current_generation(
        name="ExtractProcessGraph",
        model=model,
        input={
            "text": text,
            "process_hint": process_hint,
            "has_existing_graph": existing_graph is not None,
        },
        output={
            "has_graph": result.graph is not None,
            "graph": {
                "nodes": [
                    {"id": n.step_id, "label": n.title, "type": str(n.step_type)}
                    for n in result.graph.steps
                ] if result.graph else [],
                "edges": [
                    {"source": e.from_step, "target": e.to_step, "label": e.relation}
                    for e in result.graph.dependencies
                ] if result.graph else [],
            } if result.graph else None,
            "reasoning": result.reasoning,
            "issues": [str(i) for i in result.issues] if result.issues else [],
        },
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

