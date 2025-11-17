"""Process extraction endpoints."""

from fastapi import APIRouter, HTTPException, status

from spindle.api.models import (
    ProcessDependencyInfo,
    ProcessExtractionRequest,
    ProcessExtractionResponse,
    ProcessGraphInfo,
    ProcessStepInfo,
)
from spindle.api.utils import convert_baml_to_dict, serialize_process_graph
from spindle.baml_client.types import ProcessGraph
from spindle.extraction.process import extract_process_graph

router = APIRouter()


def _process_graph_from_dict(graph_dict: dict) -> ProcessGraph:
    """Convert dictionary to ProcessGraph object."""
    try:
        return ProcessGraph.model_validate(graph_dict)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid process graph format: {str(e)}",
        )


def _serialize_process_graph_to_info(graph: ProcessGraph) -> ProcessGraphInfo:
    """Convert ProcessGraph to ProcessGraphInfo response model."""
    steps = [
        ProcessStepInfo(
            step_id=step.step_id,
            title=step.title,
            summary=step.summary,
            step_type=step.step_type.value,
            actors=step.actors,
            inputs=step.inputs,
            outputs=step.outputs,
            duration=step.duration,
            prerequisites=step.prerequisites,
        )
        for step in graph.steps
    ]
    
    dependencies = [
        ProcessDependencyInfo(
            from_step=dep.from_step,
            to_step=dep.to_step,
            relation=dep.relation,
            condition=dep.condition,
        )
        for dep in graph.dependencies
    ]
    
    return ProcessGraphInfo(
        process_name=graph.process_name,
        scope=graph.scope,
        primary_goal=graph.primary_goal,
        start_step_ids=graph.start_step_ids,
        end_step_ids=graph.end_step_ids,
        steps=steps,
        dependencies=dependencies,
        notes=graph.notes,
    )


# ============================================================================
# Process Extraction
# ============================================================================


@router.post("/extract", response_model=ProcessExtractionResponse)
async def extract_process(request: ProcessExtractionRequest):
    """Extract a process graph from text.
    
    This endpoint analyzes text to extract process flows, steps,
    and dependencies. It supports incremental extension of existing
    process graphs.
    
    Args:
        request: Process extraction parameters
        
    Returns:
        Extracted process graph
        
    Raises:
        HTTPException: 400 for invalid input, 500 for processing errors
    """
    try:
        # Parse existing graph if provided
        existing_graph = None
        if request.existing_graph:
            existing_graph = _process_graph_from_dict(request.existing_graph)
        
        # Extract process graph
        result = extract_process_graph(
            text=request.text,
            process_hint=request.process_hint,
            existing_graph=existing_graph,
        )
        
        # Convert to response
        graph_info = None
        if result.graph:
            graph_info = _serialize_process_graph_to_info(result.graph)
        
        # Serialize issues
        issues = [convert_baml_to_dict(issue) for issue in result.issues]
        
        return ProcessExtractionResponse(
            status=result.status,
            graph=graph_info,
            reasoning=result.reasoning,
            issues=issues,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Process extraction failed: {str(e)}",
        )

