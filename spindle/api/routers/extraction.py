"""Triple extraction endpoints."""

from typing import List

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from spindle.api.dependencies import get_session
from spindle.api.models import (
    BatchExtractionRequest,
    BatchExtractionResponse,
    ExtractionRequest,
    ExtractionResponse,
    SessionExtractionRequest,
)
from spindle.api.utils import convert_baml_to_dict, serialize_extraction_result
from spindle.baml_client.types import Ontology, Triple
from spindle.extraction.extractor import SpindleExtractor

router = APIRouter()


def _ontology_from_dict(ontology_dict: dict) -> Ontology:
    """Convert dictionary to Ontology object."""
    try:
        return Ontology.model_validate(ontology_dict)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid ontology format: {str(e)}",
        )


def _triples_from_dicts(triple_dicts: List[dict]) -> List[Triple]:
    """Convert list of dictionaries to Triple objects."""
    try:
        return [Triple.model_validate(t) for t in triple_dicts]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid triple format: {str(e)}",
        )


# ============================================================================
# Stateless Extraction
# ============================================================================


@router.post("/extract", response_model=ExtractionResponse)
async def extract_triples(request: ExtractionRequest):
    """Extract triples from text (stateless mode).
    
    This endpoint extracts triples from a single text using either a provided
    ontology or auto-recommendation.
    
    Args:
        request: Extraction parameters
        
    Returns:
        Extraction result with triples
        
    Raises:
        HTTPException: 400 for invalid input, 500 for processing errors
    """
    try:
        # Parse ontology if provided
        ontology = None
        if request.ontology:
            ontology = _ontology_from_dict(request.ontology)
        
        # Parse existing triples if provided
        existing_triples = None
        if request.existing_triples:
            existing_triples = _triples_from_dicts(request.existing_triples)
        
        # Create extractor
        extractor = SpindleExtractor(
            ontology=ontology,
            ontology_scope=request.ontology_scope or "balanced",
        )
        
        # Extract triples
        result = extractor.extract(
            text=request.text,
            source_name=request.source_name,
            source_url=request.source_url,
            existing_triples=existing_triples,
            ontology_scope=request.ontology_scope,
        )
        
        # Serialize result
        result_dict = serialize_extraction_result(result)
        
        return ExtractionResponse(
            triples=result_dict["triples"],
            reasoning=result_dict["reasoning"],
            source_name=request.source_name,
            triple_count=len(result_dict["triples"]),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Extraction failed: {str(e)}",
        )


@router.post("/extract/batch", response_model=BatchExtractionResponse)
async def extract_triples_batch(request: BatchExtractionRequest):
    """Extract triples from multiple texts (batch mode).
    
    This endpoint processes multiple texts sequentially, maintaining
    entity consistency across extractions.
    
    Args:
        request: Batch extraction parameters
        
    Returns:
        Batch extraction results
    """
    try:
        # Parse ontology if provided
        ontology = None
        if request.ontology:
            ontology = _ontology_from_dict(request.ontology)
        
        # Parse existing triples if provided
        existing_triples = None
        if request.existing_triples:
            existing_triples = _triples_from_dicts(request.existing_triples)
        
        # Create extractor
        extractor = SpindleExtractor(
            ontology=ontology,
            ontology_scope=request.ontology_scope or "balanced",
        )
        
        # Prepare texts list
        texts_list = []
        for text_item in request.texts:
            texts_list.append((
                text_item["text"],
                text_item["source_name"],
                text_item.get("source_url"),
            ))
        
        # Extract batch
        import asyncio
        results = await extractor.extract_batch(
            texts=texts_list,
            existing_triples=existing_triples,
            max_concurrent=request.max_concurrent,
            ontology_scope=request.ontology_scope,
        )
        
        # Convert to response
        extraction_responses = []
        total_triples = 0
        
        for i, result in enumerate(results):
            result_dict = serialize_extraction_result(result)
            extraction_responses.append(
                ExtractionResponse(
                    triples=result_dict["triples"],
                    reasoning=result_dict["reasoning"],
                    source_name=texts_list[i][1],
                    triple_count=len(result_dict["triples"]),
                )
            )
            total_triples += len(result_dict["triples"])
        
        return BatchExtractionResponse(
            results=extraction_responses,
            total_triples=total_triples,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch extraction failed: {str(e)}",
        )


@router.post("/extract/stream")
async def extract_triples_stream(request: BatchExtractionRequest):
    """Extract triples with streaming results (Server-Sent Events).
    
    This endpoint streams extraction results as they complete.
    
    Args:
        request: Batch extraction parameters
        
    Returns:
        Streaming response with SSE events
    """
    async def event_generator():
        """Generate SSE events from extraction results."""
        try:
            # Parse ontology if provided
            ontology = None
            if request.ontology:
                ontology = _ontology_from_dict(request.ontology)
            
            # Parse existing triples if provided
            existing_triples = None
            if request.existing_triples:
                existing_triples = _triples_from_dicts(request.existing_triples)
            
            # Create extractor
            extractor = SpindleExtractor(
                ontology=ontology,
                ontology_scope=request.ontology_scope or "balanced",
            )
            
            # Prepare texts list
            texts_list = []
            for text_item in request.texts:
                texts_list.append((
                    text_item["text"],
                    text_item["source_name"],
                    text_item.get("source_url"),
                ))
            
            # Stream extractions
            async for result in extractor.extract_batch_stream(
                texts=texts_list,
                existing_triples=existing_triples,
                max_concurrent=request.max_concurrent,
                ontology_scope=request.ontology_scope,
            ):
                result_dict = serialize_extraction_result(result)
                extraction_response = ExtractionResponse(
                    triples=result_dict["triples"],
                    reasoning=result_dict["reasoning"],
                    source_name=result.triples[0].source.source_name if result.triples else "unknown",
                    triple_count=len(result_dict["triples"]),
                )
                yield f"data: {extraction_response.model_dump_json()}\n\n"
                
        except Exception as e:
            error_data = {"error": str(e)}
            import json
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


# ============================================================================
# Stateful Extraction (Session-based)
# ============================================================================


@router.post("/session/{session_id}/extract", response_model=ExtractionResponse)
async def extract_triples_session(session_id: str, request: SessionExtractionRequest):
    """Extract triples within a session context.
    
    This endpoint uses the session's ontology and accumulated triples
    for consistent extraction.
    
    Args:
        session_id: Session identifier
        request: Extraction parameters
        
    Returns:
        Extraction result
        
    Raises:
        HTTPException: 404 if session not found, 500 for processing errors
    """
    session = get_session(session_id)
    
    try:
        # Use session ontology if available
        ontology = None
        if session.ontology:
            ontology = _ontology_from_dict(session.ontology)
        
        # Create extractor with session ontology
        extractor = SpindleExtractor(
            ontology=ontology,
            ontology_scope=request.ontology_scope or "balanced",
        )
        
        # Convert session triples to Triple objects
        existing_triples = None
        if session.triples:
            existing_triples = _triples_from_dicts(session.triples)
        
        # Extract triples
        result = extractor.extract(
            text=request.text,
            source_name=request.source_name,
            source_url=request.source_url,
            existing_triples=existing_triples,
            ontology_scope=request.ontology_scope,
        )
        
        # Update session with new triples and ontology
        result_dict = serialize_extraction_result(result)
        session.add_triples(result_dict["triples"])
        
        if extractor.ontology and not session.ontology:
            session.update_ontology(convert_baml_to_dict(extractor.ontology))
        
        return ExtractionResponse(
            triples=result_dict["triples"],
            reasoning=result_dict["reasoning"],
            source_name=request.source_name,
            triple_count=len(result_dict["triples"]),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Extraction failed: {str(e)}",
        )

