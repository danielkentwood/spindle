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

    Requires an ontology to be provided in the request.
    """
    try:
        ontology = _ontology_from_dict(request.ontology)

        existing_triples = None
        if request.existing_triples:
            existing_triples = _triples_from_dicts(request.existing_triples)

        extractor = SpindleExtractor(ontology=ontology)

        result = extractor.extract(
            text=request.text,
            source_name=request.source_name,
            source_url=request.source_url,
            existing_triples=existing_triples,
        )

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

    Requires an ontology. Processes texts sequentially, maintaining
    entity consistency across extractions.
    """
    try:
        ontology = _ontology_from_dict(request.ontology)

        existing_triples = None
        if request.existing_triples:
            existing_triples = _triples_from_dicts(request.existing_triples)

        extractor = SpindleExtractor(ontology=ontology)

        texts_list = []
        for text_item in request.texts:
            texts_list.append((
                text_item["text"],
                text_item["source_name"],
                text_item.get("source_url"),
            ))

        results = await extractor.extract_batch(
            texts=texts_list,
            existing_triples=existing_triples,
            max_concurrent=request.max_concurrent,
        )

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
    """Extract triples with streaming results (Server-Sent Events)."""
    async def event_generator():
        try:
            ontology = _ontology_from_dict(request.ontology)

            existing_triples = None
            if request.existing_triples:
                existing_triples = _triples_from_dicts(request.existing_triples)

            extractor = SpindleExtractor(ontology=ontology)

            texts_list = []
            for text_item in request.texts:
                texts_list.append((
                    text_item["text"],
                    text_item["source_name"],
                    text_item.get("source_url"),
                ))

            async for result in extractor.extract_batch_stream(
                texts=texts_list,
                existing_triples=existing_triples,
                max_concurrent=request.max_concurrent,
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

    Uses the session's ontology and accumulated triples for consistent extraction.
    The session must have an ontology set before extraction.
    """
    session = get_session(session_id)

    try:
        if not session.ontology:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Session ontology must be set before extraction. Use PUT /api/sessions/{id}/ontology first.",
            )

        ontology = _ontology_from_dict(session.ontology)
        extractor = SpindleExtractor(ontology=ontology)

        existing_triples = None
        if session.triples:
            existing_triples = _triples_from_dicts(session.triples)

        result = extractor.extract(
            text=request.text,
            source_name=request.source_name,
            source_url=request.source_url,
            existing_triples=existing_triples,
        )

        result_dict = serialize_extraction_result(result)
        session.add_triples(result_dict["triples"])

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
