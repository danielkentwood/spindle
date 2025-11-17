"""Document ingestion endpoints."""

import json
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from spindle.api.dependencies import get_session, verify_file_access
from spindle.api.models import (
    ChunkInfo,
    DocumentInfo,
    IngestionMetrics,
    IngestionRequest,
    IngestionResponse,
    IngestionSessionRequest,
    IngestionStreamChunk,
)
from spindle.api.utils import convert_baml_to_dict
from spindle.ingestion.service import build_config, run_ingestion

router = APIRouter()


# ============================================================================
# Stateless Ingestion
# ============================================================================


@router.post("/ingest", response_model=IngestionResponse)
async def ingest_documents(request: IngestionRequest):
    """Ingest documents (stateless mode).
    
    This endpoint ingests documents without maintaining session state.
    All configuration must be provided in the request.
    
    Args:
        request: Ingestion parameters
        
    Returns:
        Ingestion result with documents, chunks, and metrics
        
    Raises:
        HTTPException: 400 for invalid input, 500 for processing errors
    """
    try:
        # Validate file paths
        paths = []
        for file_path in request.file_paths:
            try:
                path = verify_file_access(file_path)
                paths.append(path)
            except HTTPException:
                raise
        
        # Build config
        config = build_config(
            template_paths=[Path(p) for p in request.template_paths] if request.template_paths else None,
            catalog_url=request.catalog_url,
            vector_store_uri=request.vector_store_uri,
            cache_dir=Path(request.cache_dir) if request.cache_dir else None,
            allow_network_requests=request.allow_network_requests,
        )
        
        # Run ingestion
        result = run_ingestion(paths, config)
        
        # Convert to response model
        documents = [
            DocumentInfo(
                document_id=doc.document_id,
                source_path=str(doc.source_path),
                checksum=doc.checksum,
                loader_name=doc.loader_name,
                template_name=doc.template_name,
                metadata=doc.metadata,
                created_at=doc.created_at,
            )
            for doc in result.documents
        ]
        
        chunks = [
            ChunkInfo(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                text=chunk.text,
                metadata=chunk.metadata,
            )
            for chunk in result.chunks
        ]
        
        metrics = IngestionMetrics(
            processed_documents=result.metrics.processed_documents,
            processed_chunks=result.metrics.processed_chunks,
            bytes_read=result.metrics.bytes_read,
            errors=result.metrics.errors,
            started_at=result.metrics.started_at,
            finished_at=result.metrics.finished_at,
            extra=result.metrics.extra,
        )
        
        return IngestionResponse(
            documents=documents,
            chunks=chunks,
            metrics=metrics,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}",
        )


@router.post("/ingest/stream")
async def ingest_documents_stream(request: IngestionRequest):
    """Ingest documents with streaming progress (Server-Sent Events).
    
    This endpoint streams ingestion progress events as they occur.
    
    Args:
        request: Ingestion parameters
        
    Returns:
        Streaming response with SSE events
    """
    async def event_generator():
        """Generate SSE events from ingestion pipeline."""
        try:
            # Validate file paths
            paths = []
            for file_path in request.file_paths:
                try:
                    path = verify_file_access(file_path)
                    paths.append(path)
                except HTTPException as e:
                    error_event = IngestionStreamChunk(
                        event="error",
                        data={"message": e.detail},
                    )
                    yield f"data: {error_event.model_dump_json()}\n\n"
                    return
            
            # Build config
            config = build_config(
                template_paths=[Path(p) for p in request.template_paths] if request.template_paths else None,
                catalog_url=request.catalog_url,
                vector_store_uri=request.vector_store_uri,
                cache_dir=Path(request.cache_dir) if request.cache_dir else None,
                allow_network_requests=request.allow_network_requests,
            )
            
            # Send start event
            start_event = IngestionStreamChunk(
                event="ingestion_start",
                data={"file_count": len(paths)},
            )
            yield f"data: {start_event.model_dump_json()}\n\n"
            
            # Run ingestion (synchronous for now)
            result = run_ingestion(paths, config)
            
            # Send progress events from result
            for event in result.events:
                stream_event = IngestionStreamChunk(
                    event=event.name,
                    data=event.payload,
                    timestamp=event.timestamp,
                )
                yield f"data: {stream_event.model_dump_json()}\n\n"
            
            # Send completion event
            complete_event = IngestionStreamChunk(
                event="ingestion_complete",
                data={
                    "processed_documents": result.metrics.processed_documents,
                    "processed_chunks": result.metrics.processed_chunks,
                    "errors": result.metrics.errors,
                },
            )
            yield f"data: {complete_event.model_dump_json()}\n\n"
            
        except Exception as e:
            error_event = IngestionStreamChunk(
                event="error",
                data={"message": str(e)},
            )
            yield f"data: {error_event.model_dump_json()}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


# ============================================================================
# Stateful Ingestion (Session-based)
# ============================================================================


@router.post("/session/{session_id}/ingest", response_model=IngestionResponse)
async def ingest_documents_session(session_id: str, request: IngestionSessionRequest):
    """Ingest documents into a session.
    
    This endpoint uses session configuration for ingestion.
    
    Args:
        session_id: Session identifier
        request: Ingestion parameters (file paths only)
        
    Returns:
        Ingestion result
        
    Raises:
        HTTPException: 404 if session not found, 500 for processing errors
    """
    session = get_session(session_id)
    
    try:
        # Validate file paths
        paths = []
        for file_path in request.file_paths:
            path = verify_file_access(file_path)
            paths.append(path)
        
        # Build config from session
        config = build_config(
            catalog_url=session.catalog_url,
            vector_store_uri=session.vector_store_uri,
        )
        
        # Run ingestion
        result = run_ingestion(paths, config)
        
        # Convert to response model
        documents = [
            DocumentInfo(
                document_id=doc.document_id,
                source_path=str(doc.source_path),
                checksum=doc.checksum,
                loader_name=doc.loader_name,
                template_name=doc.template_name,
                metadata=doc.metadata,
                created_at=doc.created_at,
            )
            for doc in result.documents
        ]
        
        chunks = [
            ChunkInfo(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                text=chunk.text,
                metadata=chunk.metadata,
            )
            for chunk in result.chunks
        ]
        
        metrics = IngestionMetrics(
            processed_documents=result.metrics.processed_documents,
            processed_chunks=result.metrics.processed_chunks,
            bytes_read=result.metrics.bytes_read,
            errors=result.metrics.errors,
            started_at=result.metrics.started_at,
            finished_at=result.metrics.finished_at,
            extra=result.metrics.extra,
        )
        
        return IngestionResponse(
            documents=documents,
            chunks=chunks,
            metrics=metrics,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}",
        )

