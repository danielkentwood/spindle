"""Entity resolution endpoints."""

from pathlib import Path

from fastapi import APIRouter, HTTPException, status

from spindle.api.dependencies import get_session, verify_directory_access
from spindle.api.models import (
    ResolutionRequest,
    ResolutionResponse,
    ResolutionSessionRequest,
)
from spindle.entity_resolution.config import ResolutionConfig
from spindle.entity_resolution.resolver import EntityResolver
from spindle.graph_store import GraphStore
from spindle.vector_store import ChromaVectorStore

router = APIRouter()


# ============================================================================
# Stateless Resolution
# ============================================================================


@router.post("/resolve", response_model=ResolutionResponse)
async def resolve_entities(request: ResolutionRequest):
    """Run entity resolution on a graph (stateless mode).
    
    This endpoint runs entity resolution without maintaining session state.
    All paths and configuration must be provided in the request.
    
    Args:
        request: Resolution parameters
        
    Returns:
        Resolution results with statistics
        
    Raises:
        HTTPException: 400 for invalid input, 500 for processing errors
    """
    try:
        # Verify graph store path exists
        graph_store_path = verify_directory_access(request.graph_store_path, create=False)
        
        # Create graph store
        graph_store = GraphStore(str(graph_store_path))
        
        # Create vector store
        vector_store = ChromaVectorStore(uri=request.vector_store_uri)
        
        # Build resolution config
        config_dict = request.config or {}
        resolution_config = ResolutionConfig(**config_dict)
        
        # Create resolver
        resolver = EntityResolver(config=resolution_config)
        
        # Run resolution
        result = resolver.resolve_entities(
            graph_store=graph_store,
            vector_store=vector_store,
            apply_to_nodes=request.apply_to_nodes,
            apply_to_edges=request.apply_to_edges,
            context=request.context,
        )
        
        # Convert to response
        return ResolutionResponse(
            total_nodes_processed=result.total_nodes_processed,
            total_edges_processed=result.total_edges_processed,
            blocks_created=result.blocks_created,
            node_match_count=len(result.node_matches),
            edge_match_count=len(result.edge_matches),
            same_as_edges_created=result.same_as_edges_created,
            duplicate_clusters=result.duplicate_clusters,
            execution_time_seconds=result.execution_time_seconds,
            config=result.config.__dict__,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Entity resolution failed: {str(e)}",
        )


# ============================================================================
# Stateful Resolution (Session-based)
# ============================================================================


@router.post("/session/{session_id}/resolve", response_model=ResolutionResponse)
async def resolve_entities_session(session_id: str, request: ResolutionSessionRequest):
    """Run entity resolution within a session.
    
    This endpoint uses the session's graph store and vector store
    for resolution.
    
    Args:
        session_id: Session identifier
        request: Resolution parameters
        
    Returns:
        Resolution results
        
    Raises:
        HTTPException: 404 if session not found, 400 if session not configured, 500 for processing errors
    """
    session = get_session(session_id)
    
    # Verify session has required configuration
    if not session.graph_store_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session does not have a graph store configured",
        )
    
    if not session.vector_store_uri:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session does not have a vector store configured",
        )
    
    try:
        # Create graph store
        graph_store = GraphStore(session.graph_store_path)
        
        # Create vector store
        vector_store = ChromaVectorStore(uri=session.vector_store_uri)
        
        # Build resolution config
        config_dict = request.config or {}
        resolution_config = ResolutionConfig(**config_dict)
        
        # Create resolver
        resolver = EntityResolver(config=resolution_config)
        
        # Run resolution
        result = resolver.resolve_entities(
            graph_store=graph_store,
            vector_store=vector_store,
            apply_to_nodes=request.apply_to_nodes,
            apply_to_edges=request.apply_to_edges,
            context=request.context,
        )
        
        # Convert to response
        return ResolutionResponse(
            total_nodes_processed=result.total_nodes_processed,
            total_edges_processed=result.total_edges_processed,
            blocks_created=result.blocks_created,
            node_match_count=len(result.node_matches),
            edge_match_count=len(result.edge_matches),
            same_as_edges_created=result.same_as_edges_created,
            duplicate_clusters=result.duplicate_clusters,
            execution_time_seconds=result.execution_time_seconds,
            config=result.config.__dict__,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Entity resolution failed: {str(e)}",
        )

