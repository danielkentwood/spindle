"""Main FastAPI application for Spindle REST API."""

from datetime import datetime
from typing import List

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from spindle.api.models import (
    ErrorResponse,
    HealthResponse,
    OntologyUpdate,
    SessionCreate,
    SessionInfo,
    SessionUpdate,
)
from spindle.api.session import session_manager

# Create FastAPI app
app = FastAPI(
    title="Spindle API",
    description="REST API for Spindle knowledge graph extraction and management",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, configure specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Exception Handlers
# ============================================================================


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with structured error response."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            code=f"HTTP_{exc.status_code}",
            message=exc.detail,
            details={"headers": dict(exc.headers)} if exc.headers else None,
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            code="INTERNAL_SERVER_ERROR",
            message="An unexpected error occurred",
            details={"error": str(exc)},
        ).model_dump(),
    )


# ============================================================================
# Health Check
# ============================================================================


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health status."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
    )


# ============================================================================
# Session Management Endpoints
# ============================================================================


@app.post(
    "/api/sessions",
    response_model=SessionInfo,
    status_code=status.HTTP_201_CREATED,
    tags=["Sessions"],
)
async def create_session(request: SessionCreate):
    """Create a new session for stateful operations."""
    session = session_manager.create_session(
        name=request.name,
        graph_store_path=request.graph_store_path,
        vector_store_uri=request.vector_store_uri,
        config=request.config,
    )
    return session.to_info()


@app.get("/api/sessions", response_model=List[SessionInfo], tags=["Sessions"])
async def list_sessions():
    """List all active sessions."""
    return session_manager.list_sessions()


@app.get("/api/sessions/{session_id}", response_model=SessionInfo, tags=["Sessions"])
async def get_session(session_id: str):
    """Get information about a specific session."""
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )
    return session.to_info()


@app.delete(
    "/api/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Sessions"],
)
async def delete_session(session_id: str):
    """Delete a session."""
    deleted = session_manager.delete_session(session_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )
    return None


@app.put(
    "/api/sessions/{session_id}/ontology",
    response_model=SessionInfo,
    tags=["Sessions"],
)
async def update_session_ontology(session_id: str, request: OntologyUpdate):
    """Update the ontology for a session."""
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )

    session.update_ontology(request.ontology)
    return session.to_info()


@app.put(
    "/api/sessions/{session_id}/config",
    response_model=SessionInfo,
    tags=["Sessions"],
)
async def update_session_config(session_id: str, request: SessionUpdate):
    """Update session configuration."""
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )

    if request.name is not None:
        session.name = request.name
    if request.config is not None:
        session.update_config(request.config)

    return session.to_info()


# ============================================================================
# Router Registration
# ============================================================================

from spindle.api.routers import (
    extraction,
    resolution,
)
from spindle.api.kos_router import router as kos_router

app.include_router(extraction.router, prefix="/api/extraction", tags=["Extraction"])
app.include_router(resolution.router, prefix="/api/resolution", tags=["Resolution"])
app.include_router(kos_router, prefix="/kos", tags=["KOS"])


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run the API server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
