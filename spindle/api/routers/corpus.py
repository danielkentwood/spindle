"""API endpoints for corpus management."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from spindle.ingestion.storage import DocumentCatalog
from spindle.ingestion.storage.corpus import CorpusManager
from spindle.ingestion.types import Corpus


router = APIRouter()


# =============================================================================
# Request/Response Models
# =============================================================================


class CorpusCreate(BaseModel):
    """Request to create a new corpus."""

    name: str = Field(..., description="Corpus name")
    description: str = Field("", description="Corpus description")
    corpus_id: Optional[str] = Field(None, description="Optional custom ID")


class CorpusInfo(BaseModel):
    """Corpus information response."""

    corpus_id: str
    name: str
    description: str
    created_at: datetime
    updated_at: datetime
    document_count: int
    pipeline_state: Dict[str, Any]


class CorpusUpdate(BaseModel):
    """Request to update a corpus."""

    name: Optional[str] = Field(None, description="New name")
    description: Optional[str] = Field(None, description="New description")


class AddDocumentsRequest(BaseModel):
    """Request to add documents to a corpus."""

    document_ids: List[str] = Field(..., description="Document IDs to add")


class AddDocumentsResponse(BaseModel):
    """Response from adding documents."""

    added_count: int
    corpus_id: str


class DocumentInCorpus(BaseModel):
    """Document info within a corpus."""

    document_id: str
    source_path: str
    added_at: datetime


# =============================================================================
# Helper Functions
# =============================================================================


def get_corpus_manager() -> CorpusManager:
    """Get or create a CorpusManager instance.

    In production, this should be injected via FastAPI dependencies
    with proper database configuration.
    """
    # Default to in-memory SQLite for now
    # In production, configure via SpindleConfig
    catalog = DocumentCatalog("sqlite:///spindle_storage/catalog.db")
    return CorpusManager(catalog)


def corpus_to_info(corpus: Corpus, manager: CorpusManager) -> CorpusInfo:
    """Convert Corpus to CorpusInfo response."""
    doc_count = manager.get_corpus_document_count(corpus.corpus_id)
    return CorpusInfo(
        corpus_id=corpus.corpus_id,
        name=corpus.name,
        description=corpus.description,
        created_at=corpus.created_at,
        updated_at=corpus.updated_at,
        document_count=doc_count,
        pipeline_state=corpus.pipeline_state,
    )


# =============================================================================
# Endpoints
# =============================================================================


@router.post(
    "",
    response_model=CorpusInfo,
    status_code=status.HTTP_201_CREATED,
)
async def create_corpus(request: CorpusCreate):
    """Create a new corpus.

    A corpus is a collection of documents that will be processed
    through the Ontology Pipeline to extract semantic knowledge.

    Args:
        request: Corpus creation parameters.

    Returns:
        Created corpus information.
    """
    manager = get_corpus_manager()

    corpus = manager.create_corpus(
        name=request.name,
        description=request.description,
        corpus_id=request.corpus_id,
    )

    return corpus_to_info(corpus, manager)


@router.get("", response_model=List[CorpusInfo])
async def list_corpora():
    """List all corpora.

    Returns:
        List of corpus information.
    """
    manager = get_corpus_manager()
    corpora = manager.list_corpora()

    return [corpus_to_info(c, manager) for c in corpora]


@router.get("/{corpus_id}", response_model=CorpusInfo)
async def get_corpus(corpus_id: str):
    """Get corpus by ID.

    Args:
        corpus_id: Corpus identifier.

    Returns:
        Corpus information.

    Raises:
        HTTPException: 404 if corpus not found.
    """
    manager = get_corpus_manager()
    corpus = manager.get_corpus(corpus_id)

    if corpus is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Corpus not found: {corpus_id}",
        )

    return corpus_to_info(corpus, manager)


@router.put("/{corpus_id}", response_model=CorpusInfo)
async def update_corpus(corpus_id: str, request: CorpusUpdate):
    """Update corpus properties.

    Args:
        corpus_id: Corpus identifier.
        request: Update parameters.

    Returns:
        Updated corpus information.

    Raises:
        HTTPException: 404 if corpus not found.
    """
    manager = get_corpus_manager()

    corpus = manager.update_corpus(
        corpus_id=corpus_id,
        name=request.name,
        description=request.description,
    )

    if corpus is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Corpus not found: {corpus_id}",
        )

    return corpus_to_info(corpus, manager)


@router.delete("/{corpus_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_corpus(corpus_id: str, delete_documents: bool = False):
    """Delete a corpus.

    Args:
        corpus_id: Corpus identifier.
        delete_documents: If True, also remove document associations.

    Raises:
        HTTPException: 404 if corpus not found.
    """
    manager = get_corpus_manager()

    deleted = manager.delete_corpus(corpus_id, delete_documents=delete_documents)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Corpus not found: {corpus_id}",
        )

    return None


@router.post("/{corpus_id}/documents", response_model=AddDocumentsResponse)
async def add_documents(corpus_id: str, request: AddDocumentsRequest):
    """Add documents to a corpus.

    Documents must already exist in the catalog (ingested).

    Args:
        corpus_id: Corpus identifier.
        request: Document IDs to add.

    Returns:
        Number of documents added.

    Raises:
        HTTPException: 404 if corpus not found.
    """
    manager = get_corpus_manager()

    try:
        count = manager.add_documents(corpus_id, request.document_ids)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )

    return AddDocumentsResponse(
        added_count=count,
        corpus_id=corpus_id,
    )


@router.get("/{corpus_id}/documents", response_model=List[DocumentInCorpus])
async def get_corpus_documents(corpus_id: str):
    """Get documents in a corpus.

    Args:
        corpus_id: Corpus identifier.

    Returns:
        List of documents in the corpus.
    """
    manager = get_corpus_manager()

    # Verify corpus exists
    corpus = manager.get_corpus(corpus_id)
    if corpus is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Corpus not found: {corpus_id}",
        )

    corpus_docs = manager.get_corpus_documents(corpus_id)
    documents = manager.get_document_artifacts(corpus_id)

    # Create lookup for source paths
    doc_paths = {d.document_id: str(d.source_path) for d in documents}

    return [
        DocumentInCorpus(
            document_id=cd.document_id,
            source_path=doc_paths.get(cd.document_id, ""),
            added_at=cd.added_at,
        )
        for cd in corpus_docs
    ]


@router.delete(
    "/{corpus_id}/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def remove_document(corpus_id: str, document_id: str):
    """Remove a document from a corpus.

    This only removes the association; the document remains in the catalog.

    Args:
        corpus_id: Corpus identifier.
        document_id: Document identifier.
    """
    manager = get_corpus_manager()

    count = manager.remove_documents(corpus_id, [document_id])

    if count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found in corpus: {document_id}",
        )

    return None

