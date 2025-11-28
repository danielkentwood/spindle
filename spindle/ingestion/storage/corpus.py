"""Corpus management for organizing documents into collections."""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from spindle.ingestion.storage.catalog import (
    Base,
    ChunkRow,
    CorpusDocumentRow,
    CorpusRow,
    DocumentCatalog,
    DocumentRow,
)
from spindle.ingestion.types import (
    ChunkArtifact,
    Corpus,
    CorpusDocument,
    DocumentArtifact,
)


class CorpusManager:
    """Manage document corpora for ontology pipeline processing.

    CorpusManager provides CRUD operations for corpora and their documents,
    building on the existing DocumentCatalog infrastructure.

    Example:
        >>> catalog = DocumentCatalog("sqlite:///spindle.db")
        >>> manager = CorpusManager(catalog)
        >>> corpus = manager.create_corpus("Research Papers", "Academic papers on NLP")
        >>> manager.add_documents(corpus.corpus_id, ["doc-123", "doc-456"])
        >>> documents = manager.get_corpus_documents(corpus.corpus_id)
    """

    def __init__(self, catalog: DocumentCatalog) -> None:
        """Initialize CorpusManager with an existing DocumentCatalog.

        Args:
            catalog: DocumentCatalog instance for database access.
        """
        self._catalog = catalog
        # Ensure corpus tables exist
        Base.metadata.create_all(self._catalog._engine)

    def create_corpus(
        self,
        name: str,
        description: str = "",
        corpus_id: Optional[str] = None,
    ) -> Corpus:
        """Create a new corpus.

        Args:
            name: Human-readable name for the corpus.
            description: Optional description of the corpus contents/purpose.
            corpus_id: Optional custom ID. If not provided, a UUID is generated.

        Returns:
            The created Corpus object.
        """
        if corpus_id is None:
            corpus_id = str(uuid.uuid4())

        now = datetime.utcnow()
        corpus = Corpus(
            corpus_id=corpus_id,
            name=name,
            description=description,
            created_at=now,
            updated_at=now,
            pipeline_state={},
        )

        with self._catalog.session() as session:
            session.merge(
                CorpusRow(
                    corpus_id=corpus.corpus_id,
                    name=corpus.name,
                    description=corpus.description,
                    created_at=corpus.created_at,
                    updated_at=corpus.updated_at,
                    pipeline_state=corpus.pipeline_state,
                )
            )

        return corpus

    def get_corpus(self, corpus_id: str) -> Optional[Corpus]:
        """Retrieve a corpus by ID.

        Args:
            corpus_id: The corpus identifier.

        Returns:
            Corpus object if found, None otherwise.
        """
        with self._catalog.session() as session:
            row = session.get(CorpusRow, corpus_id)
            if row is None:
                return None
            return Corpus(
                corpus_id=row.corpus_id,
                name=row.name,
                description=row.description,
                created_at=row.created_at,
                updated_at=row.updated_at,
                pipeline_state=row.pipeline_state or {},
            )

    def list_corpora(self) -> List[Corpus]:
        """List all corpora.

        Returns:
            List of all Corpus objects.
        """
        with self._catalog.session() as session:
            rows = session.execute(select(CorpusRow)).scalars().all()
            return [
                Corpus(
                    corpus_id=row.corpus_id,
                    name=row.name,
                    description=row.description,
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                    pipeline_state=row.pipeline_state or {},
                )
                for row in rows
            ]

    def update_corpus(
        self,
        corpus_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        pipeline_state: Optional[dict] = None,
    ) -> Optional[Corpus]:
        """Update corpus properties.

        Args:
            corpus_id: The corpus identifier.
            name: New name (if provided).
            description: New description (if provided).
            pipeline_state: New pipeline state dict (if provided).

        Returns:
            Updated Corpus object if found, None otherwise.
        """
        with self._catalog.session() as session:
            row = session.get(CorpusRow, corpus_id)
            if row is None:
                return None

            if name is not None:
                row.name = name
            if description is not None:
                row.description = description
            if pipeline_state is not None:
                row.pipeline_state = pipeline_state
            row.updated_at = datetime.utcnow()

            session.merge(row)

            return Corpus(
                corpus_id=row.corpus_id,
                name=row.name,
                description=row.description,
                created_at=row.created_at,
                updated_at=row.updated_at,
                pipeline_state=row.pipeline_state or {},
            )

    def delete_corpus(self, corpus_id: str, delete_documents: bool = False) -> bool:
        """Delete a corpus and optionally its document associations.

        Args:
            corpus_id: The corpus identifier.
            delete_documents: If True, also removes document associations.
                            Documents themselves are not deleted from the catalog.

        Returns:
            True if corpus was deleted, False if not found.
        """
        with self._catalog.session() as session:
            row = session.get(CorpusRow, corpus_id)
            if row is None:
                return False

            # Delete document associations
            if delete_documents:
                session.execute(
                    CorpusDocumentRow.__table__.delete().where(
                        CorpusDocumentRow.corpus_id == corpus_id
                    )
                )

            session.delete(row)
            return True

    def add_documents(
        self,
        corpus_id: str,
        document_ids: Sequence[str],
    ) -> int:
        """Add existing documents to a corpus.

        Args:
            corpus_id: The corpus identifier.
            document_ids: List of document IDs to add.

        Returns:
            Number of documents successfully added.

        Raises:
            ValueError: If corpus does not exist.
        """
        with self._catalog.session() as session:
            # Verify corpus exists
            corpus_row = session.get(CorpusRow, corpus_id)
            if corpus_row is None:
                raise ValueError(f"Corpus not found: {corpus_id}")

            count = 0
            now = datetime.utcnow()
            for doc_id in document_ids:
                # Verify document exists
                doc_row = session.get(DocumentRow, doc_id)
                if doc_row is None:
                    continue

                # Add association (merge handles duplicates)
                session.merge(
                    CorpusDocumentRow(
                        corpus_id=corpus_id,
                        document_id=doc_id,
                        added_at=now,
                    )
                )
                count += 1

            # Update corpus timestamp
            corpus_row.updated_at = now
            session.merge(corpus_row)

            return count

    def remove_documents(
        self,
        corpus_id: str,
        document_ids: Sequence[str],
    ) -> int:
        """Remove documents from a corpus.

        Args:
            corpus_id: The corpus identifier.
            document_ids: List of document IDs to remove.

        Returns:
            Number of documents removed.
        """
        with self._catalog.session() as session:
            count = 0
            for doc_id in document_ids:
                result = session.execute(
                    CorpusDocumentRow.__table__.delete().where(
                        (CorpusDocumentRow.corpus_id == corpus_id)
                        & (CorpusDocumentRow.document_id == doc_id)
                    )
                )
                count += result.rowcount

            # Update corpus timestamp if any removed
            if count > 0:
                corpus_row = session.get(CorpusRow, corpus_id)
                if corpus_row:
                    corpus_row.updated_at = datetime.utcnow()
                    session.merge(corpus_row)

            return count

    def get_corpus_documents(self, corpus_id: str) -> List[CorpusDocument]:
        """Get all document associations for a corpus.

        Args:
            corpus_id: The corpus identifier.

        Returns:
            List of CorpusDocument objects.
        """
        with self._catalog.session() as session:
            rows = (
                session.execute(
                    select(CorpusDocumentRow).where(
                        CorpusDocumentRow.corpus_id == corpus_id
                    )
                )
                .scalars()
                .all()
            )
            return [
                CorpusDocument(
                    corpus_id=row.corpus_id,
                    document_id=row.document_id,
                    added_at=row.added_at,
                )
                for row in rows
            ]

    def get_corpus_document_count(self, corpus_id: str) -> int:
        """Get the number of documents in a corpus.

        Args:
            corpus_id: The corpus identifier.

        Returns:
            Document count.
        """
        with self._catalog.session() as session:
            from sqlalchemy import func

            result = session.execute(
                select(func.count()).where(CorpusDocumentRow.corpus_id == corpus_id)
            ).scalar()
            return result or 0

    def get_document_artifacts(self, corpus_id: str) -> List[DocumentArtifact]:
        """Get full DocumentArtifact objects for all documents in a corpus.

        Args:
            corpus_id: The corpus identifier.

        Returns:
            List of DocumentArtifact objects.
        """
        with self._catalog.session() as session:
            # Join corpus_documents with documents
            rows = (
                session.execute(
                    select(DocumentRow)
                    .join(
                        CorpusDocumentRow,
                        DocumentRow.document_id == CorpusDocumentRow.document_id,
                    )
                    .where(CorpusDocumentRow.corpus_id == corpus_id)
                )
                .scalars()
                .all()
            )
            return [
                DocumentArtifact(
                    document_id=row.document_id,
                    source_path=Path(row.source_path),
                    checksum=row.checksum,
                    loader_name=row.loader_name,
                    template_name=row.template_name,
                    metadata=row.metadata_ or {},
                    raw_bytes=None,  # Not stored in DB
                    created_at=row.created_at,
                )
                for row in rows
            ]

    def get_corpus_chunks(self, corpus_id: str) -> List[ChunkArtifact]:
        """Get all chunks for documents in a corpus.

        Args:
            corpus_id: The corpus identifier.

        Returns:
            List of ChunkArtifact objects.
        """
        with self._catalog.session() as session:
            # Join corpus_documents with chunks via document_id
            rows = (
                session.execute(
                    select(ChunkRow)
                    .join(
                        CorpusDocumentRow,
                        ChunkRow.document_id == CorpusDocumentRow.document_id,
                    )
                    .where(CorpusDocumentRow.corpus_id == corpus_id)
                )
                .scalars()
                .all()
            )
            return [
                ChunkArtifact(
                    chunk_id=row.chunk_id,
                    document_id=row.document_id,
                    text=row.text,
                    metadata=row.metadata_ or {},
                    embedding=row.embedding,
                )
                for row in rows
            ]

    def update_pipeline_state(
        self,
        corpus_id: str,
        stage: str,
        state: dict,
    ) -> Optional[Corpus]:
        """Update the pipeline state for a specific stage.

        Args:
            corpus_id: The corpus identifier.
            stage: Pipeline stage name (e.g., 'vocabulary', 'taxonomy').
            state: State dict for the stage.

        Returns:
            Updated Corpus object if found, None otherwise.
        """
        with self._catalog.session() as session:
            row = session.get(CorpusRow, corpus_id)
            if row is None:
                return None

            pipeline_state = dict(row.pipeline_state or {})
            pipeline_state[stage] = state
            row.pipeline_state = pipeline_state
            row.updated_at = datetime.utcnow()

            session.merge(row)

            return Corpus(
                corpus_id=row.corpus_id,
                name=row.name,
                description=row.description,
                created_at=row.created_at,
                updated_at=row.updated_at,
                pipeline_state=row.pipeline_state,
            )

    def get_pipeline_state(self, corpus_id: str, stage: str) -> Optional[dict]:
        """Get the pipeline state for a specific stage.

        Args:
            corpus_id: The corpus identifier.
            stage: Pipeline stage name.

        Returns:
            State dict if found, None otherwise.
        """
        corpus = self.get_corpus(corpus_id)
        if corpus is None:
            return None
        return corpus.pipeline_state.get(stage)

