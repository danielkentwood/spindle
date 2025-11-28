"""Stage 1: Controlled Vocabulary extraction.

Extracts clean, disambiguated vocabulary terms with definitions
from corpus documents. This is the foundation of the Ontology Pipeline.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import JSON, DateTime, String, Integer, select
from sqlalchemy.orm import Mapped, mapped_column

from spindle.baml_client import b
from spindle.baml_client.types import VocabularyTerm as BAMLVocabularyTerm
from spindle.ingestion.storage.catalog import Base
from spindle.ingestion.storage.corpus import CorpusManager
from spindle.pipeline.base import BasePipelineStage
from spindle.pipeline.types import (
    PipelineStage,
    PipelineState,
    VocabularyTerm,
)


class VocabularyTermRow(Base):
    """SQLAlchemy model for vocabulary term storage."""

    __tablename__ = "vocabulary_terms"

    term_id: Mapped[str] = mapped_column(String, primary_key=True)
    corpus_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    preferred_label: Mapped[str] = mapped_column(String, nullable=False)
    definition: Mapped[str] = mapped_column(String, nullable=False)
    synonyms: Mapped[list] = mapped_column(JSON, default=list)
    domain: Mapped[str | None] = mapped_column(String, nullable=True)
    source_document_ids: Mapped[list] = mapped_column(JSON, default=list)
    usage_count: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class VocabularyStage(BasePipelineStage[VocabularyTerm]):
    """Stage 1: Controlled Vocabulary extraction.

    Extracts key terms from corpus documents and builds a controlled
    vocabulary with:
    - Preferred labels (canonical forms)
    - Definitions
    - Synonyms
    - Domain classification

    The vocabulary serves as the foundation for subsequent pipeline
    stages (taxonomy, thesaurus, ontology).
    """

    stage = PipelineStage.VOCABULARY

    def __init__(
        self,
        corpus_manager: CorpusManager,
        graph_store: Optional[Any] = None,
    ) -> None:
        """Initialize vocabulary stage."""
        super().__init__(corpus_manager, graph_store)
        # Ensure vocabulary table exists
        Base.metadata.create_all(self.corpus_manager._catalog._engine)

    def extract_from_text(
        self,
        text: str,
        document_id: str,
        existing_artifacts: List[VocabularyTerm],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[VocabularyTerm]:
        """Extract vocabulary terms from text using BAML.

        Args:
            text: The text to extract from.
            document_id: Source document ID.
            existing_artifacts: Previously extracted terms for context.
            context: Optional additional context.

        Returns:
            List of extracted VocabularyTerm objects.
        """
        # Convert existing terms to BAML format
        existing_baml_terms = [
            BAMLVocabularyTerm(
                term_id=t.term_id,
                preferred_label=t.preferred_label,
                definition=t.definition,
                synonyms=list(t.synonyms),
                domain=t.domain,
            )
            for t in existing_artifacts
        ]

        # Call BAML extraction
        result = b.ExtractControlledVocabulary(
            text=text,
            existing_terms=existing_baml_terms,
            document_id=document_id,
        )

        # Convert BAML results to VocabularyTerm
        terms = []
        for baml_term in result.terms:
            term = VocabularyTerm(
                term_id=baml_term.term_id or f"term_{document_id}_{uuid.uuid4().hex[:8]}",
                preferred_label=baml_term.preferred_label,
                definition=baml_term.definition,
                synonyms=list(baml_term.synonyms) if baml_term.synonyms else [],
                domain=baml_term.domain,
                source_document_ids=[document_id],
                usage_count=1,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            terms.append(term)

        return terms

    def merge_artifacts(
        self,
        artifact_sets: List[List[VocabularyTerm]],
    ) -> List[VocabularyTerm]:
        """Merge and deduplicate vocabulary terms.

        Uses BAML consolidation for intelligent merging, or falls back
        to simple deduplication if consolidation fails.

        Args:
            artifact_sets: List of term lists to merge.

        Returns:
            Merged and deduplicated list of terms.
        """
        # Flatten all terms
        all_terms = []
        for term_set in artifact_sets:
            all_terms.extend(term_set)

        if not all_terms:
            return []

        # Try BAML consolidation
        try:
            baml_term_sets = [
                [
                    BAMLVocabularyTerm(
                        term_id=t.term_id,
                        preferred_label=t.preferred_label,
                        definition=t.definition,
                        synonyms=list(t.synonyms),
                        domain=t.domain,
                    )
                    for t in term_set
                ]
                for term_set in artifact_sets
            ]

            result = b.ConsolidateVocabulary(term_sets=baml_term_sets)

            merged_terms = []
            for i, baml_term in enumerate(result.terms):
                term = VocabularyTerm(
                    term_id=f"term_consolidated_{i}",
                    preferred_label=baml_term.preferred_label,
                    definition=baml_term.definition,
                    synonyms=list(baml_term.synonyms) if baml_term.synonyms else [],
                    domain=baml_term.domain,
                    source_document_ids=[],  # Will be populated during persist
                    usage_count=1,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                )
                merged_terms.append(term)

            return merged_terms

        except Exception:
            # Fallback to simple deduplication
            return self._simple_merge(all_terms)

    def _simple_merge(self, terms: List[VocabularyTerm]) -> List[VocabularyTerm]:
        """Simple merge by preferred label."""
        merged: Dict[str, VocabularyTerm] = {}

        for term in terms:
            key = term.preferred_label.lower().strip()

            if key in merged:
                existing = merged[key]
                # Merge synonyms
                all_synonyms = set(existing.synonyms) | set(term.synonyms)
                existing.synonyms = list(all_synonyms)
                # Update usage count
                existing.usage_count += term.usage_count
                # Merge source documents
                all_docs = set(existing.source_document_ids) | set(term.source_document_ids)
                existing.source_document_ids = list(all_docs)
                existing.updated_at = datetime.utcnow()
            else:
                merged[key] = term

        return list(merged.values())

    def persist_artifacts(
        self,
        corpus_id: str,
        artifacts: List[VocabularyTerm],
    ) -> int:
        """Persist vocabulary terms to SQLite.

        Args:
            corpus_id: The corpus identifier.
            artifacts: Terms to persist.

        Returns:
            Number of terms persisted.
        """
        with self.corpus_manager._catalog.session() as session:
            count = 0
            for term in artifacts:
                # Generate new ID if needed
                if not term.term_id or term.term_id.startswith("term_consolidated"):
                    term.term_id = f"term_{corpus_id}_{uuid.uuid4().hex[:8]}"

                session.merge(
                    VocabularyTermRow(
                        term_id=term.term_id,
                        corpus_id=corpus_id,
                        preferred_label=term.preferred_label,
                        definition=term.definition,
                        synonyms=term.synonyms,
                        domain=term.domain,
                        source_document_ids=term.source_document_ids,
                        usage_count=term.usage_count,
                        created_at=term.created_at,
                        updated_at=term.updated_at,
                    )
                )
                count += 1

            return count

    def load_artifacts(self, corpus_id: str) -> List[VocabularyTerm]:
        """Load vocabulary terms for a corpus.

        Args:
            corpus_id: The corpus identifier.

        Returns:
            List of VocabularyTerm objects.
        """
        with self.corpus_manager._catalog.session() as session:
            rows = (
                session.execute(
                    select(VocabularyTermRow).where(
                        VocabularyTermRow.corpus_id == corpus_id
                    )
                )
                .scalars()
                .all()
            )

            return [
                VocabularyTerm(
                    term_id=row.term_id,
                    preferred_label=row.preferred_label,
                    definition=row.definition,
                    synonyms=row.synonyms or [],
                    domain=row.domain,
                    source_document_ids=row.source_document_ids or [],
                    usage_count=row.usage_count,
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                )
                for row in rows
            ]

    def get_term_by_label(
        self,
        corpus_id: str,
        label: str,
    ) -> Optional[VocabularyTerm]:
        """Find a term by its preferred label.

        Args:
            corpus_id: The corpus identifier.
            label: The preferred label to search for.

        Returns:
            VocabularyTerm if found, None otherwise.
        """
        with self.corpus_manager._catalog.session() as session:
            row = (
                session.execute(
                    select(VocabularyTermRow).where(
                        (VocabularyTermRow.corpus_id == corpus_id)
                        & (VocabularyTermRow.preferred_label == label)
                    )
                )
                .scalars()
                .first()
            )

            if row is None:
                return None

            return VocabularyTerm(
                term_id=row.term_id,
                preferred_label=row.preferred_label,
                definition=row.definition,
                synonyms=row.synonyms or [],
                domain=row.domain,
                source_document_ids=row.source_document_ids or [],
                usage_count=row.usage_count,
                created_at=row.created_at,
                updated_at=row.updated_at,
            )

    def search_terms(
        self,
        corpus_id: str,
        query: str,
        include_synonyms: bool = True,
    ) -> List[VocabularyTerm]:
        """Search vocabulary terms by query string.

        Args:
            corpus_id: The corpus identifier.
            query: Search query.
            include_synonyms: Whether to search in synonyms too.

        Returns:
            List of matching VocabularyTerm objects.
        """
        query_lower = query.lower()
        all_terms = self.load_artifacts(corpus_id)

        matches = []
        for term in all_terms:
            if query_lower in term.preferred_label.lower():
                matches.append(term)
            elif query_lower in term.definition.lower():
                matches.append(term)
            elif include_synonyms:
                for syn in term.synonyms:
                    if query_lower in syn.lower():
                        matches.append(term)
                        break

        return matches

