"""Stage 4: Thesaurus extraction.

Builds semantic relationships between vocabulary terms following
ISO 25964 / SKOS vocabulary standards.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import JSON, DateTime, String, select
from sqlalchemy.orm import Mapped, mapped_column
from langfuse import observe, get_client as get_langfuse_client
import baml_py

from spindle.baml_client import b
from spindle.extraction.helpers import _extract_model_from_collector
from spindle.baml_client.types import (
    ThesaurusEntry as BAMLThesaurusEntry,
    TaxonomyRelation as BAMLTaxonomyRelation,
    VocabularyTerm as BAMLVocabularyTerm,
)
from spindle.ingestion.storage.catalog import Base
from spindle.ingestion.storage.corpus import CorpusManager
from spindle.pipeline.base import BasePipelineStage
from spindle.pipeline.types import (
    PipelineStage,
    PipelineState,
    ThesaurusEntry,
    TaxonomyRelation,
    VocabularyTerm,
)
from spindle.pipeline.vocabulary import VocabularyStage
from spindle.pipeline.taxonomy import TaxonomyStage


class ThesaurusEntryRow(Base):
    """SQLAlchemy model for thesaurus entry storage."""

    __tablename__ = "thesaurus_entries"

    entry_id: Mapped[str] = mapped_column(String, primary_key=True)
    corpus_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    term_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    preferred_label: Mapped[str] = mapped_column(String, nullable=False)
    use_for: Mapped[list] = mapped_column(JSON, default=list)
    broader_terms: Mapped[list] = mapped_column(JSON, default=list)
    narrower_terms: Mapped[list] = mapped_column(JSON, default=list)
    related_terms: Mapped[list] = mapped_column(JSON, default=list)
    scope_note: Mapped[str | None] = mapped_column(String, nullable=True)
    history_note: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ThesaurusStage(BasePipelineStage[ThesaurusEntry]):
    """Stage 4: Thesaurus extraction.

    Extends vocabulary with semantic relationships following ISO 25964:
    - USE / USE_FOR: Preferred vs non-preferred terms
    - BT (Broader Term): Hierarchical parent
    - NT (Narrower Term): Hierarchical child
    - RT (Related Term): Associative relationships
    - SN (Scope Note): Definitions and clarifications

    The thesaurus provides rich semantic context for ontology design.
    """

    stage = PipelineStage.THESAURUS

    def __init__(
        self,
        corpus_manager: CorpusManager,
        graph_store: Optional[Any] = None,
    ) -> None:
        """Initialize thesaurus stage."""
        super().__init__(corpus_manager, graph_store)
        Base.metadata.create_all(self.corpus_manager._catalog._engine)
        self._vocabulary_stage = VocabularyStage(corpus_manager, graph_store)
        self._taxonomy_stage = TaxonomyStage(corpus_manager, graph_store)

    @observe(as_type="generation", capture_input=False, capture_output=False)
    def extract_from_text(
        self,
        text: str,
        document_id: str,
        existing_artifacts: List[ThesaurusEntry],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ThesaurusEntry]:
        """Extract thesaurus entries with semantic relationships.

        Args:
            text: The text to analyze.
            document_id: Source document ID.
            existing_artifacts: Previously extracted entries.
            context: Must contain 'vocabulary' and 'taxonomy_relations'.

        Returns:
            List of ThesaurusEntry objects.
        """
        if not context:
            return []

        vocabulary: List[VocabularyTerm] = context.get("vocabulary", [])
        taxonomy_relations: List[TaxonomyRelation] = context.get("taxonomy_relations", [])

        if not vocabulary:
            return []

        # Convert to BAML format
        baml_terms = [
            BAMLVocabularyTerm(
                term_id=t.term_id,
                preferred_label=t.preferred_label,
                definition=t.definition,
                synonyms=list(t.synonyms),
                domain=t.domain,
            )
            for t in vocabulary
        ]

        baml_relations = [
            BAMLTaxonomyRelation(
                parent_term=r.parent_node_id,
                child_term=r.child_node_id,
                relation_type=r.relation_type,
                confidence="medium",
            )
            for r in taxonomy_relations
        ]

        # Call BAML extraction with collector
        collector = baml_py.baml_py.Collector("thesaurus-extraction-collector")
        result = b.with_options(collector=collector).ExtractThesaurus(
            terms=baml_terms,
            taxonomy_relations=baml_relations,
            text=text,
        )

        # Extract model from collector
        model = _extract_model_from_collector(collector) or "CustomFast"

        # Update Langfuse generation
        langfuse = get_langfuse_client()
        langfuse.update_current_generation(
            name="ExtractThesaurus",
            model=model,
            input={
                "text": text,
                "terms": [t.preferred_label for t in baml_terms],
                "taxonomy_relations": [
                    {"parent": r.parent_term, "child": r.child_term}
                    for r in baml_relations
                ],
            },
            output={
                "entries": [
                    {
                        "preferred_label": e.preferred_label,
                        "use_for": list(e.use_for) if e.use_for else [],
                        "broader_terms": list(e.broader_terms) if e.broader_terms else [],
                        "narrower_terms": list(e.narrower_terms) if e.narrower_terms else [],
                        "related_terms": list(e.related_terms) if e.related_terms else [],
                        "scope_note": e.scope_note,
                    }
                    for e in result.entries
                ],
            },
        )

        # Convert to ThesaurusEntry
        entries = []
        for baml_entry in result.entries:
            entry = ThesaurusEntry(
                entry_id=baml_entry.entry_id or f"thes_{uuid.uuid4().hex[:8]}",
                term_id=baml_entry.term_id,
                preferred_label=baml_entry.preferred_label,
                use_for=list(baml_entry.use_for) if baml_entry.use_for else [],
                broader_terms=list(baml_entry.broader_terms) if baml_entry.broader_terms else [],
                narrower_terms=list(baml_entry.narrower_terms) if baml_entry.narrower_terms else [],
                related_terms=list(baml_entry.related_terms) if baml_entry.related_terms else [],
                scope_note=baml_entry.scope_note,
                history_note=None,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            entries.append(entry)

        return entries

    def merge_artifacts(
        self,
        artifact_sets: List[List[ThesaurusEntry]],
    ) -> List[ThesaurusEntry]:
        """Merge thesaurus entries from multiple extractions.

        Args:
            artifact_sets: List of entry lists to merge.

        Returns:
            Merged list of entries.
        """
        merged: Dict[str, ThesaurusEntry] = {}

        for entry_set in artifact_sets:
            for entry in entry_set:
                key = entry.preferred_label.lower().strip()

                if key in merged:
                    existing = merged[key]
                    # Merge relationships
                    existing.use_for = list(set(existing.use_for) | set(entry.use_for))
                    existing.broader_terms = list(set(existing.broader_terms) | set(entry.broader_terms))
                    existing.narrower_terms = list(set(existing.narrower_terms) | set(entry.narrower_terms))
                    existing.related_terms = list(set(existing.related_terms) | set(entry.related_terms))
                    existing.updated_at = datetime.utcnow()
                else:
                    merged[key] = entry

        return list(merged.values())

    def persist_artifacts(
        self,
        corpus_id: str,
        artifacts: List[ThesaurusEntry],
    ) -> int:
        """Persist thesaurus entries to SQLite and optionally graph.

        Args:
            corpus_id: The corpus identifier.
            artifacts: Entries to persist.

        Returns:
            Number of entries persisted.
        """
        with self.corpus_manager._catalog.session() as session:
            count = 0

            for entry in artifacts:
                if not entry.entry_id:
                    entry.entry_id = f"thes_{corpus_id}_{uuid.uuid4().hex[:8]}"

                session.merge(
                    ThesaurusEntryRow(
                        entry_id=entry.entry_id,
                        corpus_id=corpus_id,
                        term_id=entry.term_id,
                        preferred_label=entry.preferred_label,
                        use_for=entry.use_for,
                        broader_terms=entry.broader_terms,
                        narrower_terms=entry.narrower_terms,
                        related_terms=entry.related_terms,
                        scope_note=entry.scope_note,
                        history_note=entry.history_note,
                        created_at=entry.created_at,
                        updated_at=entry.updated_at,
                    )
                )
                count += 1

            # Also store in graph if available
            if self.graph_store:
                self._persist_to_graph(corpus_id, artifacts)

            return count

    def _persist_to_graph(
        self,
        corpus_id: str,
        entries: List[ThesaurusEntry],
    ) -> None:
        """Persist thesaurus relationships to graph store."""
        for entry in entries:
            # Add RT (related term) edges - not covered by taxonomy
            for related in entry.related_terms:
                self.graph_store.add_edge(
                    subject=entry.preferred_label,
                    predicate="RELATED_TO",
                    obj=related,
                    metadata={
                        "corpus_id": corpus_id,
                        "relation_type": "RT",
                    },
                )

            # Add USE_FOR edges
            for non_preferred in entry.use_for:
                self.graph_store.add_edge(
                    subject=entry.preferred_label,
                    predicate="USE_FOR",
                    obj=non_preferred,
                    metadata={
                        "corpus_id": corpus_id,
                        "relation_type": "UF",
                    },
                )

    def load_artifacts(self, corpus_id: str) -> List[ThesaurusEntry]:
        """Load thesaurus entries for a corpus.

        Args:
            corpus_id: The corpus identifier.

        Returns:
            List of ThesaurusEntry objects.
        """
        with self.corpus_manager._catalog.session() as session:
            rows = (
                session.execute(
                    select(ThesaurusEntryRow).where(
                        ThesaurusEntryRow.corpus_id == corpus_id
                    )
                )
                .scalars()
                .all()
            )

            return [
                ThesaurusEntry(
                    entry_id=row.entry_id,
                    term_id=row.term_id,
                    preferred_label=row.preferred_label,
                    use_for=row.use_for or [],
                    broader_terms=row.broader_terms or [],
                    narrower_terms=row.narrower_terms or [],
                    related_terms=row.related_terms or [],
                    scope_note=row.scope_note,
                    history_note=row.history_note,
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                )
                for row in rows
            ]

    def get_previous_stage_context(
        self,
        corpus_id: str,
        pipeline_state: PipelineState,
    ) -> Dict[str, Any]:
        """Get vocabulary and taxonomy from previous stages."""
        vocabulary = self._vocabulary_stage.load_artifacts(corpus_id)
        taxonomy_relations = self._taxonomy_stage.load_relations(corpus_id)

        return {
            "vocabulary": vocabulary,
            "taxonomy_relations": taxonomy_relations,
        }

    def get_entry_by_label(
        self,
        corpus_id: str,
        label: str,
    ) -> Optional[ThesaurusEntry]:
        """Find an entry by preferred label.

        Args:
            corpus_id: The corpus identifier.
            label: The preferred label to search for.

        Returns:
            ThesaurusEntry if found, None otherwise.
        """
        entries = self.load_artifacts(corpus_id)
        for entry in entries:
            if entry.preferred_label.lower() == label.lower():
                return entry
        return None

    def get_related_terms(
        self,
        corpus_id: str,
        label: str,
    ) -> List[str]:
        """Get all related terms for a concept.

        Args:
            corpus_id: The corpus identifier.
            label: The concept label.

        Returns:
            List of related term labels.
        """
        entry = self.get_entry_by_label(corpus_id, label)
        if not entry:
            return []

        related = set(entry.related_terms)
        related |= set(entry.broader_terms)
        related |= set(entry.narrower_terms)
        related |= set(entry.use_for)

        return list(related)

    def export_skos(self, corpus_id: str) -> str:
        """Export thesaurus as SKOS-like representation.

        Args:
            corpus_id: The corpus identifier.

        Returns:
            SKOS-formatted string representation.
        """
        entries = self.load_artifacts(corpus_id)
        lines = []

        for entry in entries:
            lines.append(f"skos:Concept <{entry.preferred_label}>")
            lines.append(f"  skos:prefLabel \"{entry.preferred_label}\"")

            if entry.scope_note:
                lines.append(f"  skos:scopeNote \"{entry.scope_note}\"")

            for uf in entry.use_for:
                lines.append(f"  skos:altLabel \"{uf}\"")

            for bt in entry.broader_terms:
                lines.append(f"  skos:broader <{bt}>")

            for nt in entry.narrower_terms:
                lines.append(f"  skos:narrower <{nt}>")

            for rt in entry.related_terms:
                lines.append(f"  skos:related <{rt}>")

            lines.append("")

        return "\n".join(lines)

