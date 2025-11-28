"""Stage 3: Taxonomy extraction.

Builds hierarchical parent-child relationships between vocabulary terms
to create a taxonomic structure.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import JSON, DateTime, String, Integer, select
from sqlalchemy.orm import Mapped, mapped_column

from spindle.baml_client import b
from spindle.baml_client.types import (
    TaxonomyRelation as BAMLTaxonomyRelation,
    VocabularyTerm as BAMLVocabularyTerm,
)
from spindle.ingestion.storage.catalog import Base
from spindle.ingestion.storage.corpus import CorpusManager
from spindle.pipeline.base import BasePipelineStage
from spindle.pipeline.types import (
    PipelineStage,
    PipelineState,
    TaxonomyNode,
    TaxonomyRelation,
    VocabularyTerm,
)
from spindle.pipeline.vocabulary import VocabularyStage


class TaxonomyNodeRow(Base):
    """SQLAlchemy model for taxonomy node storage."""

    __tablename__ = "taxonomy_nodes"

    node_id: Mapped[str] = mapped_column(String, primary_key=True)
    corpus_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    term_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    label: Mapped[str] = mapped_column(String, nullable=False)
    level: Mapped[int] = mapped_column(Integer, default=0)
    parent_node_id: Mapped[str | None] = mapped_column(String, nullable=True)
    child_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class TaxonomyRelationRow(Base):
    """SQLAlchemy model for taxonomy relation storage."""

    __tablename__ = "taxonomy_relations"

    relation_id: Mapped[str] = mapped_column(String, primary_key=True)
    corpus_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    parent_node_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    child_node_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    relation_type: Mapped[str] = mapped_column(String, default="broader")
    confidence: Mapped[str] = mapped_column(String, default="medium")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class TaxonomyStage(BasePipelineStage[TaxonomyNode]):
    """Stage 3: Taxonomy extraction.

    Builds hierarchical relationships between vocabulary terms:
    - Parent-child (broader/narrower) relationships
    - Tree structure with root terms
    - Level assignments for hierarchy depth

    The taxonomy provides the structural foundation for the thesaurus.
    """

    stage = PipelineStage.TAXONOMY

    def __init__(
        self,
        corpus_manager: CorpusManager,
        graph_store: Optional[Any] = None,
    ) -> None:
        """Initialize taxonomy stage."""
        super().__init__(corpus_manager, graph_store)
        Base.metadata.create_all(self.corpus_manager._catalog._engine)
        self._vocabulary_stage = VocabularyStage(corpus_manager, graph_store)

    def extract_from_text(
        self,
        text: str,
        document_id: str,
        existing_artifacts: List[TaxonomyNode],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[TaxonomyNode]:
        """Extract taxonomy relationships from text.

        Args:
            text: The text to analyze.
            document_id: Source document ID.
            existing_artifacts: Previously extracted nodes.
            context: Must contain 'vocabulary' key with VocabularyTerm list.

        Returns:
            List of TaxonomyNode objects.
        """
        if not context or "vocabulary" not in context:
            return []

        vocabulary: List[VocabularyTerm] = context["vocabulary"]
        if not vocabulary:
            return []

        # Convert vocabulary to BAML format
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

        # Convert existing relations
        existing_relations = context.get("existing_relations", [])
        baml_relations = [
            BAMLTaxonomyRelation(
                parent_term=r.parent_node_id,
                child_term=r.child_node_id,
                relation_type=r.relation_type,
                confidence="medium",
            )
            for r in existing_relations
        ]

        # Call BAML extraction
        result = b.ExtractTaxonomy(
            terms=baml_terms,
            text=text,
            existing_relations=baml_relations,
        )

        # Build node map from vocabulary
        nodes: Dict[str, TaxonomyNode] = {}
        for term in vocabulary:
            node_id = f"taxon_{term.term_id}"
            nodes[term.preferred_label] = TaxonomyNode(
                node_id=node_id,
                term_id=term.term_id,
                label=term.preferred_label,
                level=0,
                parent_node_id=None,
                child_count=0,
                created_at=datetime.utcnow(),
            )

        # Apply relations from BAML
        for rel in result.relations:
            parent_label = rel.parent_term
            child_label = rel.child_term

            if parent_label in nodes and child_label in nodes:
                child_node = nodes[child_label]
                parent_node = nodes[parent_label]

                child_node.parent_node_id = parent_node.node_id
                parent_node.child_count += 1

        # Calculate levels
        self._calculate_levels(nodes)

        return list(nodes.values())

    def _calculate_levels(self, nodes: Dict[str, TaxonomyNode]) -> None:
        """Calculate hierarchy levels for nodes."""
        # Build parent lookup
        id_to_node = {n.node_id: n for n in nodes.values()}

        def get_level(node: TaxonomyNode, visited: set) -> int:
            if node.node_id in visited:
                return 0  # Cycle detected
            if node.parent_node_id is None:
                return 0
            visited.add(node.node_id)
            parent = id_to_node.get(node.parent_node_id)
            if parent:
                return 1 + get_level(parent, visited)
            return 0

        for node in nodes.values():
            node.level = get_level(node, set())

    def merge_artifacts(
        self,
        artifact_sets: List[List[TaxonomyNode]],
    ) -> List[TaxonomyNode]:
        """Merge taxonomy nodes from multiple extractions.

        Args:
            artifact_sets: List of node lists to merge.

        Returns:
            Merged list of nodes.
        """
        # Merge by label
        merged: Dict[str, TaxonomyNode] = {}

        for node_set in artifact_sets:
            for node in node_set:
                key = node.label.lower().strip()
                if key not in merged:
                    merged[key] = node
                else:
                    # Keep the one with more information
                    existing = merged[key]
                    if node.parent_node_id and not existing.parent_node_id:
                        merged[key] = node

        return list(merged.values())

    def persist_artifacts(
        self,
        corpus_id: str,
        artifacts: List[TaxonomyNode],
    ) -> int:
        """Persist taxonomy nodes and relations to SQLite.

        Args:
            corpus_id: The corpus identifier.
            artifacts: Nodes to persist.

        Returns:
            Number of nodes persisted.
        """
        with self.corpus_manager._catalog.session() as session:
            count = 0

            # First pass: persist nodes
            for node in artifacts:
                session.merge(
                    TaxonomyNodeRow(
                        node_id=node.node_id,
                        corpus_id=corpus_id,
                        term_id=node.term_id,
                        label=node.label,
                        level=node.level,
                        parent_node_id=node.parent_node_id,
                        child_count=node.child_count,
                        created_at=node.created_at,
                    )
                )
                count += 1

            # Second pass: persist relations
            for node in artifacts:
                if node.parent_node_id:
                    relation_id = f"rel_{node.parent_node_id}_{node.node_id}"
                    session.merge(
                        TaxonomyRelationRow(
                            relation_id=relation_id,
                            corpus_id=corpus_id,
                            parent_node_id=node.parent_node_id,
                            child_node_id=node.node_id,
                            relation_type="broader",
                            confidence="medium",
                            created_at=datetime.utcnow(),
                        )
                    )

            # Also store in graph if available
            if self.graph_store:
                self._persist_to_graph(corpus_id, artifacts)

            return count

    def _persist_to_graph(
        self,
        corpus_id: str,
        nodes: List[TaxonomyNode],
    ) -> None:
        """Persist taxonomy to graph store as BROADER_THAN edges."""
        for node in nodes:
            # Add node
            self.graph_store.add_node(
                name=node.label,
                entity_type="TaxonomyTerm",
                metadata={
                    "corpus_id": corpus_id,
                    "term_id": node.term_id,
                    "level": node.level,
                },
                description=f"Taxonomy term at level {node.level}",
            )

        # Add edges
        id_to_node = {n.node_id: n for n in nodes}
        for node in nodes:
            if node.parent_node_id and node.parent_node_id in id_to_node:
                parent = id_to_node[node.parent_node_id]
                self.graph_store.add_edge(
                    subject=parent.label,
                    predicate="BROADER_THAN",
                    obj=node.label,
                    metadata={"corpus_id": corpus_id},
                )

    def load_artifacts(self, corpus_id: str) -> List[TaxonomyNode]:
        """Load taxonomy nodes for a corpus.

        Args:
            corpus_id: The corpus identifier.

        Returns:
            List of TaxonomyNode objects.
        """
        with self.corpus_manager._catalog.session() as session:
            rows = (
                session.execute(
                    select(TaxonomyNodeRow).where(
                        TaxonomyNodeRow.corpus_id == corpus_id
                    )
                )
                .scalars()
                .all()
            )

            return [
                TaxonomyNode(
                    node_id=row.node_id,
                    term_id=row.term_id,
                    label=row.label,
                    level=row.level,
                    parent_node_id=row.parent_node_id,
                    child_count=row.child_count,
                    created_at=row.created_at,
                )
                for row in rows
            ]

    def load_relations(self, corpus_id: str) -> List[TaxonomyRelation]:
        """Load taxonomy relations for a corpus.

        Args:
            corpus_id: The corpus identifier.

        Returns:
            List of TaxonomyRelation objects.
        """
        with self.corpus_manager._catalog.session() as session:
            rows = (
                session.execute(
                    select(TaxonomyRelationRow).where(
                        TaxonomyRelationRow.corpus_id == corpus_id
                    )
                )
                .scalars()
                .all()
            )

            return [
                TaxonomyRelation(
                    relation_id=row.relation_id,
                    parent_node_id=row.parent_node_id,
                    child_node_id=row.child_node_id,
                    relation_type=row.relation_type,
                    created_at=row.created_at,
                )
                for row in rows
            ]

    def get_previous_stage_context(
        self,
        corpus_id: str,
        pipeline_state: PipelineState,
    ) -> Dict[str, Any]:
        """Get vocabulary from Stage 1."""
        vocabulary = self._vocabulary_stage.load_artifacts(corpus_id)
        existing_relations = self.load_relations(corpus_id)

        return {
            "vocabulary": vocabulary,
            "existing_relations": existing_relations,
        }

    def get_root_terms(self, corpus_id: str) -> List[TaxonomyNode]:
        """Get root terms (nodes with no parent).

        Args:
            corpus_id: The corpus identifier.

        Returns:
            List of root TaxonomyNode objects.
        """
        nodes = self.load_artifacts(corpus_id)
        return [n for n in nodes if n.parent_node_id is None]

    def get_children(self, corpus_id: str, parent_node_id: str) -> List[TaxonomyNode]:
        """Get child nodes of a parent.

        Args:
            corpus_id: The corpus identifier.
            parent_node_id: The parent node ID.

        Returns:
            List of child TaxonomyNode objects.
        """
        nodes = self.load_artifacts(corpus_id)
        return [n for n in nodes if n.parent_node_id == parent_node_id]

