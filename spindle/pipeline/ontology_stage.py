"""Stage 5: Ontology generation.

Generates a domain ontology informed by the controlled vocabulary,
taxonomy, and thesaurus built in previous stages.
"""

from __future__ import annotations

import json
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
    Ontology,
    EntityType,
    RelationType,
    VocabularyTerm as BAMLVocabularyTerm,
    ThesaurusEntry as BAMLThesaurusEntry,
)
from spindle.extraction.recommender import OntologyRecommender
from spindle.ingestion.storage.catalog import Base
from spindle.ingestion.storage.corpus import CorpusManager
from spindle.pipeline.base import BasePipelineStage
from spindle.pipeline.types import (
    PipelineStage,
    PipelineStageResult,
    PipelineState,
    ThesaurusEntry,
    VocabularyTerm,
)
from spindle.pipeline.vocabulary import VocabularyStage
from spindle.pipeline.thesaurus import ThesaurusStage


class OntologyRow(Base):
    """SQLAlchemy model for ontology storage."""

    __tablename__ = "corpus_ontologies"

    ontology_id: Mapped[str] = mapped_column(String, primary_key=True)
    corpus_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    scope: Mapped[str] = mapped_column(String, default="balanced")
    entity_types: Mapped[list] = mapped_column(JSON, default=list)
    relation_types: Mapped[list] = mapped_column(JSON, default=list)
    text_purpose: Mapped[str | None] = mapped_column(String, nullable=True)
    reasoning: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class OntologyStage(BasePipelineStage[Ontology]):
    """Stage 5: Ontology generation.

    Generates a domain ontology that:
    - Uses vocabulary terms to inform entity types
    - Uses thesaurus relationships to inform relation types
    - Integrates with existing OntologyRecommender
    - Supports minimal/balanced/comprehensive scope levels

    The ontology defines the schema for knowledge graph extraction.
    """

    stage = PipelineStage.ONTOLOGY

    def __init__(
        self,
        corpus_manager: CorpusManager,
        graph_store: Optional[Any] = None,
        scope: str = "balanced",
    ) -> None:
        """Initialize ontology stage.

        Args:
            corpus_manager: CorpusManager for corpus data access.
            graph_store: Optional GraphStore for graph operations.
            scope: Ontology scope level (minimal, balanced, comprehensive).
        """
        super().__init__(corpus_manager, graph_store)
        Base.metadata.create_all(self.corpus_manager._catalog._engine)
        self._vocabulary_stage = VocabularyStage(corpus_manager, graph_store)
        self._thesaurus_stage = ThesaurusStage(corpus_manager, graph_store)
        self._recommender = OntologyRecommender()
        self.scope = scope

    @observe(as_type="generation", capture_input=False, capture_output=False)
    def extract_from_text(
        self,
        text: str,
        document_id: str,
        existing_artifacts: List[Ontology],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Ontology]:
        """Generate ontology from text with vocabulary/thesaurus context.

        This method uses the BAML EnhanceOntologyFromPipeline function
        to generate an ontology informed by previous pipeline stages.

        Args:
            text: Sample text for context.
            document_id: Source document ID.
            existing_artifacts: Previously generated ontologies.
            context: Must contain 'vocabulary' and 'thesaurus_entries'.

        Returns:
            List containing one Ontology object.
        """
        if not context:
            # Fallback to basic recommendation
            recommendation = self._recommender.recommend(text, scope=self.scope)
            return [recommendation.ontology]

        vocabulary: List[VocabularyTerm] = context.get("vocabulary", [])
        thesaurus_entries: List[ThesaurusEntry] = context.get("thesaurus_entries", [])

        if not vocabulary:
            # Fallback to basic recommendation
            recommendation = self._recommender.recommend(text, scope=self.scope)
            return [recommendation.ontology]

        # Convert to BAML format
        baml_vocab = [
            BAMLVocabularyTerm(
                term_id=t.term_id,
                preferred_label=t.preferred_label,
                definition=t.definition,
                synonyms=list(t.synonyms),
                domain=t.domain,
            )
            for t in vocabulary
        ]

        baml_thesaurus = [
            BAMLThesaurusEntry(
                entry_id=e.entry_id,
                term_id=e.term_id,
                preferred_label=e.preferred_label,
                use_for=list(e.use_for),
                broader_terms=list(e.broader_terms),
                narrower_terms=list(e.narrower_terms),
                related_terms=list(e.related_terms),
                scope_note=e.scope_note,
            )
            for e in thesaurus_entries
        ]

        # Call BAML enhanced ontology generation with collector
        collector = baml_py.baml_py.Collector("ontology-enhancement-collector")
        result = b.with_options(collector=collector).EnhanceOntologyFromPipeline(
            text=text[:3000],  # Sample text
            vocabulary=baml_vocab,
            thesaurus_entries=baml_thesaurus,
            scope=self.scope,
        )

        # Extract model from collector
        model = _extract_model_from_collector(collector) or "CustomFast"

        # Update Langfuse generation
        langfuse = get_langfuse_client()
        langfuse.update_current_generation(
            name="EnhanceOntologyFromPipeline",
            model=model,
            input={
                "text": text[:3000],
                "scope": self.scope,
                "vocabulary_terms": [v.preferred_label for v in baml_vocab],
                "thesaurus_entry_count": len(baml_thesaurus),
            },
            output={
                "entity_types": [
                    {"name": et.name, "description": et.description}
                    for et in result.entity_types
                ],
                "relation_types": [
                    {"name": rt.name, "domain": rt.domain, "range": rt.range}
                    for rt in result.relation_types
                ],
            },
        )

        # Build Ontology from result
        ontology = Ontology(
            entity_types=list(result.entity_types),
            relation_types=list(result.relation_types),
        )

        return [ontology]

    def merge_artifacts(
        self,
        artifact_sets: List[List[Ontology]],
    ) -> List[Ontology]:
        """Merge multiple ontologies into one.

        Args:
            artifact_sets: List of ontology lists to merge.

        Returns:
            List containing one merged Ontology.
        """
        all_entity_types: Dict[str, EntityType] = {}
        all_relation_types: Dict[str, RelationType] = {}

        for ontology_set in artifact_sets:
            for ontology in ontology_set:
                for et in ontology.entity_types:
                    key = et.name.lower()
                    if key not in all_entity_types:
                        all_entity_types[key] = et
                    else:
                        # Merge attributes
                        existing = all_entity_types[key]
                        existing_attr_names = {a.name for a in existing.attributes}
                        for attr in et.attributes:
                            if attr.name not in existing_attr_names:
                                existing.attributes.append(attr)

                for rt in ontology.relation_types:
                    key = rt.name.lower()
                    if key not in all_relation_types:
                        all_relation_types[key] = rt

        merged = Ontology(
            entity_types=list(all_entity_types.values()),
            relation_types=list(all_relation_types.values()),
        )

        return [merged]

    def persist_artifacts(
        self,
        corpus_id: str,
        artifacts: List[Ontology],
    ) -> int:
        """Persist ontology to SQLite.

        Args:
            corpus_id: The corpus identifier.
            artifacts: Ontologies to persist.

        Returns:
            Number of ontologies persisted.
        """
        if not artifacts:
            return 0

        ontology = artifacts[0]  # Single ontology per corpus

        with self.corpus_manager._catalog.session() as session:
            ontology_id = f"onto_{corpus_id}"

            # Serialize entity and relation types
            entity_types_data = [
                {
                    "name": et.name,
                    "description": et.description,
                    "attributes": [
                        {
                            "name": a.name,
                            "type": a.type,
                            "description": a.description,
                        }
                        for a in et.attributes
                    ],
                }
                for et in ontology.entity_types
            ]

            relation_types_data = [
                {
                    "name": rt.name,
                    "description": rt.description,
                    "domain": rt.domain,
                    "range": rt.range,
                }
                for rt in ontology.relation_types
            ]

            session.merge(
                OntologyRow(
                    ontology_id=ontology_id,
                    corpus_id=corpus_id,
                    scope=self.scope,
                    entity_types=entity_types_data,
                    relation_types=relation_types_data,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                )
            )

            return 1

    def load_artifacts(self, corpus_id: str) -> List[Ontology]:
        """Load ontology for a corpus.

        Args:
            corpus_id: The corpus identifier.

        Returns:
            List containing the corpus Ontology (or empty list).
        """
        with self.corpus_manager._catalog.session() as session:
            row = (
                session.execute(
                    select(OntologyRow).where(OntologyRow.corpus_id == corpus_id)
                )
                .scalars()
                .first()
            )

            if row is None:
                return []

            # Deserialize entity types
            entity_types = []
            for et_data in row.entity_types or []:
                from spindle.baml_client.types import AttributeDefinition

                attributes = [
                    AttributeDefinition(
                        name=a["name"],
                        type=a["type"],
                        description=a["description"],
                    )
                    for a in et_data.get("attributes", [])
                ]

                entity_types.append(
                    EntityType(
                        name=et_data["name"],
                        description=et_data["description"],
                        attributes=attributes,
                    )
                )

            # Deserialize relation types
            relation_types = [
                RelationType(
                    name=rt_data["name"],
                    description=rt_data["description"],
                    domain=rt_data["domain"],
                    range=rt_data["range"],
                )
                for rt_data in row.relation_types or []
            ]

            return [
                Ontology(
                    entity_types=entity_types,
                    relation_types=relation_types,
                )
            ]

    def get_previous_stage_context(
        self,
        corpus_id: str,
        pipeline_state: PipelineState,
    ) -> Dict[str, Any]:
        """Get vocabulary and thesaurus from previous stages."""
        vocabulary = self._vocabulary_stage.load_artifacts(corpus_id)
        thesaurus_entries = self._thesaurus_stage.load_artifacts(corpus_id)

        return {
            "vocabulary": vocabulary,
            "thesaurus_entries": thesaurus_entries,
        }

    def get_ontology(self, corpus_id: str) -> Optional[Ontology]:
        """Get the ontology for a corpus.

        Args:
            corpus_id: The corpus identifier.

        Returns:
            Ontology if exists, None otherwise.
        """
        ontologies = self.load_artifacts(corpus_id)
        return ontologies[0] if ontologies else None

    def export_ontology_json(self, corpus_id: str) -> Optional[str]:
        """Export ontology as JSON.

        Args:
            corpus_id: The corpus identifier.

        Returns:
            JSON string representation of the ontology.
        """
        ontology = self.get_ontology(corpus_id)
        if not ontology:
            return None

        data = {
            "entity_types": [
                {
                    "name": et.name,
                    "description": et.description,
                    "attributes": [
                        {
                            "name": a.name,
                            "type": a.type,
                            "description": a.description,
                        }
                        for a in et.attributes
                    ],
                }
                for et in ontology.entity_types
            ],
            "relation_types": [
                {
                    "name": rt.name,
                    "description": rt.description,
                    "domain": rt.domain,
                    "range": rt.range,
                }
                for rt in ontology.relation_types
            ],
        }

        return json.dumps(data, indent=2)

    def use_existing_recommender(
        self,
        corpus_id: str,
        text: str,
    ) -> Ontology:
        """Use the existing OntologyRecommender for ontology generation.

        This is an alternative to the pipeline-enhanced approach,
        useful when previous stages haven't been run.

        Args:
            corpus_id: The corpus identifier.
            text: Text to analyze.

        Returns:
            Generated Ontology.
        """
        recommendation = self._recommender.recommend(text, scope=self.scope)
        return recommendation.ontology

