"""Stage 6: Knowledge Graph synthesis.

The final stage that synthesizes all previous pipeline stages into
a queryable knowledge graph using SpindleExtractor and GraphStore.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import JSON, DateTime, String, Integer, select
from sqlalchemy.orm import Mapped, mapped_column

from spindle.baml_client.types import Ontology, Triple
from spindle.extraction.extractor import SpindleExtractor
from spindle.graph_store import GraphStore
from spindle.ingestion.storage.catalog import Base
from spindle.ingestion.storage.corpus import CorpusManager
from spindle.ingestion.types import ChunkArtifact
from spindle.pipeline.base import BasePipelineStage
from spindle.pipeline.types import (
    PipelineStage,
    PipelineState,
)
from spindle.pipeline.ontology_stage import OntologyStage


class KnowledgeGraphRunRow(Base):
    """SQLAlchemy model for KG extraction run tracking."""

    __tablename__ = "knowledge_graph_runs"

    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    corpus_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    graph_store_path: Mapped[str] = mapped_column(String, nullable=False)
    triple_count: Mapped[int] = mapped_column(Integer, default=0)
    node_count: Mapped[int] = mapped_column(Integer, default=0)
    edge_count: Mapped[int] = mapped_column(Integer, default=0)
    documents_processed: Mapped[int] = mapped_column(Integer, default=0)
    chunks_processed: Mapped[int] = mapped_column(Integer, default=0)
    extraction_config: Mapped[dict] = mapped_column(JSON, default=dict)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class KnowledgeGraphStage(BasePipelineStage[Triple]):
    """Stage 6: Knowledge Graph synthesis.

    The final pipeline stage that:
    - Uses the generated ontology from Stage 5
    - Extracts triples from all corpus documents using SpindleExtractor
    - Stores results in GraphStore (Kùzu)
    - Integrates vocabulary, taxonomy, thesaurus context

    This produces a queryable knowledge graph that synthesizes
    all the semantic knowledge built through the pipeline.
    """

    stage = PipelineStage.KNOWLEDGE_GRAPH

    def __init__(
        self,
        corpus_manager: CorpusManager,
        graph_store: Optional[GraphStore] = None,
    ) -> None:
        """Initialize knowledge graph stage.

        Args:
            corpus_manager: CorpusManager for corpus data access.
            graph_store: GraphStore for knowledge graph persistence.
        """
        super().__init__(corpus_manager, graph_store)
        Base.metadata.create_all(self.corpus_manager._catalog._engine)
        self._ontology_stage = OntologyStage(corpus_manager, graph_store)
        self._extractor: Optional[SpindleExtractor] = None

    def _ensure_graph_store(self, corpus_id: str) -> GraphStore:
        """Ensure graph store is available.

        Args:
            corpus_id: The corpus identifier.

        Returns:
            GraphStore instance.
        """
        if self.graph_store is not None:
            return self.graph_store

        # Create a new graph store for this corpus
        graph_path = f"corpus_{corpus_id}_kg"
        self.graph_store = GraphStore(graph_path)
        return self.graph_store

    def extract_from_text(
        self,
        text: str,
        document_id: str,
        existing_artifacts: List[Triple],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Triple]:
        """Extract triples from text using SpindleExtractor.

        Args:
            text: The text to extract from.
            document_id: Source document ID.
            existing_artifacts: Previously extracted triples.
            context: Must contain 'ontology'.

        Returns:
            List of extracted Triple objects.
        """
        if not context or "ontology" not in context:
            return []

        ontology: Ontology = context["ontology"]
        source_name = context.get("source_name", document_id)

        # Create or update extractor with ontology
        if self._extractor is None or self._extractor.ontology != ontology:
            self._extractor = SpindleExtractor(ontology=ontology)

        # Extract triples
        result = self._extractor.extract(
            text=text,
            source_name=source_name,
            existing_triples=existing_artifacts,
        )

        return list(result.triples)

    def merge_artifacts(
        self,
        artifact_sets: List[List[Triple]],
    ) -> List[Triple]:
        """Merge triples from multiple extractions.

        For knowledge graphs, we generally keep all triples
        (duplicates from different sources are allowed).

        Args:
            artifact_sets: List of triple lists to merge.

        Returns:
            Combined list of triples.
        """
        all_triples = []
        for triple_set in artifact_sets:
            all_triples.extend(triple_set)
        return all_triples

    def persist_artifacts(
        self,
        corpus_id: str,
        artifacts: List[Triple],
    ) -> int:
        """Persist triples to GraphStore.

        Args:
            corpus_id: The corpus identifier.
            artifacts: Triples to persist.

        Returns:
            Number of triples persisted.
        """
        if not artifacts:
            return 0

        graph_store = self._ensure_graph_store(corpus_id)
        count = graph_store.add_triples(artifacts)

        # Track the run
        self._record_run(corpus_id, graph_store, artifacts)

        return count

    def _record_run(
        self,
        corpus_id: str,
        graph_store: GraphStore,
        triples: List[Triple],
    ) -> None:
        """Record the KG extraction run in SQLite."""
        with self.corpus_manager._catalog.session() as session:
            run_id = f"kg_run_{corpus_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

            stats = graph_store.get_statistics()

            session.merge(
                KnowledgeGraphRunRow(
                    run_id=run_id,
                    corpus_id=corpus_id,
                    graph_store_path=graph_store.db_path,
                    triple_count=len(triples),
                    node_count=stats.get("node_count", 0),
                    edge_count=stats.get("edge_count", 0),
                    documents_processed=len(set(t.source.source_name for t in triples)),
                    chunks_processed=0,  # Will be updated in run()
                    extraction_config={},
                    started_at=datetime.utcnow(),
                    finished_at=datetime.utcnow(),
                )
            )

    def load_artifacts(self, corpus_id: str) -> List[Triple]:
        """Load triples from GraphStore.

        Args:
            corpus_id: The corpus identifier.

        Returns:
            List of Triple objects from the graph.
        """
        graph_store = self._ensure_graph_store(corpus_id)
        return graph_store.get_triples()

    def get_previous_stage_context(
        self,
        corpus_id: str,
        pipeline_state: PipelineState,
    ) -> Dict[str, Any]:
        """Get ontology from Stage 5."""
        ontology = self._ontology_stage.get_ontology(corpus_id)

        if ontology is None:
            return {}

        return {
            "ontology": ontology,
        }

    def run(
        self,
        corpus: Any,  # Corpus type
        pipeline_state: PipelineState,
        strategy: str = "sequential",
    ) -> Any:  # PipelineStageResult
        """Execute knowledge graph extraction.

        This override handles the special requirements of KG extraction:
        - Uses ontology from Stage 5
        - Processes chunks with entity consistency
        - Stores to GraphStore

        Args:
            corpus: The corpus to process.
            pipeline_state: Current pipeline state.
            strategy: Extraction strategy.

        Returns:
            PipelineStageResult with execution details.
        """
        from spindle.pipeline.types import PipelineStageResult

        started_at = datetime.utcnow()

        try:
            # Check prerequisites
            if not pipeline_state.can_run_stage(self.stage):
                return PipelineStageResult(
                    stage=self.stage,
                    corpus_id=corpus.corpus_id,
                    success=False,
                    started_at=started_at,
                    finished_at=datetime.utcnow(),
                    error_message="Prerequisites not met. Complete ontology stage first.",
                )

            # Get ontology
            context = self.get_previous_stage_context(corpus.corpus_id, pipeline_state)
            if "ontology" not in context:
                return PipelineStageResult(
                    stage=self.stage,
                    corpus_id=corpus.corpus_id,
                    success=False,
                    started_at=started_at,
                    finished_at=datetime.utcnow(),
                    error_message="No ontology found. Run ontology stage first.",
                )

            # Get chunks
            chunks = self.get_corpus_chunks(corpus.corpus_id)
            if not chunks:
                return PipelineStageResult(
                    stage=self.stage,
                    corpus_id=corpus.corpus_id,
                    success=False,
                    started_at=started_at,
                    finished_at=datetime.utcnow(),
                    error_message="No chunks found in corpus.",
                )

            # Get document artifacts for source names
            documents = self.corpus_manager.get_document_artifacts(corpus.corpus_id)
            doc_id_to_source = {
                d.document_id: str(d.source_path) for d in documents
            }

            # Extract triples from chunks
            all_triples: List[Triple] = []

            if strategy == "sequential":
                all_triples = self._extract_sequential_kg(
                    chunks, context, doc_id_to_source
                )
            else:
                # For KG, sequential is usually preferred for consistency
                all_triples = self._extract_sequential_kg(
                    chunks, context, doc_id_to_source
                )

            # Persist to graph
            count = self.persist_artifacts(corpus.corpus_id, all_triples)

            return PipelineStageResult(
                stage=self.stage,
                corpus_id=corpus.corpus_id,
                success=True,
                started_at=started_at,
                finished_at=datetime.utcnow(),
                artifact_count=count,
                metrics={
                    "chunks_processed": len(chunks),
                    "triples_extracted": len(all_triples),
                    "strategy": strategy,
                    "graph_path": self.graph_store.db_path if self.graph_store else None,
                },
            )

        except Exception as e:
            return PipelineStageResult(
                stage=self.stage,
                corpus_id=corpus.corpus_id,
                success=False,
                started_at=started_at,
                finished_at=datetime.utcnow(),
                error_message=str(e),
            )

    def _extract_sequential_kg(
        self,
        chunks: List[ChunkArtifact],
        context: Dict[str, Any],
        doc_id_to_source: Dict[str, str],
    ) -> List[Triple]:
        """Extract triples sequentially for entity consistency."""
        all_triples: List[Triple] = []

        for chunk in chunks:
            source_name = doc_id_to_source.get(chunk.document_id, chunk.document_id)
            chunk_context = {
                **context,
                "source_name": source_name,
            }

            triples = self.extract_from_text(
                text=chunk.text,
                document_id=chunk.document_id,
                existing_artifacts=all_triples,
                context=chunk_context,
            )
            all_triples.extend(triples)

        return all_triples

    def get_graph_statistics(self, corpus_id: str) -> Dict[str, Any]:
        """Get statistics about the knowledge graph.

        Args:
            corpus_id: The corpus identifier.

        Returns:
            Dictionary with graph statistics.
        """
        graph_store = self._ensure_graph_store(corpus_id)
        return graph_store.get_statistics()

    def query_graph(
        self,
        corpus_id: str,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query the knowledge graph by pattern.

        Args:
            corpus_id: The corpus identifier.
            subject: Optional subject filter.
            predicate: Optional predicate filter.
            obj: Optional object filter.

        Returns:
            List of matching edges.
        """
        graph_store = self._ensure_graph_store(corpus_id)
        return graph_store.query_by_pattern(subject, predicate, obj)

    def get_entities(self, corpus_id: str) -> List[Dict[str, Any]]:
        """Get all entities (nodes) from the knowledge graph.

        Args:
            corpus_id: The corpus identifier.

        Returns:
            List of entity dictionaries.
        """
        graph_store = self._ensure_graph_store(corpus_id)
        return graph_store.nodes()

    def export_graph_json(self, corpus_id: str) -> str:
        """Export knowledge graph as JSON.

        Args:
            corpus_id: The corpus identifier.

        Returns:
            JSON string representation of the graph.
        """
        graph_store = self._ensure_graph_store(corpus_id)

        nodes = graph_store.nodes()
        edges = graph_store.edges()

        return json.dumps(
            {
                "nodes": nodes,
                "edges": edges,
                "statistics": graph_store.get_statistics(),
            },
            indent=2,
            default=str,
        )

