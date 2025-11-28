"""Base class for pipeline stages.

Each pipeline stage (vocabulary, metadata, taxonomy, thesaurus, ontology,
knowledge graph) inherits from BasePipelineStage and implements the
extract, merge, and persist methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Generic, List, Optional, TypeVar

from spindle.ingestion.storage.corpus import CorpusManager
from spindle.ingestion.types import ChunkArtifact, Corpus
from spindle.pipeline.types import (
    PipelineStage,
    PipelineStageResult,
    PipelineState,
)

# Type variable for stage-specific artifact types
T = TypeVar("T")


class BasePipelineStage(ABC, Generic[T]):
    """Abstract base class for ontology pipeline stages.

    Each stage must implement:
    - extract_from_text: Extract artifacts from a single text chunk
    - merge_artifacts: Merge/deduplicate artifacts from multiple extractions
    - persist_artifacts: Save artifacts to storage
    - load_artifacts: Load previously persisted artifacts

    The stage is responsible for its own artifact type (VocabularyTerm,
    TaxonomyNode, etc.) and manages extraction, merging, and persistence.
    """

    stage: PipelineStage

    def __init__(
        self,
        corpus_manager: CorpusManager,
        graph_store: Optional[Any] = None,
    ) -> None:
        """Initialize the pipeline stage.

        Args:
            corpus_manager: CorpusManager for accessing corpus data.
            graph_store: Optional GraphStore for stages that need graph storage.
        """
        self.corpus_manager = corpus_manager
        self.graph_store = graph_store

    @abstractmethod
    def extract_from_text(
        self,
        text: str,
        document_id: str,
        existing_artifacts: List[T],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[T]:
        """Extract artifacts from a single text.

        Args:
            text: The text to extract from.
            document_id: ID of the source document.
            existing_artifacts: Previously extracted artifacts for context.
            context: Optional additional context (e.g., previous stage results).

        Returns:
            List of extracted artifacts.
        """
        pass

    @abstractmethod
    def merge_artifacts(self, artifact_sets: List[List[T]]) -> List[T]:
        """Merge and deduplicate artifacts from multiple extractions.

        Args:
            artifact_sets: List of artifact lists to merge.

        Returns:
            Merged and deduplicated list of artifacts.
        """
        pass

    @abstractmethod
    def persist_artifacts(
        self,
        corpus_id: str,
        artifacts: List[T],
    ) -> int:
        """Persist artifacts to storage.

        Args:
            corpus_id: The corpus identifier.
            artifacts: List of artifacts to persist.

        Returns:
            Number of artifacts persisted.
        """
        pass

    @abstractmethod
    def load_artifacts(self, corpus_id: str) -> List[T]:
        """Load previously persisted artifacts for a corpus.

        Args:
            corpus_id: The corpus identifier.

        Returns:
            List of loaded artifacts.
        """
        pass

    def get_corpus_chunks(self, corpus_id: str) -> List[ChunkArtifact]:
        """Get all text chunks for a corpus.

        Args:
            corpus_id: The corpus identifier.

        Returns:
            List of ChunkArtifact objects.
        """
        return self.corpus_manager.get_corpus_chunks(corpus_id)

    def get_previous_stage_context(
        self,
        corpus_id: str,
        pipeline_state: PipelineState,
    ) -> Dict[str, Any]:
        """Get context from previous pipeline stages.

        Override in subclasses to gather specific context needed
        from earlier stages.

        Args:
            corpus_id: The corpus identifier.
            pipeline_state: Current pipeline state.

        Returns:
            Context dictionary with previous stage artifacts.
        """
        return {}

    def run(
        self,
        corpus: Corpus,
        pipeline_state: PipelineState,
        strategy: str = "sequential",
    ) -> PipelineStageResult:
        """Execute the pipeline stage.

        Args:
            corpus: The corpus to process.
            pipeline_state: Current pipeline state.
            strategy: Extraction strategy ('sequential', 'batch_consolidate', 'sample_based').

        Returns:
            PipelineStageResult with execution details.
        """
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
                    error_message="Prerequisites not met. Complete earlier stages first.",
                )

            # Get chunks to process
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

            # Get context from previous stages
            context = self.get_previous_stage_context(corpus.corpus_id, pipeline_state)

            # Extract based on strategy
            if strategy == "sequential":
                artifacts = self._extract_sequential(chunks, context)
            elif strategy == "batch_consolidate":
                artifacts = self._extract_batch_consolidate(chunks, context)
            elif strategy == "sample_based":
                artifacts = self._extract_sample_based(chunks, context)
            else:
                artifacts = self._extract_sequential(chunks, context)

            # Persist artifacts
            count = self.persist_artifacts(corpus.corpus_id, artifacts)

            return PipelineStageResult(
                stage=self.stage,
                corpus_id=corpus.corpus_id,
                success=True,
                started_at=started_at,
                finished_at=datetime.utcnow(),
                artifact_count=count,
                metrics={
                    "chunks_processed": len(chunks),
                    "strategy": strategy,
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

    def _extract_sequential(
        self,
        chunks: List[ChunkArtifact],
        context: Dict[str, Any],
    ) -> List[T]:
        """Process chunks sequentially, accumulating artifacts.

        Each extraction sees all previously extracted artifacts,
        enabling consistent terminology and deduplication.
        """
        all_artifacts: List[T] = []

        for chunk in chunks:
            new_artifacts = self.extract_from_text(
                text=chunk.text,
                document_id=chunk.document_id,
                existing_artifacts=all_artifacts,
                context=context,
            )
            all_artifacts.extend(new_artifacts)

        return all_artifacts

    def _extract_batch_consolidate(
        self,
        chunks: List[ChunkArtifact],
        context: Dict[str, Any],
    ) -> List[T]:
        """Process all chunks independently, then merge results.

        Faster for large corpora but may produce more duplicates
        that need merging.
        """
        artifact_sets: List[List[T]] = []

        for chunk in chunks:
            artifacts = self.extract_from_text(
                text=chunk.text,
                document_id=chunk.document_id,
                existing_artifacts=[],
                context=context,
            )
            artifact_sets.append(artifacts)

        return self.merge_artifacts(artifact_sets)

    def _extract_sample_based(
        self,
        chunks: List[ChunkArtifact],
        context: Dict[str, Any],
        sample_ratio: float = 0.3,
    ) -> List[T]:
        """Use representative samples to build initial structure, then refine.

        1. Extract from a sample of chunks
        2. Use results as context for remaining chunks
        3. Merge all results
        """
        import random

        # Sample chunks
        sample_size = max(1, int(len(chunks) * sample_ratio))
        sample_chunks = random.sample(chunks, min(sample_size, len(chunks)))
        remaining_chunks = [c for c in chunks if c not in sample_chunks]

        # Extract from sample
        sample_artifacts: List[T] = []
        for chunk in sample_chunks:
            artifacts = self.extract_from_text(
                text=chunk.text,
                document_id=chunk.document_id,
                existing_artifacts=sample_artifacts,
                context=context,
            )
            sample_artifacts.extend(artifacts)

        # Extract from remaining with sample as context
        all_artifact_sets = [sample_artifacts]
        for chunk in remaining_chunks:
            artifacts = self.extract_from_text(
                text=chunk.text,
                document_id=chunk.document_id,
                existing_artifacts=sample_artifacts,
                context=context,
            )
            all_artifact_sets.append(artifacts)

        return self.merge_artifacts(all_artifact_sets)

