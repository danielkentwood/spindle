"""Pipeline orchestration for running ontology pipeline stages.

The PipelineOrchestrator coordinates execution of pipeline stages,
manages state transitions, and ensures prerequisites are met.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from spindle.graph_store import GraphStore
from spindle.ingestion.storage.corpus import CorpusManager
from spindle.ingestion.types import Corpus
from spindle.pipeline.strategies import ExtractionStrategy, get_strategy
from spindle.pipeline.types import (
    ExtractionStrategyType,
    PipelineStage,
    PipelineStageResult,
    PipelineState,
)

if TYPE_CHECKING:
    from spindle.pipeline.base import BasePipelineStage


class PipelineOrchestrator:
    """Orchestrates execution of ontology pipeline stages.

    The orchestrator manages:
    - Stage registration and lookup
    - Pipeline state tracking
    - Prerequisite validation
    - Sequential or selective stage execution

    Example:
        >>> from spindle.ingestion.storage import DocumentCatalog, CorpusManager
        >>> from spindle.graph_store import GraphStore
        >>>
        >>> catalog = DocumentCatalog("sqlite:///spindle.db")
        >>> corpus_manager = CorpusManager(catalog)
        >>> graph_store = GraphStore("my_graph")
        >>>
        >>> orchestrator = PipelineOrchestrator(corpus_manager, graph_store)
        >>> orchestrator.register_default_stages()
        >>>
        >>> # Run specific stage
        >>> result = orchestrator.run_stage(corpus, PipelineStage.VOCABULARY)
        >>>
        >>> # Run all stages
        >>> results = orchestrator.run_all(corpus)
    """

    def __init__(
        self,
        corpus_manager: CorpusManager,
        graph_store: Optional[GraphStore] = None,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            corpus_manager: CorpusManager for corpus data access.
            graph_store: Optional GraphStore for graph-based stages.
        """
        self.corpus_manager = corpus_manager
        self.graph_store = graph_store
        self._stages: Dict[PipelineStage, "BasePipelineStage"] = {}

    def register_stage(
        self,
        stage_type: PipelineStage,
        stage_instance: "BasePipelineStage",
    ) -> None:
        """Register a stage implementation.

        Args:
            stage_type: The pipeline stage type.
            stage_instance: The stage implementation instance.
        """
        self._stages[stage_type] = stage_instance

    def register_default_stages(self) -> None:
        """Register all default stage implementations.

        This method lazily imports and registers all standard stages.
        Call this after creating the orchestrator to enable all stages.
        """
        # Import stages here to avoid circular imports
        from spindle.pipeline.vocabulary import VocabularyStage
        from spindle.pipeline.metadata import MetadataStage
        from spindle.pipeline.taxonomy import TaxonomyStage
        from spindle.pipeline.thesaurus import ThesaurusStage
        from spindle.pipeline.ontology_stage import OntologyStage
        from spindle.pipeline.knowledge_graph import KnowledgeGraphStage

        self.register_stage(
            PipelineStage.VOCABULARY,
            VocabularyStage(self.corpus_manager, self.graph_store),
        )
        self.register_stage(
            PipelineStage.METADATA,
            MetadataStage(self.corpus_manager, self.graph_store),
        )
        self.register_stage(
            PipelineStage.TAXONOMY,
            TaxonomyStage(self.corpus_manager, self.graph_store),
        )
        self.register_stage(
            PipelineStage.THESAURUS,
            ThesaurusStage(self.corpus_manager, self.graph_store),
        )
        self.register_stage(
            PipelineStage.ONTOLOGY,
            OntologyStage(self.corpus_manager, self.graph_store),
        )
        self.register_stage(
            PipelineStage.KNOWLEDGE_GRAPH,
            KnowledgeGraphStage(self.corpus_manager, self.graph_store),
        )

    def get_stage(self, stage_type: PipelineStage) -> Optional["BasePipelineStage"]:
        """Get a registered stage by type.

        Args:
            stage_type: The pipeline stage type.

        Returns:
            The stage instance if registered, None otherwise.
        """
        return self._stages.get(stage_type)

    def get_pipeline_state(self, corpus: Corpus) -> PipelineState:
        """Get or create pipeline state for a corpus.

        Always fetches fresh state from the database to ensure consistency
        when running multiple stages in sequence.

        Args:
            corpus: The corpus to get state for.

        Returns:
            PipelineState object (from database or newly created).
        """
        # Always fetch fresh state from database to avoid stale in-memory state
        fresh_corpus = self.corpus_manager.get_corpus(corpus.corpus_id)
        if fresh_corpus and fresh_corpus.pipeline_state:
            try:
                return PipelineState.from_dict(fresh_corpus.pipeline_state)
            except (KeyError, ValueError):
                pass

        return PipelineState(corpus_id=corpus.corpus_id)

    def save_pipeline_state(
        self,
        corpus_id: str,
        state: PipelineState,
    ) -> None:
        """Save pipeline state to corpus.

        Args:
            corpus_id: The corpus identifier.
            state: The pipeline state to save.
        """
        self.corpus_manager.update_corpus(
            corpus_id,
            pipeline_state=state.to_dict(),
        )

    def run_stage(
        self,
        corpus: Corpus,
        stage_type: PipelineStage,
        strategy_type: ExtractionStrategyType = ExtractionStrategyType.SEQUENTIAL,
    ) -> PipelineStageResult:
        """Run a specific pipeline stage.

        Args:
            corpus: The corpus to process.
            stage_type: The stage to run.
            strategy_type: Extraction strategy to use.

        Returns:
            PipelineStageResult with execution details.

        Raises:
            ValueError: If stage is not registered.
        """
        stage = self.get_stage(stage_type)
        if stage is None:
            raise ValueError(f"Stage not registered: {stage_type}")

        # Get current state
        state = self.get_pipeline_state(corpus)

        # Check prerequisites
        if not state.can_run_stage(stage_type):
            missing = [
                s.value
                for s in list(PipelineStage)[: list(PipelineStage).index(stage_type)]
                if s not in state.completed_stages
            ]
            return PipelineStageResult(
                stage=stage_type,
                corpus_id=corpus.corpus_id,
                success=False,
                started_at=datetime.utcnow(),
                finished_at=datetime.utcnow(),
                error_message=f"Prerequisites not met. Complete these stages first: {missing}",
            )

        # Update state to mark stage as running
        state.current_stage = stage_type
        state.strategy = strategy_type
        if state.started_at is None:
            state.started_at = datetime.utcnow()
        self.save_pipeline_state(corpus.corpus_id, state)

        # Execute stage
        strategy = get_strategy(strategy_type)
        result = strategy.execute(stage, corpus, state)

        # Update state with result
        state.mark_stage_complete(result)
        if result.success and stage_type == PipelineStage.KNOWLEDGE_GRAPH:
            state.finished_at = datetime.utcnow()
        self.save_pipeline_state(corpus.corpus_id, state)

        return result

    def run_all(
        self,
        corpus: Corpus,
        strategy_type: ExtractionStrategyType = ExtractionStrategyType.SEQUENTIAL,
        stop_on_error: bool = True,
    ) -> List[PipelineStageResult]:
        """Run all pipeline stages in order.

        Args:
            corpus: The corpus to process.
            strategy_type: Extraction strategy to use for all stages.
            stop_on_error: If True, stop on first error. Otherwise continue.

        Returns:
            List of PipelineStageResult for each stage.
        """
        results: List[PipelineStageResult] = []

        for stage_type in PipelineStage:
            # Skip if stage not registered
            if stage_type not in self._stages:
                continue

            result = self.run_stage(corpus, stage_type, strategy_type)
            results.append(result)

            if not result.success and stop_on_error:
                break

        return results

    def run_from_stage(
        self,
        corpus: Corpus,
        start_stage: PipelineStage,
        strategy_type: ExtractionStrategyType = ExtractionStrategyType.SEQUENTIAL,
        stop_on_error: bool = True,
    ) -> List[PipelineStageResult]:
        """Run pipeline starting from a specific stage.

        Useful for resuming a partially completed pipeline or
        re-running specific stages after modifications.

        Args:
            corpus: The corpus to process.
            start_stage: The stage to start from.
            strategy_type: Extraction strategy to use.
            stop_on_error: If True, stop on first error.

        Returns:
            List of PipelineStageResult for executed stages.
        """
        results: List[PipelineStageResult] = []
        started = False

        for stage_type in PipelineStage:
            if stage_type == start_stage:
                started = True

            if not started:
                continue

            if stage_type not in self._stages:
                continue

            result = self.run_stage(corpus, stage_type, strategy_type)
            results.append(result)

            if not result.success and stop_on_error:
                break

        return results

    def get_status(self, corpus: Corpus) -> Dict[str, Any]:
        """Get pipeline status summary for a corpus.

        Args:
            corpus: The corpus to get status for.

        Returns:
            Dictionary with status information.
        """
        state = self.get_pipeline_state(corpus)

        return {
            "corpus_id": corpus.corpus_id,
            "corpus_name": corpus.name,
            "started_at": state.started_at.isoformat() if state.started_at else None,
            "finished_at": state.finished_at.isoformat() if state.finished_at else None,
            "current_stage": state.current_stage.value if state.current_stage else None,
            "completed_stages": [s.value for s in state.completed_stages],
            "pending_stages": [
                s.value
                for s in PipelineStage
                if s not in state.completed_stages
            ],
            "strategy": state.strategy.value,
            "stage_results": {
                k: {
                    "success": v.success,
                    "artifact_count": v.artifact_count,
                    "error_message": v.error_message,
                    "duration_seconds": (
                        v.finished_at - v.started_at
                    ).total_seconds(),
                }
                for k, v in state.stage_results.items()
            },
        }

    def reset_pipeline(self, corpus: Corpus) -> None:
        """Reset pipeline state for a corpus.

        Clears all pipeline state, allowing stages to be re-run.
        Does not delete extracted artifacts from storage.

        Args:
            corpus: The corpus to reset.
        """
        self.corpus_manager.update_corpus(
            corpus.corpus_id,
            pipeline_state={},
        )

    def reset_from_stage(
        self,
        corpus: Corpus,
        stage: PipelineStage,
    ) -> None:
        """Reset pipeline state from a specific stage onward.

        Clears state for the specified stage and all following stages,
        allowing them to be re-run while preserving earlier results.

        Args:
            corpus: The corpus to reset.
            stage: The stage to reset from (inclusive).
        """
        state = self.get_pipeline_state(corpus)
        stage_order = list(PipelineStage)
        stage_idx = stage_order.index(stage)

        # Remove completed stages from reset point onward
        state.completed_stages = [
            s for s in state.completed_stages if stage_order.index(s) < stage_idx
        ]

        # Remove stage results from reset point onward
        state.stage_results = {
            k: v
            for k, v in state.stage_results.items()
            if stage_order.index(PipelineStage(k)) < stage_idx
        }

        # Clear finished time if we're resetting
        state.finished_at = None
        state.current_stage = None

        self.save_pipeline_state(corpus.corpus_id, state)

