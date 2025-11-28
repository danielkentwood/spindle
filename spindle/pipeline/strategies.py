"""Extraction strategies for the ontology pipeline.

Strategies determine how documents in a corpus are processed:
- Sequential: Process one at a time, accumulating context
- Batch Consolidate: Process all in parallel, then merge
- Sample Based: Use samples to build initial structure, then refine
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, TYPE_CHECKING

from spindle.ingestion.types import ChunkArtifact, Corpus
from spindle.pipeline.types import (
    ExtractionStrategyType,
    PipelineStageResult,
    PipelineState,
)

if TYPE_CHECKING:
    from spindle.pipeline.base import BasePipelineStage


class ExtractionStrategy(ABC):
    """Abstract base class for extraction strategies.

    Strategies determine how documents/chunks are processed during
    pipeline stage execution. Different strategies offer trade-offs
    between speed, accuracy, and context utilization.
    """

    strategy_type: ExtractionStrategyType

    @abstractmethod
    def execute(
        self,
        stage: "BasePipelineStage",
        corpus: Corpus,
        pipeline_state: PipelineState,
    ) -> PipelineStageResult:
        """Execute the pipeline stage using this strategy.

        Args:
            stage: The pipeline stage to execute.
            corpus: The corpus to process.
            pipeline_state: Current pipeline state.

        Returns:
            PipelineStageResult with execution details.
        """
        pass


class SequentialStrategy(ExtractionStrategy):
    """Process documents one at a time, accumulating results.

    Each document extraction sees all previously extracted artifacts,
    enabling consistent terminology and effective deduplication.

    Best for:
    - Small to medium corpora
    - When entity consistency is critical
    - When order matters (e.g., building on prior context)

    Trade-offs:
    - Slower for large corpora
    - Better context utilization
    - More consistent results
    """

    strategy_type = ExtractionStrategyType.SEQUENTIAL

    def execute(
        self,
        stage: "BasePipelineStage",
        corpus: Corpus,
        pipeline_state: PipelineState,
    ) -> PipelineStageResult:
        """Execute stage by processing chunks sequentially."""
        return stage.run(corpus, pipeline_state, strategy="sequential")


class BatchConsolidateStrategy(ExtractionStrategy):
    """Process all documents in parallel, then merge/deduplicate.

    Documents are processed independently (potentially in parallel),
    then results are merged and deduplicated in a consolidation phase.

    Best for:
    - Large corpora
    - When speed is prioritized
    - When documents are relatively independent

    Trade-offs:
    - Faster processing
    - May produce more duplicates initially
    - Requires robust merging logic
    """

    strategy_type = ExtractionStrategyType.BATCH_CONSOLIDATE

    def execute(
        self,
        stage: "BasePipelineStage",
        corpus: Corpus,
        pipeline_state: PipelineState,
    ) -> PipelineStageResult:
        """Execute stage by processing all chunks then consolidating."""
        return stage.run(corpus, pipeline_state, strategy="batch_consolidate")


class SampleBasedStrategy(ExtractionStrategy):
    """Use representative samples to build initial structure, then refine.

    1. Extract from a sample of documents (default 30%)
    2. Use sample results as context/seed for remaining documents
    3. Merge all results

    Best for:
    - Very large corpora
    - When documents have significant overlap
    - When initial structure can guide later extraction

    Trade-offs:
    - Good balance of speed and context
    - Sample selection affects quality
    - May miss patterns only in non-sampled documents
    """

    strategy_type = ExtractionStrategyType.SAMPLE_BASED

    def __init__(self, sample_ratio: float = 0.3) -> None:
        """Initialize with sample ratio.

        Args:
            sample_ratio: Fraction of documents to use as sample (0.0-1.0).
        """
        self.sample_ratio = sample_ratio

    def execute(
        self,
        stage: "BasePipelineStage",
        corpus: Corpus,
        pipeline_state: PipelineState,
    ) -> PipelineStageResult:
        """Execute stage using sample-based extraction."""
        return stage.run(corpus, pipeline_state, strategy="sample_based")


def get_strategy(strategy_type: ExtractionStrategyType) -> ExtractionStrategy:
    """Factory function to get a strategy by type.

    Args:
        strategy_type: The type of strategy to create.

    Returns:
        An ExtractionStrategy instance.

    Raises:
        ValueError: If strategy type is unknown.
    """
    strategies = {
        ExtractionStrategyType.SEQUENTIAL: SequentialStrategy,
        ExtractionStrategyType.BATCH_CONSOLIDATE: BatchConsolidateStrategy,
        ExtractionStrategyType.SAMPLE_BASED: SampleBasedStrategy,
    }

    if strategy_type not in strategies:
        raise ValueError(f"Unknown strategy type: {strategy_type}")

    return strategies[strategy_type]()

