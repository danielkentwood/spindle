"""Ontology Pipeline: A systematic approach to semantic knowledge management.

This module implements the six-stage Ontology Pipeline as described in
"How the Ontology Pipeline Powers Semantic Knowledge Systems":

1. Controlled Vocabulary - Clean, disambiguated terms with definitions
2. Metadata Standards - Schema-based controls for data description
3. Taxonomy - Hierarchical parent-child relationships
4. Thesaurus - Semantic relationships (BT, NT, RT, USE, USE_FOR)
5. Ontology - Domain-specific entity types and relations
6. Knowledge Graph - Queryable synthesis of all stages

The pipeline stages build on each other iteratively, with each stage
preparing data for the next. Results are stored in hybrid SQLite
(metadata) and Kuzu graph (relationships) storage.
"""

from spindle.pipeline.base import BasePipelineStage
from spindle.pipeline.orchestrator import PipelineOrchestrator
from spindle.pipeline.strategies import (
    BatchConsolidateStrategy,
    ExtractionStrategy,
    SampleBasedStrategy,
    SequentialStrategy,
    get_strategy,
)
from spindle.pipeline.types import (
    ExtractionStrategyType,
    MetadataElement,
    MetadataElementType,
    PipelineStage,
    PipelineStageResult,
    PipelineState,
    TaxonomyNode,
    TaxonomyRelation,
    ThesaurusEntry,
    ThesaurusRelationType,
    VocabularyTerm,
)

__all__ = [
    # Types
    "VocabularyTerm",
    "MetadataElement",
    "MetadataElementType",
    "TaxonomyNode",
    "TaxonomyRelation",
    "ThesaurusEntry",
    "ThesaurusRelationType",
    "PipelineStage",
    "PipelineStageResult",
    "PipelineState",
    "ExtractionStrategyType",
    # Base and orchestration
    "BasePipelineStage",
    "PipelineOrchestrator",
    # Strategies
    "ExtractionStrategy",
    "SequentialStrategy",
    "BatchConsolidateStrategy",
    "SampleBasedStrategy",
    "get_strategy",
]

