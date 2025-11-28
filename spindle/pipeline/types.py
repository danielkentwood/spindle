"""Type definitions for the Ontology Pipeline stages.

This module defines dataclasses for artifacts produced at each pipeline stage:
- Controlled Vocabulary: VocabularyTerm
- Metadata Standards: MetadataElement
- Taxonomy: TaxonomyNode, TaxonomyRelation
- Thesaurus: ThesaurusEntry
- Ontology: Uses existing spindle.baml_client.types.Ontology
- Knowledge Graph: Uses existing spindle.graph_store.GraphStore
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence


class PipelineStage(str, Enum):
    """Enumeration of pipeline stages."""

    VOCABULARY = "vocabulary"
    METADATA = "metadata"
    TAXONOMY = "taxonomy"
    THESAURUS = "thesaurus"
    ONTOLOGY = "ontology"
    KNOWLEDGE_GRAPH = "knowledge_graph"


class ExtractionStrategyType(str, Enum):
    """Enumeration of extraction strategies."""

    SEQUENTIAL = "sequential"
    BATCH_CONSOLIDATE = "batch_consolidate"
    SAMPLE_BASED = "sample_based"


class MetadataElementType(str, Enum):
    """Types of metadata elements following library science standards."""

    STRUCTURAL = "structural"  # For machine readability (format, encoding)
    DESCRIPTIVE = "descriptive"  # For context (title, author, subject)
    ADMINISTRATIVE = "administrative"  # For maintenance and lineage


class ThesaurusRelationType(str, Enum):
    """Thesaurus relationship types following ISO 25964 / SKOS vocabulary."""

    USE = "USE"  # Preferred term indicator
    USE_FOR = "UF"  # Non-preferred term indicator (inverse of USE)
    BROADER_TERM = "BT"  # Hierarchical broader concept
    NARROWER_TERM = "NT"  # Hierarchical narrower concept
    RELATED_TERM = "RT"  # Associative/related concept
    SCOPE_NOTE = "SN"  # Definition or scope clarification
    HISTORY_NOTE = "HN"  # Historical context


@dataclass(slots=True)
class VocabularyTerm:
    """A controlled vocabulary term with definition and synonyms.

    Controlled vocabulary is the first building block in the Ontology Pipeline.
    Terms are de-duplicated, merged, and defined to arrive at a clean,
    disambiguated vocabulary fit for purpose.
    """

    term_id: str
    preferred_label: str
    definition: str
    synonyms: List[str] = field(default_factory=list)
    domain: Optional[str] = None
    source_document_ids: List[str] = field(default_factory=list)
    usage_count: int = 1
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True)
class MetadataElement:
    """A metadata schema element for describing data assets.

    Metadata standards provide schema-based control for databases and
    information systems. Elements are classified as structural (machine),
    descriptive (context), or administrative (maintenance/lineage).
    """

    element_id: str
    name: str
    element_type: MetadataElementType
    description: str
    data_type: str  # string, int, float, bool, date, etc.
    required: bool = False
    allowed_values: Optional[List[str]] = None
    default_value: Optional[str] = None
    examples: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True)
class TaxonomyNode:
    """A node in the taxonomy hierarchy.

    Taxonomy provides hierarchical structure via parent-child relations,
    serving as the foundation for thesaurus construction.
    """

    node_id: str
    term_id: str  # References VocabularyTerm
    label: str
    level: int = 0  # Depth in hierarchy (0 = root)
    parent_node_id: Optional[str] = None
    child_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True)
class TaxonomyRelation:
    """A hierarchical relationship between taxonomy nodes.

    Encodes parent-child (broader/narrower) relationships that form
    the backbone of the taxonomy tree structure.
    """

    relation_id: str
    parent_node_id: str
    child_node_id: str
    relation_type: str = "broader"  # broader or narrower
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True)
class ThesaurusEntry:
    """An entry in the thesaurus with semantic relationships.

    Thesaurus extends taxonomy with additional semantic relationships:
    USE/USE_FOR (synonymy), BT/NT (hierarchy), RT (association).
    This follows ISO 25964 / SKOS vocabulary standards.
    """

    entry_id: str
    term_id: str  # References VocabularyTerm
    preferred_label: str
    use_for: List[str] = field(default_factory=list)  # Non-preferred synonyms
    broader_terms: List[str] = field(default_factory=list)  # Parent concepts
    narrower_terms: List[str] = field(default_factory=list)  # Child concepts
    related_terms: List[str] = field(default_factory=list)  # Associated concepts
    scope_note: Optional[str] = None  # Definition/clarification
    history_note: Optional[str] = None  # Historical context
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True)
class PipelineStageResult:
    """Result from executing a pipeline stage.

    Captures the artifacts produced, metrics, and any errors
    encountered during stage execution.
    """

    stage: PipelineStage
    corpus_id: str
    success: bool
    started_at: datetime
    finished_at: datetime
    artifact_count: int = 0
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PipelineState:
    """Tracks the overall state of pipeline execution for a corpus.

    Stored in Corpus.pipeline_state to track which stages have been
    completed and their results.
    """

    corpus_id: str
    current_stage: Optional[PipelineStage] = None
    completed_stages: List[PipelineStage] = field(default_factory=list)
    stage_results: Dict[str, PipelineStageResult] = field(default_factory=dict)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    strategy: ExtractionStrategyType = ExtractionStrategyType.SEQUENTIAL

    def is_stage_complete(self, stage: PipelineStage) -> bool:
        """Check if a stage has been completed."""
        return stage in self.completed_stages

    def can_run_stage(self, stage: PipelineStage) -> bool:
        """Check if prerequisites are met to run a stage."""
        stage_order = list(PipelineStage)
        stage_idx = stage_order.index(stage)

        # First stage can always run
        if stage_idx == 0:
            return True

        # Check all previous stages are complete
        for prev_stage in stage_order[:stage_idx]:
            if prev_stage not in self.completed_stages:
                return False
        return True

    def mark_stage_complete(self, result: PipelineStageResult) -> None:
        """Mark a stage as complete with its result."""
        if result.success and result.stage not in self.completed_stages:
            self.completed_stages.append(result.stage)
        self.stage_results[result.stage.value] = result
        self.current_stage = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "corpus_id": self.corpus_id,
            "current_stage": self.current_stage.value if self.current_stage else None,
            "completed_stages": [s.value for s in self.completed_stages],
            "stage_results": {
                k: {
                    "stage": v.stage.value,
                    "corpus_id": v.corpus_id,
                    "success": v.success,
                    "started_at": v.started_at.isoformat(),
                    "finished_at": v.finished_at.isoformat(),
                    "artifact_count": v.artifact_count,
                    "error_message": v.error_message,
                    "metrics": v.metrics,
                }
                for k, v in self.stage_results.items()
            },
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "strategy": self.strategy.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineState":
        """Create from dictionary."""
        stage_results = {}
        for k, v in data.get("stage_results", {}).items():
            stage_results[k] = PipelineStageResult(
                stage=PipelineStage(v["stage"]),
                corpus_id=v["corpus_id"],
                success=v["success"],
                started_at=datetime.fromisoformat(v["started_at"]),
                finished_at=datetime.fromisoformat(v["finished_at"]),
                artifact_count=v.get("artifact_count", 0),
                error_message=v.get("error_message"),
                metrics=v.get("metrics", {}),
            )

        return cls(
            corpus_id=data["corpus_id"],
            current_stage=(
                PipelineStage(data["current_stage"])
                if data.get("current_stage")
                else None
            ),
            completed_stages=[
                PipelineStage(s) for s in data.get("completed_stages", [])
            ],
            stage_results=stage_results,
            started_at=(
                datetime.fromisoformat(data["started_at"])
                if data.get("started_at")
                else None
            ),
            finished_at=(
                datetime.fromisoformat(data["finished_at"])
                if data.get("finished_at")
                else None
            ),
            strategy=ExtractionStrategyType(
                data.get("strategy", ExtractionStrategyType.SEQUENTIAL.value)
            ),
        )

