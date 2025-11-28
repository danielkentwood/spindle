"""API endpoints for Ontology Pipeline operations."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from spindle.graph_store import GraphStore
from spindle.ingestion.storage import DocumentCatalog
from spindle.ingestion.storage.corpus import CorpusManager
from spindle.pipeline import (
    ExtractionStrategyType,
    PipelineOrchestrator,
    PipelineStage,
    PipelineState,
)


router = APIRouter()


# =============================================================================
# Request/Response Models
# =============================================================================


class RunStageRequest(BaseModel):
    """Request to run a pipeline stage."""

    strategy: str = Field(
        "sequential",
        description="Extraction strategy: 'sequential', 'batch_consolidate', 'sample_based'",
    )


class RunAllRequest(BaseModel):
    """Request to run all pipeline stages."""

    strategy: str = Field("sequential", description="Extraction strategy")
    stop_on_error: bool = Field(True, description="Stop on first error")


class StageResultInfo(BaseModel):
    """Information about a stage execution result."""

    stage: str
    success: bool
    artifact_count: int
    error_message: Optional[str]
    duration_seconds: float


class PipelineStatusResponse(BaseModel):
    """Pipeline status response."""

    corpus_id: str
    corpus_name: str
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    current_stage: Optional[str]
    completed_stages: List[str]
    pending_stages: List[str]
    strategy: str
    stage_results: Dict[str, StageResultInfo]


class VocabularyTermResponse(BaseModel):
    """Vocabulary term response."""

    term_id: str
    preferred_label: str
    definition: str
    synonyms: List[str]
    domain: Optional[str]
    usage_count: int


class TaxonomyNodeResponse(BaseModel):
    """Taxonomy node response."""

    node_id: str
    label: str
    level: int
    parent_node_id: Optional[str]
    child_count: int


class ThesaurusEntryResponse(BaseModel):
    """Thesaurus entry response."""

    entry_id: str
    preferred_label: str
    use_for: List[str]
    broader_terms: List[str]
    narrower_terms: List[str]
    related_terms: List[str]
    scope_note: Optional[str]


class OntologyResponse(BaseModel):
    """Ontology response."""

    entity_types: List[Dict[str, Any]]
    relation_types: List[Dict[str, Any]]


class KnowledgeGraphStatsResponse(BaseModel):
    """Knowledge graph statistics response."""

    node_count: int
    edge_count: int
    triple_count: int


# =============================================================================
# Helper Functions
# =============================================================================


def get_orchestrator() -> PipelineOrchestrator:
    """Get or create a PipelineOrchestrator instance."""
    catalog = DocumentCatalog("sqlite:///spindle_storage/catalog.db")
    corpus_manager = CorpusManager(catalog)
    graph_store = GraphStore("spindle_kg")

    orchestrator = PipelineOrchestrator(corpus_manager, graph_store)
    orchestrator.register_default_stages()

    return orchestrator


def get_strategy_type(strategy: str) -> ExtractionStrategyType:
    """Convert strategy string to enum."""
    mapping = {
        "sequential": ExtractionStrategyType.SEQUENTIAL,
        "batch_consolidate": ExtractionStrategyType.BATCH_CONSOLIDATE,
        "batch": ExtractionStrategyType.BATCH_CONSOLIDATE,
        "sample_based": ExtractionStrategyType.SAMPLE_BASED,
        "sample": ExtractionStrategyType.SAMPLE_BASED,
    }
    return mapping.get(strategy.lower(), ExtractionStrategyType.SEQUENTIAL)


# =============================================================================
# Pipeline Execution Endpoints
# =============================================================================


@router.post("/{corpus_id}/pipeline/{stage_name}", response_model=StageResultInfo)
async def run_stage(
    corpus_id: str,
    stage_name: str,
    request: RunStageRequest = RunStageRequest(),
):
    """Run a specific pipeline stage.

    Stages must be run in order: vocabulary -> metadata -> taxonomy ->
    thesaurus -> ontology -> knowledge_graph

    Args:
        corpus_id: Corpus identifier.
        stage_name: Stage to run.
        request: Execution parameters.

    Returns:
        Stage execution result.

    Raises:
        HTTPException: 404 if corpus not found, 400 if invalid stage.
    """
    orchestrator = get_orchestrator()

    # Validate stage name
    try:
        stage = PipelineStage(stage_name)
    except ValueError:
        valid_stages = [s.value for s in PipelineStage]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid stage: {stage_name}. Valid stages: {valid_stages}",
        )

    # Get corpus
    corpus = orchestrator.corpus_manager.get_corpus(corpus_id)
    if corpus is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Corpus not found: {corpus_id}",
        )

    # Run stage
    strategy_type = get_strategy_type(request.strategy)
    result = orchestrator.run_stage(corpus, stage, strategy_type)

    return StageResultInfo(
        stage=result.stage.value,
        success=result.success,
        artifact_count=result.artifact_count,
        error_message=result.error_message,
        duration_seconds=(result.finished_at - result.started_at).total_seconds(),
    )


@router.post("/{corpus_id}/pipeline/run-all", response_model=List[StageResultInfo])
async def run_all_stages(
    corpus_id: str,
    request: RunAllRequest = RunAllRequest(),
):
    """Run all pipeline stages in order.

    Args:
        corpus_id: Corpus identifier.
        request: Execution parameters.

    Returns:
        List of stage execution results.

    Raises:
        HTTPException: 404 if corpus not found.
    """
    orchestrator = get_orchestrator()

    corpus = orchestrator.corpus_manager.get_corpus(corpus_id)
    if corpus is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Corpus not found: {corpus_id}",
        )

    strategy_type = get_strategy_type(request.strategy)
    results = orchestrator.run_all(
        corpus,
        strategy_type,
        stop_on_error=request.stop_on_error,
    )

    return [
        StageResultInfo(
            stage=r.stage.value,
            success=r.success,
            artifact_count=r.artifact_count,
            error_message=r.error_message,
            duration_seconds=(r.finished_at - r.started_at).total_seconds(),
        )
        for r in results
    ]


@router.get("/{corpus_id}/pipeline/status", response_model=PipelineStatusResponse)
async def get_pipeline_status(corpus_id: str):
    """Get pipeline status for a corpus.

    Args:
        corpus_id: Corpus identifier.

    Returns:
        Pipeline status information.

    Raises:
        HTTPException: 404 if corpus not found.
    """
    orchestrator = get_orchestrator()

    corpus = orchestrator.corpus_manager.get_corpus(corpus_id)
    if corpus is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Corpus not found: {corpus_id}",
        )

    status_dict = orchestrator.get_status(corpus)

    # Convert stage results
    stage_results = {}
    for k, v in status_dict.get("stage_results", {}).items():
        stage_results[k] = StageResultInfo(
            stage=k,
            success=v["success"],
            artifact_count=v["artifact_count"],
            error_message=v.get("error_message"),
            duration_seconds=v["duration_seconds"],
        )

    return PipelineStatusResponse(
        corpus_id=status_dict["corpus_id"],
        corpus_name=status_dict["corpus_name"],
        started_at=datetime.fromisoformat(status_dict["started_at"]) if status_dict.get("started_at") else None,
        finished_at=datetime.fromisoformat(status_dict["finished_at"]) if status_dict.get("finished_at") else None,
        current_stage=status_dict.get("current_stage"),
        completed_stages=status_dict["completed_stages"],
        pending_stages=status_dict["pending_stages"],
        strategy=status_dict["strategy"],
        stage_results=stage_results,
    )


@router.post("/{corpus_id}/pipeline/reset", status_code=status.HTTP_204_NO_CONTENT)
async def reset_pipeline(corpus_id: str):
    """Reset pipeline state for a corpus.

    This clears pipeline state but does not delete extracted artifacts.

    Args:
        corpus_id: Corpus identifier.

    Raises:
        HTTPException: 404 if corpus not found.
    """
    orchestrator = get_orchestrator()

    corpus = orchestrator.corpus_manager.get_corpus(corpus_id)
    if corpus is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Corpus not found: {corpus_id}",
        )

    orchestrator.reset_pipeline(corpus)
    return None


# =============================================================================
# Pipeline Artifact Endpoints
# =============================================================================


@router.get("/{corpus_id}/vocabulary", response_model=List[VocabularyTermResponse])
async def get_vocabulary(corpus_id: str):
    """Get extracted vocabulary for a corpus.

    Args:
        corpus_id: Corpus identifier.

    Returns:
        List of vocabulary terms.

    Raises:
        HTTPException: 404 if corpus not found.
    """
    orchestrator = get_orchestrator()

    corpus = orchestrator.corpus_manager.get_corpus(corpus_id)
    if corpus is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Corpus not found: {corpus_id}",
        )

    stage = orchestrator.get_stage(PipelineStage.VOCABULARY)
    if stage is None:
        return []

    terms = stage.load_artifacts(corpus_id)

    return [
        VocabularyTermResponse(
            term_id=t.term_id,
            preferred_label=t.preferred_label,
            definition=t.definition,
            synonyms=t.synonyms,
            domain=t.domain,
            usage_count=t.usage_count,
        )
        for t in terms
    ]


@router.get("/{corpus_id}/taxonomy", response_model=List[TaxonomyNodeResponse])
async def get_taxonomy(corpus_id: str):
    """Get extracted taxonomy for a corpus.

    Args:
        corpus_id: Corpus identifier.

    Returns:
        List of taxonomy nodes.

    Raises:
        HTTPException: 404 if corpus not found.
    """
    orchestrator = get_orchestrator()

    corpus = orchestrator.corpus_manager.get_corpus(corpus_id)
    if corpus is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Corpus not found: {corpus_id}",
        )

    stage = orchestrator.get_stage(PipelineStage.TAXONOMY)
    if stage is None:
        return []

    nodes = stage.load_artifacts(corpus_id)

    return [
        TaxonomyNodeResponse(
            node_id=n.node_id,
            label=n.label,
            level=n.level,
            parent_node_id=n.parent_node_id,
            child_count=n.child_count,
        )
        for n in nodes
    ]


@router.get("/{corpus_id}/thesaurus", response_model=List[ThesaurusEntryResponse])
async def get_thesaurus(corpus_id: str):
    """Get extracted thesaurus for a corpus.

    Args:
        corpus_id: Corpus identifier.

    Returns:
        List of thesaurus entries.

    Raises:
        HTTPException: 404 if corpus not found.
    """
    orchestrator = get_orchestrator()

    corpus = orchestrator.corpus_manager.get_corpus(corpus_id)
    if corpus is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Corpus not found: {corpus_id}",
        )

    stage = orchestrator.get_stage(PipelineStage.THESAURUS)
    if stage is None:
        return []

    entries = stage.load_artifacts(corpus_id)

    return [
        ThesaurusEntryResponse(
            entry_id=e.entry_id,
            preferred_label=e.preferred_label,
            use_for=e.use_for,
            broader_terms=e.broader_terms,
            narrower_terms=e.narrower_terms,
            related_terms=e.related_terms,
            scope_note=e.scope_note,
        )
        for e in entries
    ]


@router.get("/{corpus_id}/ontology", response_model=OntologyResponse)
async def get_ontology(corpus_id: str):
    """Get generated ontology for a corpus.

    Args:
        corpus_id: Corpus identifier.

    Returns:
        Ontology with entity and relation types.

    Raises:
        HTTPException: 404 if corpus or ontology not found.
    """
    orchestrator = get_orchestrator()

    corpus = orchestrator.corpus_manager.get_corpus(corpus_id)
    if corpus is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Corpus not found: {corpus_id}",
        )

    stage = orchestrator.get_stage(PipelineStage.ONTOLOGY)
    if stage is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Ontology stage not available",
        )

    ontologies = stage.load_artifacts(corpus_id)
    if not ontologies:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No ontology found. Run ontology stage first.",
        )

    ontology = ontologies[0]

    return OntologyResponse(
        entity_types=[
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
        relation_types=[
            {
                "name": rt.name,
                "description": rt.description,
                "domain": rt.domain,
                "range": rt.range,
            }
            for rt in ontology.relation_types
        ],
    )


@router.get("/{corpus_id}/knowledge-graph/stats", response_model=KnowledgeGraphStatsResponse)
async def get_kg_stats(corpus_id: str):
    """Get knowledge graph statistics for a corpus.

    Args:
        corpus_id: Corpus identifier.

    Returns:
        Knowledge graph statistics.

    Raises:
        HTTPException: 404 if corpus not found.
    """
    orchestrator = get_orchestrator()

    corpus = orchestrator.corpus_manager.get_corpus(corpus_id)
    if corpus is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Corpus not found: {corpus_id}",
        )

    stage = orchestrator.get_stage(PipelineStage.KNOWLEDGE_GRAPH)
    if stage is None:
        return KnowledgeGraphStatsResponse(
            node_count=0,
            edge_count=0,
            triple_count=0,
        )

    stats = stage.get_graph_statistics(corpus_id)

    return KnowledgeGraphStatsResponse(
        node_count=stats.get("node_count", 0),
        edge_count=stats.get("edge_count", 0),
        triple_count=stats.get("edge_count", 0),  # Same as edge count
    )


@router.get("/{corpus_id}/knowledge-graph/query")
async def query_kg(
    corpus_id: str,
    subject: Optional[str] = None,
    predicate: Optional[str] = None,
    object: Optional[str] = None,
):
    """Query the knowledge graph.

    Args:
        corpus_id: Corpus identifier.
        subject: Optional subject filter.
        predicate: Optional predicate filter.
        object: Optional object filter.

    Returns:
        List of matching edges.

    Raises:
        HTTPException: 404 if corpus not found.
    """
    orchestrator = get_orchestrator()

    corpus = orchestrator.corpus_manager.get_corpus(corpus_id)
    if corpus is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Corpus not found: {corpus_id}",
        )

    stage = orchestrator.get_stage(PipelineStage.KNOWLEDGE_GRAPH)
    if stage is None:
        return []

    results = stage.query_graph(corpus_id, subject, predicate, object)
    return results

