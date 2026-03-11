"""spindle-eval integration bridge.

Provides ``get_pipeline_definition()`` — the factory function that spindle-eval
calls to discover the Spindle pipeline stages.

Usage from spindle-eval::

    from spindle import get_pipeline_definition
    defn = get_pipeline_definition(cfg, ontology=my_ontology)
    for stage_def in defn.stages:
        output = stage_def.stage.run(...)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class StageDef:
    """Describes a pipeline stage with optional metrics and quality gates.

    Attributes:
        name: Human-readable stage identifier.
        stage: Stage instance implementing ``run()``.
        input_keys: Maps param name to ``"stage.output_key"`` documenting
            inter-stage data flow.
        metrics: Metric callables (populated when spindle-eval publishes them).
        gate: Quality gate callable (populated when spindle-eval publishes them).
    """

    name: str
    stage: Any
    input_keys: Dict[str, str] = field(default_factory=dict)
    metrics: List[Any] = field(default_factory=list)
    gate: Optional[Any] = None


@dataclass
class PipelineDefinition:
    """Describes a Spindle pipeline for spindle-eval.

    Attributes:
        stages: Ordered list of StageDef objects.
        metadata: Arbitrary metadata dict for the eval framework.
    """

    stages: List[StageDef] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def get_pipeline_definition(
    cfg: Optional[Any] = None,
    kos_dir: Optional[Path] = None,
    include_kos: bool = True,
    include_generation: bool = True,
    tracker: Optional[Any] = None,
    ontology: Optional[Any] = None,
    graph_store: Optional[Any] = None,
    vector_store: Optional[Any] = None,
) -> PipelineDefinition:
    """Factory that creates a Spindle pipeline definition for spindle-eval.

    Stages are created lazily; missing optional dependencies (pyoxigraph, etc.)
    are handled gracefully by each Stage constructor.

    Args:
        cfg: Hydra DictConfig or plain dict with pipeline configuration.
        kos_dir: Override the KOS directory (default: ``kos/``).
        include_kos: Whether to include KOS extraction stages.
        include_generation: Whether to include the generation stage.
        tracker: Optional tracker passed to all stages.
        ontology: Ontology object threaded to GenerationStage.
        graph_store: GraphStore instance for RetrievalStage.
        vector_store: ChromaVectorStore instance for RetrievalStage.

    Returns:
        PipelineDefinition with ordered stages wrapped in StageDef.
    """
    from spindle.stages.preprocessing import PreprocessingStage
    from spindle.stages.retrieval import RetrievalStage

    kos_dir = kos_dir or Path("kos")
    stage_defs: List[StageDef] = []

    # Stage 1: Document preprocessing
    preprocessing_cfg = _get(cfg, "preprocessing", {})
    stage_defs.append(StageDef(
        name="preprocessing",
        stage=PreprocessingStage(cfg=preprocessing_cfg, tracker=tracker),
        input_keys={},
    ))

    if include_kos:
        # Stage 2: KOS extraction
        kos_svc = _build_kos_service(kos_dir, tracker)
        if kos_svc is not None:
            from spindle.stages.kos_extraction import KOSExtractionStage
            stage_dir = kos_dir / "staging"
            stage_defs.append(StageDef(
                name="kos_extraction",
                stage=KOSExtractionStage(
                    kos_service=kos_svc,
                    stage_dir=stage_dir,
                    tracker=tracker,
                ),
                input_keys={"chunks": "preprocessing.chunks"},
            ))

            # Stage 3: Ontology synthesis
            from spindle.stages.ontology_synthesis import OntologySynthesisStage
            synthesis_cfg = _get(cfg, "ontology_synthesis", {})
            stage_defs.append(StageDef(
                name="ontology_synthesis",
                stage=OntologySynthesisStage(
                    kos_service=kos_svc,
                    output_dir=kos_dir,
                    max_axioms_per_class=_get(synthesis_cfg, "max_axioms_per_class", 10),
                    generate_shacl=_get(synthesis_cfg, "generate_shacl", True),
                ),
                input_keys={"kos_path": "kos_extraction.kos_path"},
            ))

            # Stage 4: Retrieval
            retrieval_cfg = _get(cfg, "retrieval", {})
            stage_defs.append(StageDef(
                name="retrieval",
                stage=RetrievalStage(
                    kos_service=kos_svc,
                    graph_store=graph_store,
                    vector_store=vector_store,
                    mode=_get(retrieval_cfg, "mode", "hybrid"),
                    top_k=_get(retrieval_cfg, "top_k", 10),
                ),
                input_keys={"graph": "ontology_synthesis.graph"},
            ))

    if include_generation:
        # Stage 5: LLM generation
        from spindle.stages.generation import GenerationStage
        stage_defs.append(StageDef(
            name="generation",
            stage=GenerationStage(tracker=tracker, ontology=ontology),
            input_keys={"contexts": "retrieval.contexts"},
        ))

    all_stages = stage_defs
    return PipelineDefinition(
        stages=all_stages,
        metadata={
            "kos_dir": str(kos_dir),
            "stage_count": len(all_stages),
            "stage_names": [sd.name for sd in all_stages],
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_kos_service(kos_dir: Path, tracker: Optional[Any]) -> Optional[Any]:
    """Attempt to build a KOSService; return None if pyoxigraph unavailable."""
    try:
        from spindle.kos.service import KOSService
        return KOSService(kos_dir=kos_dir)
    except Exception:
        return None


def _get(cfg: Any, key: str, default: Any) -> Any:
    """Safely retrieve a config value from dict, object, or Hydra DictConfig."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    try:
        from omegaconf import DictConfig
        if isinstance(cfg, DictConfig):
            return cfg.get(key, default)
    except ImportError:
        pass
    return getattr(cfg, key, default)
