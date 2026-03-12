"""spindle-eval integration bridge.

Provides ``get_pipeline_definition()`` — the factory function that spindle-eval's
runner calls to discover the Spindle pipeline stages.

Usage from spindle-eval::

    from spindle import get_pipeline_definition
    stages = get_pipeline_definition(cfg, tracker=tracker)
    # stages is a list[StageDef]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Conditional imports from spindle-eval
# ---------------------------------------------------------------------------

try:
    from spindle_eval.protocols import StageDef, StageResult
    _EVAL_AVAILABLE = True
except ImportError:
    _EVAL_AVAILABLE = False

    @dataclass
    class StageResult:  # type: ignore[no-redef]
        """Fallback StageResult when spindle-eval is not installed."""
        outputs: Dict[str, Any] = field(default_factory=dict)
        metrics: Dict[str, Any] = field(default_factory=dict)
        events: List[Any] = field(default_factory=list)

    @dataclass
    class StageDef:  # type: ignore[no-redef]
        """Fallback StageDef when spindle-eval is not installed."""
        name: str
        stage: Any
        input_keys: Dict[str, str] = field(default_factory=dict)
        metrics: List[Any] = field(default_factory=list)
        gate: Optional[Any] = None


# ---------------------------------------------------------------------------
# Metric loaders (conditional)
# ---------------------------------------------------------------------------

def _load_chunk_metrics() -> List[Any]:
    """Return chunk metric callables if spindle-eval is installed."""
    try:
        from spindle_eval.metrics import chunk_metrics
        return [
            chunk_metrics.boundary_coherence,
            chunk_metrics.size_distribution,
        ]
    except (ImportError, AttributeError):
        return []


def _load_kos_extraction_metrics() -> List[Any]:
    """Return KOS extraction metric callables."""
    try:
        from spindle_eval.metrics import kos_metrics
        return [
            kos_metrics.taxonomy_depth,
            kos_metrics.label_quality,
            kos_metrics.thesaurus_connectivity,
        ]
    except (ImportError, AttributeError):
        return []


def _load_ontology_synthesis_metrics() -> List[Any]:
    """Return ontology synthesis metric callables."""
    try:
        from spindle_eval.metrics import kos_metrics
        return [
            kos_metrics.axiom_density,
            kos_metrics.shacl_conformance,
        ]
    except (ImportError, AttributeError):
        return []


# ---------------------------------------------------------------------------
# Adapter classes — conform standalone stages to the Stage protocol
# ---------------------------------------------------------------------------

class PreprocessingAdapter:
    """Wraps PreprocessingStage to conform to Stage protocol."""

    name = "preprocessing"

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def run(self, inputs: dict, cfg: Any) -> StageResult:
        paths = _get(cfg, "dataset_path", [])
        if isinstance(paths, str):
            paths = [paths]
        chunks = self._inner.run(paths=paths)
        return StageResult(
            outputs={"chunks": chunks},
            metrics={"chunk_count": len(chunks)},
        )


class KOSExtractionAdapter:
    """Wraps KOSExtractionStage to conform to Stage protocol."""

    name = "kos_extraction"

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def run(self, inputs: dict, cfg: Any) -> StageResult:
        chunks = inputs.get("chunks", [])
        result = self._inner.run(chunks=chunks)
        return StageResult(
            outputs={"kos_path": result.get("kos_path"), **result},
            metrics={
                k: v for k, v in result.items()
                if isinstance(v, (int, float))
            },
        )


class OntologySynthesisAdapter:
    """Wraps OntologySynthesisStage to conform to Stage protocol."""

    name = "ontology_synthesis"

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def run(self, inputs: dict, cfg: Any) -> StageResult:
        result = self._inner.run()
        return StageResult(
            outputs={
                "ontology_path": result.get("ontology_path"),
                "shapes_path": result.get("shapes_path"),
                "graph": result.get("graph"),
            },
            metrics={
                k: v for k, v in result.items()
                if isinstance(v, (int, float))
            },
        )


class RetrievalAdapter:
    """Wraps RetrievalStage to conform to Stage protocol."""

    name = "retrieval"

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def run(self, inputs: dict, cfg: Any) -> StageResult:
        query = inputs.get("query", _get(cfg, "query", ""))
        result = self._inner.run(query=query)
        return StageResult(
            outputs={"contexts": result},
            metrics={},
        )


class GenerationAdapter:
    """Wraps GenerationStage to conform to Stage protocol."""

    name = "generation"

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def run(self, inputs: dict, cfg: Any) -> StageResult:
        contexts = inputs.get("contexts", {})
        all_triples: list = []
        # contexts may be a dict with retrieval source keys mapping to lists
        texts: list = []
        if isinstance(contexts, dict):
            for source_results in contexts.values():
                if isinstance(source_results, list):
                    for item in source_results:
                        if isinstance(item, dict) and "text" in item:
                            texts.append(item["text"])
                        elif isinstance(item, str):
                            texts.append(item)
        elif isinstance(contexts, list):
            for item in contexts:
                if isinstance(item, dict) and "text" in item:
                    texts.append(item["text"])
                elif isinstance(item, str):
                    texts.append(item)

        for text in texts:
            result = self._inner.run(text=text)
            all_triples.extend(result.get("triples", []))

        return StageResult(
            outputs={"triples": all_triples},
            metrics={"triple_count": len(all_triples)},
        )


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------

def get_pipeline_definition(
    cfg: Optional[Any] = None,
    *,
    tracker: Optional[Any] = None,
    kos_dir: Optional[Path] = None,
    include_kos: bool = True,
    include_generation: bool = True,
    ontology: Optional[Any] = None,
    graph_store: Optional[Any] = None,
    vector_store: Optional[Any] = None,
) -> List[StageDef]:
    """Factory that creates a Spindle pipeline definition for spindle-eval.

    Called by spindle-eval's runner as::

        spindle.get_pipeline_definition(cfg, tracker=tracker)

    Returns:
        list[StageDef] — directly consumable by PipelineExecutor.
    """
    from spindle.stages.preprocessing import PreprocessingStage
    from spindle.stages.retrieval import RetrievalStage

    if kos_dir is None:
        from spindle.configuration import find_stores_root
        kos_dir = find_stores_root() / "kos"
    stage_defs: List[StageDef] = []

    # Stage 1: Document preprocessing
    preprocessing_cfg = _get(cfg, "preprocessing", {})
    stage_defs.append(StageDef(
        name="preprocessing",
        stage=PreprocessingAdapter(
            PreprocessingStage(cfg=preprocessing_cfg, tracker=tracker),
        ),
        input_keys={},
        metrics=_load_chunk_metrics(),
    ))

    if include_kos:
        kos_svc = _build_kos_service(kos_dir, tracker)
        if kos_svc is not None:
            from spindle.stages.kos_extraction import KOSExtractionStage
            from spindle.stages.ontology_synthesis import OntologySynthesisStage

            stage_dir = kos_dir / "staging"

            # Stage 2: KOS extraction
            stage_defs.append(StageDef(
                name="kos_extraction",
                stage=KOSExtractionAdapter(
                    KOSExtractionStage(
                        kos_service=kos_svc,
                        stage_dir=stage_dir,
                        tracker=tracker,
                    ),
                ),
                input_keys={"chunks": "preprocessing.chunks"},
                metrics=_load_kos_extraction_metrics(),
                gate=lambda m: m.get("orphan_concept_ratio", 1.0) <= 0.3,
            ))

            # Stage 3: Ontology synthesis
            synthesis_cfg = _get(cfg, "ontology_synthesis", {})
            stage_defs.append(StageDef(
                name="ontology_synthesis",
                stage=OntologySynthesisAdapter(
                    OntologySynthesisStage(
                        kos_service=kos_svc,
                        output_dir=kos_dir,
                        max_axioms_per_class=_get(synthesis_cfg, "max_axioms_per_class", 10),
                        generate_shacl=_get(synthesis_cfg, "generate_shacl", True),
                    ),
                ),
                input_keys={"kos_path": "kos_extraction.kos_path"},
                metrics=_load_ontology_synthesis_metrics(),
            ))

            # Stage 4: Retrieval
            retrieval_cfg = _get(cfg, "retrieval", {})
            stage_defs.append(StageDef(
                name="retrieval",
                stage=RetrievalAdapter(
                    RetrievalStage(
                        kos_service=kos_svc,
                        graph_store=graph_store,
                        vector_store=vector_store,
                        mode=_get(retrieval_cfg, "mode", "hybrid"),
                        top_k=_get(retrieval_cfg, "top_k", 10),
                    ),
                ),
                input_keys={"graph": "ontology_synthesis.graph"},
            ))

    if include_generation:
        from spindle.stages.generation import GenerationStage
        stage_defs.append(StageDef(
            name="generation",
            stage=GenerationAdapter(
                GenerationStage(tracker=tracker, ontology=ontology),
            ),
            input_keys={"contexts": "retrieval.contexts"},
        ))

    return stage_defs


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
