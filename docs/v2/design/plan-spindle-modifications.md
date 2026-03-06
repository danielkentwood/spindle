# Plan: Spindle Modifications for spindle-eval v2 Integration

This document contains concrete, actionable plans for modifying the `spindle` package to support the new Stage-based pipeline architecture defined in [spindle-eval-architecture-analysis.md](../../spindle-eval-architecture-analysis.md).

Changes are organized by phase. Each phase lists the files to create or modify, what to put in them, and any dependencies on spindle-eval phases completing first.

---

## Table of Contents

1. [Phase 1: No changes required (compatibility shim)](#phase-1-no-changes-required)
2. [Phase 2: Tracker protocol alignment](#phase-2-tracker-protocol-alignment)
3. [Phase 3: KOS stage outputs for RDF evaluation](#phase-3-kos-stage-outputs)
4. [Phase 4: Hydra config registration](#phase-4-hydra-config-registration)
5. [Phase 5: eval_bridge and Stage implementations](#phase-5-eval-bridge-and-stage-implementations)
6. [Phase 6: No changes required (golden datasets)](#phase-6-no-changes-required)
7. [Phase 7: Documentation](#phase-7-documentation)
8. [Dependency graph](#dependency-graph)

---

## Phase 1: No changes required

**Depends on:** spindle-eval Phase 1 (core protocols + pipeline executor)

spindle continues to work as-is. spindle-eval's `compat.py` shim wraps the existing `get_eval_components` factory (once spindle exposes one) into the new `list[StageDef]` format. No code changes in spindle are needed until Phase 2.

**Action:** None. Verify spindle still works with spindle-eval's compat shim once spindle-eval Phase 1 lands.

---

## Phase 2: Tracker protocol alignment

**Depends on:** spindle-eval Phase 1 complete (extended `Tracker` protocol defined in `spindle_eval.protocols`)

### 2.1 Create `spindle/tracking.py`

This file does not exist yet. Create it with a `NoOpTracker` that satisfies the extended `Tracker` protocol.

**File:** `spindle/tracking.py`

```python
"""Tracker protocol and NoOpTracker for standalone spindle usage.

When running under spindle-eval, the runner injects a real tracker (MLflowTracker,
FileTracker, etc.). When running standalone, NoOpTracker routes all events to
Python's logging module at DEBUG level — silent by default, visible when opted in.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Generator

if TYPE_CHECKING:
    from spindle_eval.protocols import Tracker

logger = logging.getLogger("spindle.tracking")


class NoOpTracker:
    """Default tracker when not running under spindle-eval.

    Satisfies the spindle_eval Tracker protocol — verified by type checker.
    Routes all events/metrics to Python logging at DEBUG level.
    """

    def log_metric(self, key: str, value: float) -> None:
        logger.debug("metric %s=%s", key, value)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        logger.debug("metrics %s", metrics)

    def log_param(self, key: str, value: Any) -> None:
        logger.debug("param %s=%s", key, value)

    def log_params(self, params: dict[str, Any]) -> None:
        logger.debug("params %s", params)

    def log_event(
        self,
        service: str,
        name: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        logger.debug(
            "event %s/%s %s",
            service,
            name,
            json.dumps(payload, default=str) if payload else "{}",
        )

    def log_artifact(self, path: str) -> None:
        logger.debug("artifact %s", path)

    @contextmanager
    def start_stage(self, name: str) -> Generator[None, None, None]:
        logger.debug("stage_start %s", name)
        try:
            yield
        finally:
            logger.debug("stage_end %s", name)

    def end_run(self) -> None:
        logger.debug("run_end")
```

### 2.2 Update `pyproject.toml` — add eval optional dependency

**File:** `pyproject.toml`

Add to the `[project.optional-dependencies]` section:

```toml
[project.optional-dependencies]
eval = ["spindle-eval>=0.2.0"]
```

### 2.3 Add CI type-checking job

**File:** `.github/workflows/ci.yml` (add a new job or step)

```yaml
- name: Type check with protocol verification
  run: |
    uv pip install -e ".[eval]"
    uv run mypy spindle/tracking.py --strict
```

### 2.4 Begin emitting structured events from pipeline stages

Instrument existing pipeline components to emit events through the tracker. Each component that currently does work silently should accept a tracker and call `log_event`.

**Files to modify:**

| File | Change |
|---|---|
| `spindle/extraction/extractor.py` | Accept `tracker` in `__init__`, emit `extractor/extract.start`, `extractor/extract.complete` events with payload `{triple_count, total_tokens, elapsed_ms}` |
| `spindle/entity_resolution/resolver.py` | Accept `tracker`, emit `entity_resolution/resolve.start`, `entity_resolution/resolve.complete` with payload `{candidates, merged, elapsed_ms}` |
| `spindle/graph_store/graph_store.py` | Accept `tracker`, emit `graph_store/ingest.start`, `graph_store/ingest.complete` with payload `{triple_count, node_count, elapsed_ms}` |
| `spindle/ingestion/pipeline/pipeline.py` | Accept `tracker`, emit `preprocessor/chunk.start`, `preprocessor/chunk.complete` with payload `{chunk_count, avg_chunk_size}` |

**Pattern for each component:**

```python
from spindle.tracking import NoOpTracker

class SpindleExtractor:
    def __init__(self, ..., tracker=None):
        self._tracker = tracker or NoOpTracker()

    def extract(self, text, ontology):
        self._tracker.log_event("extractor", "extract.start", {"text_length": len(text)})
        # ... existing extraction logic ...
        self._tracker.log_event("extractor", "extract.complete", {
            "triple_count": len(triples),
            "elapsed_ms": elapsed_ms,
        })
        return triples
```

---

## Phase 3: KOS stage outputs

**Depends on:** spindle-eval Phase 3 complete (KOS metrics + `kos_loader.py` exist)

### 3.1 Ensure KOS pipeline stages return RDF file paths

This phase is about making spindle's KOS/ontology outputs consumable by spindle-eval's new KOS metrics. The KOS pipeline (vocabulary extraction → taxonomy → thesaurus → OWL synthesis) already produces RDF files. The key change is ensuring these paths are exposed in a structured way.

**What already exists in spindle v2 design:**

- `kos/kos.ttls` — SKOS thesaurus as Turtle-star
- `kos/ontology.owl` — OWL ontology
- `kos/shapes.ttl` — SHACL shapes

**What needs to happen:**

The `StageResult.outputs` dict from KOS-related stages must include these paths:

```python
StageResult(
    outputs={
        "kos_path": "kos/kos.ttls",
        "ontology_path": "kos/ontology.owl",
        "shapes_path": "kos/shapes.ttl",
    },
    metrics={},
    events=[],
)
```

This is straightforward once the Stage implementations exist (Phase 5). In Phase 3, the concrete task is to **verify that spindle's KOS pipeline writes these files to predictable paths** and document the output schema.

### 3.2 Document KOS output paths

**File:** `docs/v2/design/kos_extraction.md` (update existing)

Add a section documenting the file output contract:

```markdown
## Output Artifacts

The KOS pipeline writes the following files to `{output_root}/kos/`:

| File | Format | Content |
|---|---|---|
| `kos.ttls` | Turtle-star | SKOS thesaurus with RDF-star provenance annotations |
| `ontology.owl` | OWL/XML | OWL ontology synthesized from the thesaurus |
| `shapes.ttl` | Turtle | SHACL shapes for graph validation |

These paths are returned in `StageResult.outputs` when running under spindle-eval.
```

---

## Phase 4: Hydra config registration

**Depends on:** Nothing (can proceed independently of spindle-eval phases)

### 4.1 Create `spindle/hydra_plugin.py`

**File:** `spindle/hydra_plugin.py`

```python
"""Hydra SearchPathPlugin for spindle config composition.

When spindle is installed, this plugin injects spindle's config directory
into Hydra's search path, making spindle-specific config groups available
for composition with spindle-eval's evaluation configs.
"""

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class SpindleSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(
            provider="spindle",
            path="pkg://spindle.conf",
        )
```

### 4.2 Create `spindle/conf/` directory with pipeline config groups

**Directory structure:**

```
spindle/conf/
├── preprocessing/
│   ├── spindle_default.yaml
│   └── spindle_fast.yaml
├── kos_extraction/
│   ├── cold_start.yaml
│   └── incremental.yaml
├── ontology_synthesis/
│   └── default.yaml
└── retrieval/
    ├── local.yaml
    ├── global.yaml
    └── hybrid.yaml
```

**File: `spindle/conf/preprocessing/spindle_default.yaml`**

```yaml
# Default spindle preprocessing configuration
chunk_size: 600
overlap: 0.10
strategy: semantic
coref_model: fastcoref
ner_cascade:
  - aho_corasick
  - semantic_search
  - gliner2
```

**File: `spindle/conf/preprocessing/spindle_fast.yaml`**

```yaml
# Lighter config for CI — skip coref, use only dictionary NER
chunk_size: 600
overlap: 0.10
strategy: recursive
coref_model: null
ner_cascade:
  - aho_corasick
```

**File: `spindle/conf/kos_extraction/cold_start.yaml`**

```yaml
# LLM-based extraction from scratch (no existing KOS)
mode: cold_start
llm_model: claude-sonnet-4-20250514
max_concepts_per_chunk: 20
min_confidence: 0.6
```

**File: `spindle/conf/kos_extraction/incremental.yaml`**

```yaml
# NER cascade against existing KOS — for iterative refinement
mode: incremental
match_threshold: 0.85
promote_candidates: true
staging_review_mode: auto
```

**File: `spindle/conf/ontology_synthesis/default.yaml`**

```yaml
# OWL ontology synthesis from SKOS thesaurus
max_axioms_per_class: 10
generate_shacl: true
shacl_severity: warning
```

**File: `spindle/conf/retrieval/local.yaml`**

```yaml
strategy: local
top_k: 10
traversal_depth: 2
```

**File: `spindle/conf/retrieval/global.yaml`**

```yaml
strategy: global
top_k: 10
community_level: C0
```

**File: `spindle/conf/retrieval/hybrid.yaml`**

```yaml
strategy: hybrid
top_k: 10
traversal_depth: 2
community_level: C0
vector_weight: 0.5
graph_weight: 0.5
```

### 4.3 Register the Hydra entry point in `pyproject.toml`

**File:** `pyproject.toml`

Add the entry point:

```toml
[project.entry-points."hydra.searchpath"]
spindle = "spindle.hydra_plugin:SpindleSearchPathPlugin"
```

### 4.4 Add `hydra-core` as optional dependency

**File:** `pyproject.toml`

```toml
[project.optional-dependencies]
eval = ["spindle-eval>=0.2.0", "hydra-core>=1.3"]
```

Note: `hydra-core` is only needed when spindle runs under spindle-eval's Hydra-driven runner. It should not be a core dependency.

---

## Phase 5: eval_bridge and Stage implementations

**Depends on:** spindle-eval Phase 1 complete, spindle Phase 2 complete

This is the largest phase. spindle creates the new `get_pipeline_definition` factory and implements the `Stage` protocol for each pipeline stage.

### 5.1 Create `spindle/eval_bridge.py`

**File:** `spindle/eval_bridge.py`

```python
"""Integration bridge between spindle and spindle-eval.

Defines get_pipeline_definition(), which returns spindle's pipeline
as a list of StageDef objects that spindle-eval's generic executor
can run and evaluate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig

from spindle_eval.protocols import StageDef, StageResult

if TYPE_CHECKING:
    from spindle_eval.protocols import Tracker

from spindle.stages.preprocessing import SpindlePreprocessingStage
from spindle.stages.kos_extraction import SpindleKOSExtractionStage
from spindle.stages.ontology_synthesis import SpindleOntologySynthesisStage
from spindle.stages.retrieval import SpindleRetrievalStage
from spindle.stages.generation import SpindleGenerationStage


def get_pipeline_definition(
    cfg: DictConfig,
    *,
    tracker: Tracker,
) -> list[StageDef]:
    """Define the spindle evaluation pipeline.

    Returns stages in execution order with their dependencies.
    spindle-eval's generic executor runs these stages, computes
    metrics, checks gates, and logs everything through the tracker.
    """
    try:
        from spindle_eval.metrics import chunk_metrics, kos_metrics
    except ImportError:
        chunk_metrics = None
        kos_metrics = None

    preprocessor = SpindlePreprocessingStage(cfg, tracker=tracker)
    kos_extractor = SpindleKOSExtractionStage(cfg, tracker=tracker)
    ontology_synthesizer = SpindleOntologySynthesisStage(cfg, tracker=tracker)
    retriever = SpindleRetrievalStage(cfg, tracker=tracker)
    generator = SpindleGenerationStage(cfg, tracker=tracker)

    stages = [
        StageDef(
            name="preprocessing",
            stage=preprocessor,
            input_keys={},
            metrics=(
                [chunk_metrics.boundary_coherence, chunk_metrics.size_distribution]
                if chunk_metrics
                else []
            ),
        ),
        StageDef(
            name="kos_extraction",
            stage=kos_extractor,
            input_keys={"chunks": "preprocessing.chunks"},
            metrics=(
                [
                    kos_metrics.taxonomy_depth,
                    kos_metrics.label_quality,
                    kos_metrics.thesaurus_connectivity,
                ]
                if kos_metrics
                else []
            ),
            gate=lambda m: m.get("orphan_concept_ratio", 1.0) <= 0.3,
        ),
        StageDef(
            name="ontology_synthesis",
            stage=ontology_synthesizer,
            input_keys={"kos_path": "kos_extraction.kos_path"},
            metrics=(
                [kos_metrics.axiom_density, kos_metrics.shacl_conformance]
                if kos_metrics
                else []
            ),
        ),
        StageDef(
            name="retrieval",
            stage=retriever,
            input_keys={"graph": "ontology_synthesis.graph"},
            metrics=[],
        ),
        StageDef(
            name="generation",
            stage=generator,
            input_keys={"contexts": "retrieval.contexts"},
            metrics=[],
        ),
    ]

    return stages
```

### 5.2 Create `spindle/stages/` package with Stage implementations

**Directory structure:**

```
spindle/stages/
├── __init__.py
├── preprocessing.py
├── kos_extraction.py
├── ontology_synthesis.py
├── retrieval.py
└── generation.py
```

Each stage implements the `Stage` protocol: a `name` attribute and a `run(inputs, cfg) -> StageResult` method.

**File: `spindle/stages/__init__.py`**

```python
"""Stage implementations for spindle-eval integration."""
```

**File: `spindle/stages/preprocessing.py`**

```python
"""Preprocessing stage: document loading, chunking, coref, NER."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig

from spindle_eval.protocols import StageResult
from spindle.tracking import NoOpTracker

if TYPE_CHECKING:
    from spindle_eval.protocols import Tracker


class SpindlePreprocessingStage:
    name = "preprocessing"

    def __init__(self, cfg: DictConfig, *, tracker: Tracker | None = None):
        self._cfg = cfg
        self._tracker = tracker or NoOpTracker()

    def run(self, inputs: dict[str, Any], cfg: Any) -> StageResult:
        self._tracker.log_event("preprocessor", "chunk.start")

        # Wire up spindle's existing preprocessing pipeline:
        # 1. Load documents (Docling conversion + change detection)
        # 2. Chunk with Chonkie (semantic chunking)
        # 3. Coreference resolution (fastcoref)
        # 4. Three-pass NER cascade
        #
        # Implementation delegates to existing spindle modules:
        # - spindle.ingestion.loaders
        # - spindle.ingestion.splitters
        # - spindle.pipeline (coref, NER)

        # TODO: Wire to existing spindle preprocessing code
        chunks = []  # Placeholder — replace with actual pipeline call
        chunk_count = len(chunks)

        self._tracker.log_event("preprocessor", "chunk.complete", {
            "chunk_count": chunk_count,
        })

        return StageResult(
            outputs={"chunks": chunks},
            metrics={"num_chunks": float(chunk_count)},
        )
```

**File: `spindle/stages/kos_extraction.py`**

```python
"""KOS extraction stage: vocabulary → taxonomy → thesaurus."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig

from spindle_eval.protocols import StageResult
from spindle.tracking import NoOpTracker

if TYPE_CHECKING:
    from spindle_eval.protocols import Tracker


class SpindleKOSExtractionStage:
    name = "kos_extraction"

    def __init__(self, cfg: DictConfig, *, tracker: Tracker | None = None):
        self._cfg = cfg
        self._tracker = tracker or NoOpTracker()

    def run(self, inputs: dict[str, Any], cfg: Any) -> StageResult:
        chunks = inputs["chunks"]
        self._tracker.log_event("kos_extractor", "extract.start", {
            "chunk_count": len(chunks),
        })

        # Wire up spindle's KOS extraction pipeline:
        # 1. Vocabulary extraction (terms from chunks)
        # 2. Taxonomy construction (broader/narrower hierarchy)
        # 3. Thesaurus enrichment (related terms, definitions)
        # 4. Staging for human review (if configured)
        # 5. KOS merge
        #
        # Output: SKOS thesaurus as Turtle-star file

        # TODO: Wire to existing spindle KOS extraction code
        kos_path = "kos/kos.ttls"  # Placeholder

        self._tracker.log_event("kos_extractor", "extract.complete", {
            "kos_path": kos_path,
        })

        return StageResult(
            outputs={"kos_path": kos_path},
            metrics={},
        )
```

**File: `spindle/stages/ontology_synthesis.py`**

```python
"""Ontology synthesis stage: SKOS thesaurus → OWL + SHACL."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig

from spindle_eval.protocols import StageResult
from spindle.tracking import NoOpTracker

if TYPE_CHECKING:
    from spindle_eval.protocols import Tracker


class SpindleOntologySynthesisStage:
    name = "ontology_synthesis"

    def __init__(self, cfg: DictConfig, *, tracker: Tracker | None = None):
        self._cfg = cfg
        self._tracker = tracker or NoOpTracker()

    def run(self, inputs: dict[str, Any], cfg: Any) -> StageResult:
        kos_path = inputs["kos_path"]
        self._tracker.log_event("ontology_synth", "synthesize.start", {
            "kos_path": kos_path,
        })

        # Wire up spindle's ontology synthesis:
        # 1. Read SKOS thesaurus
        # 2. Synthesize OWL ontology (classes, properties, axioms)
        # 3. Generate SHACL shapes for validation
        # 4. Build knowledge graph in Kùzu

        # TODO: Wire to existing spindle ontology synthesis code
        ontology_path = "kos/ontology.owl"
        shapes_path = "kos/shapes.ttl"
        graph = None  # Placeholder for Kùzu graph handle

        self._tracker.log_event("ontology_synth", "synthesize.complete", {
            "ontology_path": ontology_path,
            "shapes_path": shapes_path,
        })

        return StageResult(
            outputs={
                "ontology_path": ontology_path,
                "shapes_path": shapes_path,
                "graph": graph,
            },
            metrics={},
        )
```

**File: `spindle/stages/retrieval.py`**

```python
"""Retrieval stage: query the knowledge graph."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig

from spindle_eval.protocols import StageResult
from spindle.tracking import NoOpTracker

if TYPE_CHECKING:
    from spindle_eval.protocols import Tracker


class SpindleRetrievalStage:
    name = "retrieval"

    def __init__(self, cfg: DictConfig, *, tracker: Tracker | None = None):
        self._cfg = cfg
        self._tracker = tracker or NoOpTracker()

    def run(self, inputs: dict[str, Any], cfg: Any) -> StageResult:
        graph = inputs.get("graph")

        # TODO: Wire to spindle's retrieval logic
        # Uses graph_store + vector_store for hybrid retrieval
        contexts = []  # Placeholder

        return StageResult(
            outputs={"contexts": contexts},
            metrics={},
        )
```

**File: `spindle/stages/generation.py`**

```python
"""Generation stage: produce answers from retrieved contexts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig

from spindle_eval.protocols import StageResult
from spindle.tracking import NoOpTracker

if TYPE_CHECKING:
    from spindle_eval.protocols import Tracker


class SpindleGenerationStage:
    name = "generation"

    def __init__(self, cfg: DictConfig, *, tracker: Tracker | None = None):
        self._cfg = cfg
        self._tracker = tracker or NoOpTracker()

    def run(self, inputs: dict[str, Any], cfg: Any) -> StageResult:
        contexts = inputs.get("contexts", [])

        # TODO: Wire to spindle's generation logic
        answer = ""  # Placeholder

        return StageResult(
            outputs={"answer": answer, "contexts": contexts},
            metrics={},
        )
```

### 5.3 Update `spindle/__init__.py` — export the new factory

**File:** `spindle/__init__.py`

Add:

```python
try:
    from spindle.eval_bridge import get_pipeline_definition
except ImportError:
    pass
```

### 5.4 Deprecate the old factory (if it exists)

If `get_eval_components` exists in spindle at this point, wrap it with a deprecation warning and keep it for one release cycle:

```python
import warnings

def get_eval_components(cfg):
    warnings.warn(
        "get_eval_components is deprecated. Use get_pipeline_definition instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Legacy implementation
    ...
```

---

## Phase 6: No changes required

**Depends on:** spindle-eval Phase 6 (extended golden datasets)

No spindle changes needed initially. When gold-standard KOS data is created, update the `StageDef` declarations in `eval_bridge.py` to include extrinsic KOS metrics alongside the intrinsic ones.

---

## Phase 7: Documentation

**Depends on:** All prior phases substantially complete

### 7.1 Document the Hydra config registration plugin

**File:** `docs/v2/design/hydra_config.md` (new)

Contents:
- How `SpindleSearchPathPlugin` works
- List of config groups spindle provides and their parameters
- Naming convention (prefix with `spindle_` to avoid collisions)
- How to add a new tunable parameter (add to YAML, no cross-repo change)

### 7.2 Document the Stage implementations

**File:** `docs/v2/design/eval_stages.md` (new)

Contents:
- How `get_pipeline_definition` works
- Each stage's expected inputs and `StageResult.outputs` schema
- How to add a new stage to the pipeline
- How stages map to spindle's internal modules

### 7.3 Document the event schemas

**File:** `docs/v2/design/event_schemas.md` (new)

Contents:
- List of all events spindle emits (service/name pairs)
- Payload schema for each event
- `.start` / `.complete` / `.error` naming convention
- How to add new events

---

## Dependency graph

```
Phase 1 (no spindle changes)
    │
    ▼
Phase 2 (tracking.py, event instrumentation)
    │                               Phase 4 (Hydra plugin — independent)
    ▼                                   │
Phase 5 (eval_bridge, Stage impls) ◄────┘
    │
    ▼
Phase 3 (KOS output paths — trivial once stages exist)
    │
    ▼
Phase 7 (documentation)
```

Phase 4 can proceed at any time. Phases 2 and 5 are the critical path for spindle.

---

## Summary of new files

| File | Phase | Purpose |
|---|---|---|
| `spindle/tracking.py` | 2 | `NoOpTracker` + logging-based event routing |
| `spindle/hydra_plugin.py` | 4 | Hydra `SearchPathPlugin` for config composition |
| `spindle/conf/preprocessing/spindle_default.yaml` | 4 | Default preprocessing config |
| `spindle/conf/preprocessing/spindle_fast.yaml` | 4 | CI-friendly preprocessing config |
| `spindle/conf/kos_extraction/cold_start.yaml` | 4 | LLM-based KOS extraction config |
| `spindle/conf/kos_extraction/incremental.yaml` | 4 | Incremental NER-based KOS extraction config |
| `spindle/conf/ontology_synthesis/default.yaml` | 4 | Ontology synthesis config |
| `spindle/conf/retrieval/local.yaml` | 4 | Local retrieval config |
| `spindle/conf/retrieval/global.yaml` | 4 | Global retrieval config |
| `spindle/conf/retrieval/hybrid.yaml` | 4 | Hybrid retrieval config |
| `spindle/eval_bridge.py` | 5 | `get_pipeline_definition()` factory |
| `spindle/stages/__init__.py` | 5 | Stage package init |
| `spindle/stages/preprocessing.py` | 5 | Preprocessing stage |
| `spindle/stages/kos_extraction.py` | 5 | KOS extraction stage |
| `spindle/stages/ontology_synthesis.py` | 5 | Ontology synthesis stage |
| `spindle/stages/retrieval.py` | 5 | Retrieval stage |
| `spindle/stages/generation.py` | 5 | Generation stage |
| `docs/v2/design/hydra_config.md` | 7 | Hydra plugin documentation |
| `docs/v2/design/eval_stages.md` | 7 | Stage implementation documentation |
| `docs/v2/design/event_schemas.md` | 7 | Event schema documentation |

## Summary of modified files

| File | Phase | Change |
|---|---|---|
| `pyproject.toml` | 2, 4 | Add `eval` and `hydra-core` optional deps, Hydra entry point |
| `spindle/__init__.py` | 5 | Export `get_pipeline_definition` |
| `spindle/extraction/extractor.py` | 2 | Accept tracker, emit events |
| `spindle/entity_resolution/resolver.py` | 2 | Accept tracker, emit events |
| `spindle/graph_store/graph_store.py` | 2 | Accept tracker, emit events |
| `spindle/ingestion/pipeline/pipeline.py` | 2 | Accept tracker, emit events |
| `.github/workflows/ci.yml` | 2 | Add type-check job with protocol verification |
| `docs/v2/design/kos_extraction.md` | 3 | Document KOS output file paths |
