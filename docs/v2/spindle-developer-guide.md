# Developing Spindle for spindle-eval Compatibility

This guide explains everything you need to know while building the `spindle` package so that it integrates seamlessly with `spindle-eval`. It covers the integration contract, data models, component interfaces, configuration expectations, instrumentation hooks, and common pitfalls.

---

## Table of Contents

1. [How spindle-eval discovers spindle](#1-how-spindle-eval-discovers-spindle)
2. [The component factory contract](#2-the-component-factory-contract)
3. [Data models you must use](#3-data-models-you-must-use)
4. [Component protocols in detail](#4-component-protocols-in-detail)
5. [Hydra configuration expectations](#5-hydra-configuration-expectations)
6. [The Tracker protocol and instrumentation](#6-the-tracker-protocol-and-instrumentation)
7. [Stage gates and quality thresholds](#7-stage-gates-and-quality-thresholds)
8. [Metrics spindle-eval computes from your outputs](#8-metrics-spindle-eval-computes-from-your-outputs)
9. [Retriever and Generator flexibility](#9-retriever-and-generator-flexibility)
10. [Mock fallback behavior](#10-mock-fallback-behavior)
11. [Golden dataset format](#11-golden-dataset-format)
12. [Sweep parameters your configs must support](#12-sweep-parameters-your-configs-must-support)
13. [CI regression detection](#13-ci-regression-detection)
14. [Production monitoring hooks](#14-production-monitoring-hooks)
15. [Dependency boundaries](#15-dependency-boundaries)
16. [Recommended spindle package structure](#16-recommended-spindle-package-structure)
17. [End-to-end integration example](#17-end-to-end-integration-example)
18. [Common mistakes and how to avoid them](#18-common-mistakes-and-how-to-avoid-them)

---

## 1. How spindle-eval discovers spindle

spindle-eval uses dynamic import discovery at runtime. The runner does the following:

```python
import importlib

spindle_module = importlib.import_module("spindle")
factory = getattr(spindle_module, "get_eval_components", None)
```

This means:

- spindle must be importable as `import spindle` (i.e., the package name on `sys.path` must be `spindle`).
- The top-level `spindle` module (or `spindle/__init__.py`) must expose a callable named `get_eval_components`.
- If spindle is not installed or `get_eval_components` is not found, spindle-eval falls back to mocks (if `allow_mock_fallback` is `true` in the Hydra config).

**What you need in spindle:**

```python
# spindle/__init__.py
from spindle.factory import get_eval_components

__all__ = ["get_eval_components"]
```

---

## 2. The component factory contract

`get_eval_components` receives the full Hydra `DictConfig` and must return a dict with these exact keys:

```python
REQUIRED_COMPONENT_KEYS = {
    "preprocessor",
    "ontology_extractor",
    "triple_extractor",
    "graph_store",
    "retriever",
    "generator",
}
```

**Signature:**

```python
from omegaconf import DictConfig
from typing import Any

def get_eval_components(cfg: DictConfig) -> dict[str, Any]:
    ...
```

**Behavior rules:**

| Scenario | Result |
|---|---|
| All 6 keys returned | spindle-eval uses all spindle components |
| Some keys missing, `allow_mock_fallback=true` | Missing components filled with mocks |
| Some keys missing, `allow_mock_fallback=false` | `RuntimeError` with names of missing keys |
| Returns `None` or non-dict | Treated as "spindle not available" |

If you are building spindle incrementally, you can return only the components that are ready. spindle-eval will use mocks for the rest:

```python
def get_eval_components(cfg: DictConfig) -> dict[str, Any]:
    """Return only the components that are implemented so far."""
    from spindle.preprocessing import SpindlePreprocessor
    from spindle.extraction import SpindleTripleExtractor

    return {
        "preprocessor": SpindlePreprocessor(cfg),
        "triple_extractor": SpindleTripleExtractor(cfg),
        # retriever, generator, etc. will fall back to mocks
    }
```

---

## 3. Data models you must use

spindle-eval defines three core dataclasses in `spindle_eval.protocols`. Your components must produce and consume these exact types. Do **not** define your own versions — import them from spindle-eval.

### Chunk

```python
from dataclasses import dataclass
from typing import Any

@dataclass
class Chunk:
    text: str              # The chunk text content
    metadata: dict[str, Any]  # Arbitrary metadata (page, section, etc.)
    source_id: str         # Identifier for the source document
```

### Triple

```python
@dataclass
class Triple:
    subject: str           # Entity name (normalized)
    predicate: str         # Relation type (e.g., "WORKS_AT")
    object: str            # Entity name (normalized)
    confidence: float      # Extraction confidence in [0, 1]
    source_chunk_id: str   # ID of the chunk this triple was extracted from
```

### OntologySchema

```python
@dataclass
class OntologySchema:
    entity_types: list[str]      # e.g., ["person", "organization", "location"]
    relation_types: list[str]    # e.g., ["WORKS_AT", "LOCATED_IN", "FOUNDED"]
```

**Key constraint:** These dataclasses are used directly in metric computation. For example, graph connectivity metrics iterate over `triple.subject` and `triple.object` to build adjacency lists, and extraction metrics compare `(triple.subject, triple.predicate, triple.object)` tuples. If you wrap or substitute these with incompatible types, metric computation will fail silently or raise errors.

### Importing from spindle-eval

Add `spindle-eval` as a dependency (or optional dependency) in your spindle package:

```toml
# spindle/pyproject.toml
[project.optional-dependencies]
eval = ["spindle-eval>=0.1.0"]
```

Then import the types:

```python
from spindle_eval.protocols import Chunk, Triple, OntologySchema
```

Alternatively, if you want spindle to work without spindle-eval installed, use structural subtyping — your classes just need to have the same fields. Python's `Protocol` matching is structural, so any dataclass with the same attributes will satisfy the protocol. However, explicitly importing the types is safer and prevents subtle drift.

---

## 4. Component protocols in detail

Each component has a specific call signature that spindle-eval invokes. Here is exactly how the runner calls each one.

### Preprocessor

```python
class Preprocessor(Protocol):
    def __call__(self, cfg: Any) -> list[Chunk]: ...
```

**How it's called:**

```python
chunks = components["preprocessor"](cfg.preprocessing)
```

The preprocessor receives the `cfg.preprocessing` sub-config. It must return a list of `Chunk` objects. The config will have at minimum these fields (from the default config):

| Field | Type | Default | Purpose |
|---|---|---|---|
| `chunk_size` | int | 600 | Target characters per chunk |
| `overlap` | float | 0.10 | Fractional overlap between chunks |
| `strategy` | str | `"recursive"` | Chunking strategy |
| `num_chunks` | int | 0 | 0 means derive from documents |

**Example implementation:**

```python
from spindle_eval.protocols import Chunk

class SpindlePreprocessor:
    def __init__(self, documents: list[str]):
        self._documents = documents

    def __call__(self, cfg) -> list[Chunk]:
        chunk_size = int(cfg.chunk_size)
        overlap = float(cfg.overlap)
        strategy = str(cfg.strategy)

        chunks = []
        for doc_idx, doc in enumerate(self._documents):
            doc_chunks = self._chunk_document(doc, chunk_size, overlap, strategy)
            for i, text in enumerate(doc_chunks):
                chunks.append(Chunk(
                    text=text,
                    metadata={"chunk_index": i, "doc_index": doc_idx},
                    source_id=f"doc_{doc_idx}",
                ))
        return chunks

    def _chunk_document(self, text, size, overlap, strategy):
        # Your chunking logic here
        ...
```

### OntologyExtractor

```python
class OntologyExtractor(Protocol):
    def __call__(self, chunks: list[Chunk], cfg: Any) -> OntologySchema: ...
```

**How it's called:**

```python
ontology = components["ontology_extractor"](chunks, cfg.ontology)
```

Config fields from `conf/ontology/hybrid.yaml`:

| Field | Type | Default | Purpose |
|---|---|---|---|
| `mode` | str | `"hybrid"` | `schema_first`, `schema_free`, or `hybrid` |
| `discover_entity_types` | bool | `true` | Whether to discover types from text |
| `entity_types` | list[str] | `["person", ...]` | Predefined entity types |
| `relation_types` | list[str] | `["WORKS_AT", ...]` | Predefined relation types |

**Example implementation:**

```python
from spindle_eval.protocols import Chunk, OntologySchema

class SpindleOntologyExtractor:
    def __call__(self, chunks: list[Chunk], cfg) -> OntologySchema:
        mode = str(cfg.mode)

        if mode == "schema_first":
            return OntologySchema(
                entity_types=list(cfg.entity_types),
                relation_types=list(cfg.relation_types),
            )

        discovered = self._discover_types_from_chunks(chunks)

        if mode == "hybrid":
            entity_types = list(set(cfg.entity_types) | set(discovered["entities"]))
            relation_types = list(set(cfg.relation_types) | set(discovered["relations"]))
        else:
            entity_types = discovered["entities"]
            relation_types = discovered["relations"]

        return OntologySchema(
            entity_types=entity_types,
            relation_types=relation_types,
        )

    def _discover_types_from_chunks(self, chunks):
        # Your NER / LLM-based type discovery logic
        ...
```

### TripleExtractor

```python
class TripleExtractor(Protocol):
    def __call__(
        self,
        chunks: list[Chunk],
        ontology: OntologySchema,
        cfg: Any,
    ) -> list[Triple]: ...
```

**How it's called:**

```python
triples = components["triple_extractor"](chunks, ontology, cfg.extraction)
```

Config fields from `conf/extraction/llm.yaml`:

| Field | Type | Default | Purpose |
|---|---|---|---|
| `backend` | str | `"llm"` | `llm`, `nlp`, or `finetuned` |
| `model` | str | `"gpt-4o-mini"` | LLM model for extraction |
| `max_gleanings` | int | 1 | Number of re-extraction passes |
| `entity_resolution_threshold` | float | 0.85 | Similarity threshold for deduplication |

**Critical:** The `entity_resolution_threshold` is a sweep parameter. Your triple extractor must respect this value and use it for entity deduplication/resolution. If you hard-code this, sweeps over it will have no effect.

**Example implementation:**

```python
from spindle_eval.protocols import Chunk, OntologySchema, Triple

class SpindleTripleExtractor:
    def __call__(self, chunks, ontology, cfg) -> list[Triple]:
        backend = str(cfg.backend)
        er_threshold = float(cfg.entity_resolution_threshold)

        raw_triples = []
        for chunk in chunks:
            extracted = self._extract_from_chunk(chunk, ontology, cfg)
            raw_triples.extend(extracted)

        resolved = self._resolve_entities(raw_triples, er_threshold)
        return resolved

    def _extract_from_chunk(self, chunk, ontology, cfg):
        # Call LLM / NLP pipeline to extract triples
        # Return list[Triple] with source_chunk_id = chunk.source_id
        ...

    def _resolve_entities(self, triples, threshold):
        # Deduplicate entities based on string similarity
        # threshold controls how aggressive deduplication is
        ...
```

### GraphStore

```python
class GraphStore(Protocol):
    def ingest(self, triples: list[Triple]) -> None: ...
    def query_subgraph(self, query: str, cfg: Any) -> dict: ...
```

**How it's called:**

```python
graph_store.ingest(triples)
# query_subgraph is available for retriever implementations
```

The `ingest` method is called after triple extraction with all extracted triples. The `query_subgraph` method should return a dict with these expected keys (based on the mock):

```python
{
    "entities": ["Entity_0", "Entity_1", ...],
    "triples": [
        {"s": "Entity_0", "p": "RELATES_TO", "o": "Entity_1"},
        ...
    ],
    "summary": "Subgraph summary text",
}
```

### Retriever

```python
class Retriever(Protocol):
    def retrieve(self, query: str, cfg: Any) -> list[dict]: ...
```

**How it's called:**

```python
contexts = retriever.retrieve(query, cfg.retrieval)
# OR if callable:
contexts = retriever(query, cfg.retrieval)
```

Config fields from `conf/retrieval/hybrid.yaml`:

| Field | Type | Default | Purpose |
|---|---|---|---|
| `strategy` | str | `"hybrid"` | `hybrid`, `local`, `global`, `drift` |
| `top_k` | int | 10 | Number of context chunks to return |
| `traversal_depth` | int | 2 | Graph traversal hops |
| `community_level` | str | `"C0"` | Community hierarchy level |

**Return format is critical.** Each dict in the returned list **must** have a `"content"` key. This is extracted in the runner for Ragas evaluation:

```python
# How the runner uses your retriever output:
contexts = retriever.retrieve(example.question, cfg.retrieval)
context_texts = [item.get("content", "") for item in contexts]
```

**Example implementation:**

```python
class SpindleRetriever:
    def __init__(self, graph_store, vector_store):
        self._graph_store = graph_store
        self._vector_store = vector_store

    def retrieve(self, query: str, cfg) -> list[dict]:
        top_k = int(cfg.top_k)
        strategy = str(cfg.strategy)
        traversal_depth = int(cfg.traversal_depth)

        if strategy == "local":
            results = self._local_search(query, top_k, traversal_depth)
        elif strategy == "global":
            results = self._global_search(query, top_k, cfg.community_level)
        elif strategy == "hybrid":
            results = self._hybrid_search(query, top_k, traversal_depth)
        else:
            results = self._drift_search(query, top_k)

        return [
            {
                "content": r["text"],   # REQUIRED key
                "score": r["score"],
                "source_id": r["source"],
            }
            for r in results[:top_k]
        ]
```

### Generator

```python
class Generator(Protocol):
    def generate(self, query: str, contexts: list[dict], cfg: Any) -> str: ...
```

**How it's called:**

```python
answer = generator.generate(query, contexts, cfg.generation)
# OR if callable:
answer = generator(query, contexts, cfg.generation)
```

Config fields from `conf/generation/gpt4.yaml`:

| Field | Type | Default | Purpose |
|---|---|---|---|
| `model` | str | `"gpt-4o"` | LLM model name |
| `temperature` | float | 0.2 | Sampling temperature |
| `max_tokens` | int | 1024 | Max response tokens |

**Example implementation:**

```python
class SpindleGenerator:
    def generate(self, query: str, contexts: list[dict], cfg) -> str:
        model = str(cfg.model)
        temperature = float(cfg.temperature)
        max_tokens = int(cfg.max_tokens)

        context_text = "\n\n".join(c.get("content", "") for c in contexts)
        prompt = f"Context:\n{context_text}\n\nQuestion: {query}\nAnswer:"

        response = self._call_llm(prompt, model, temperature, max_tokens)
        return response
```

---

## 5. Hydra configuration expectations

spindle-eval uses [Hydra](https://hydra.cc/) for configuration management. The full config is composed from YAML files in groups:

```
conf/
├── config.yaml            # Root config
├── preprocessing/          # default.yaml, small_chunks.yaml, large_chunks.yaml
├── ontology/              # hybrid.yaml, schema_first.yaml, schema_free.yaml
├── extraction/            # llm.yaml, nlp.yaml, finetuned.yaml
├── retrieval/             # hybrid.yaml, local.yaml, global.yaml, drift.yaml
├── generation/            # gpt4.yaml, claude.yaml, gemini.yaml
├── evaluation/            # quick.yaml, full.yaml
└── sweep/                 # none.yaml, chunk_size.yaml, retrieval.yaml, er_threshold.yaml
```

Your components receive `cfg.<group>` sub-configs, which are `DictConfig` objects from OmegaConf. You access fields with attribute access (`cfg.chunk_size`) or dict access (`cfg["chunk_size"]`).

**Important:** All config values arrive as OmegaConf types. Always cast to Python types explicitly:

```python
# Do this:
chunk_size = int(cfg.chunk_size)
overlap = float(cfg.overlap)
strategy = str(cfg.strategy)

# Not this (may cause issues with some libraries):
chunk_size = cfg.chunk_size  # Still an OmegaConf node
```

### Adding new config parameters

If spindle introduces a new tunable parameter, the corresponding Hydra config file in spindle-eval should be updated. Until it is, your component should use sensible defaults when a config key is missing:

```python
def __call__(self, chunks, ontology, cfg) -> list[Triple]:
    max_gleanings = int(getattr(cfg, "max_gleanings", 1))
    new_param = float(getattr(cfg, "my_new_param", 0.5))
    ...
```

---

## 6. The Tracker protocol and instrumentation

spindle-eval provides an `MLflowTracker` that implements the `Tracker` protocol. The recommended architecture is that spindle defines a `Tracker` protocol internally and accepts a tracker instance at initialization, defaulting to a no-op:

```python
# spindle/tracking.py
from typing import Any, Protocol


class Tracker(Protocol):
    def log_metric(self, key: str, value: float) -> None: ...
    def log_params(self, params: dict[str, Any]) -> None: ...
    def log_param(self, key: str, value: Any) -> None: ...
    def log_metrics(self, metrics: dict[str, float]) -> None: ...


class NoOpTracker:
    """Zero-overhead default when not running under spindle-eval."""

    def log_metric(self, key: str, value: float) -> None:
        pass

    def log_params(self, params: dict[str, Any]) -> None:
        pass

    def log_param(self, key: str, value: Any) -> None:
        pass

    def log_metrics(self, metrics: dict[str, float]) -> None:
        pass
```

Then in your pipeline stages, accept and use the tracker:

```python
class SpindleTripleExtractor:
    def __init__(self, tracker: Tracker | None = None):
        self._tracker = tracker or NoOpTracker()

    def __call__(self, chunks, ontology, cfg) -> list[Triple]:
        triples = self._extract(chunks, ontology, cfg)

        self._tracker.log_metrics({
            "extraction_time_ms": elapsed_ms,
            "raw_triples_before_er": len(raw_triples),
            "triples_after_er": len(triples),
            "er_merge_ratio": 1 - len(triples) / max(len(raw_triples), 1),
        })

        return triples
```

When spindle-eval runs the pipeline, it creates an `MLflowTracker` and can inject it via the component factory:

```python
def get_eval_components(cfg: DictConfig) -> dict[str, Any]:
    from spindle_eval.tracking import MLflowTracker

    tracker = MLflowTracker()

    return {
        "preprocessor": SpindlePreprocessor(documents, tracker=tracker),
        "ontology_extractor": SpindleOntologyExtractor(tracker=tracker),
        "triple_extractor": SpindleTripleExtractor(tracker=tracker),
        "graph_store": SpindleGraphStore(tracker=tracker),
        "retriever": SpindleRetriever(graph_store, tracker=tracker),
        "generator": SpindleGenerator(tracker=tracker),
    }
```

This way, spindle never imports MLflow directly. When used standalone (not under spindle-eval), the `NoOpTracker` means zero overhead.

---

## 7. Stage gates and quality thresholds

spindle-eval enforces a stage gate after Stage 3 (triple extraction):

```python
extraction_f1 = triple_prf1_exact(triples, triples).f1
if extraction_f1 < 0.7:
    raise RuntimeError("Stage gate failed: entity extraction F1 < 0.7")
```

Currently this compares triples against themselves (a placeholder), which always passes. Once golden triples are available, this will compare against reference triples with a real threshold.

**What this means for spindle development:**

- Your triple extractor output will be compared against reference triples using exact-match P/R/F1 on `(subject, predicate, object)` tuples.
- Normalize entity names consistently. `"Jane Doe"` and `"jane doe"` are different in exact match.
- Normalize relation types to match the ontology schema. `"works_at"` vs `"WORKS_AT"` will cause mismatches.
- If F1 drops below 0.7, the entire pipeline run is aborted before reaching RAG evaluation.

**Soft matching is also available** (threshold-based string similarity), so even if exact match is strict, the framework can fall back to fuzzy comparison. Design your entity resolution to produce consistent, normalized outputs.

---

## 8. Metrics spindle-eval computes from your outputs

Understanding what metrics are derived from your outputs helps you build components that produce evaluation-friendly data.

### From Preprocessor output (`list[Chunk]`)

| Metric | Computation |
|---|---|
| `num_chunks` | `len(chunks)` |
| `chunk_boundary_coherence` | Fraction of chunks whose text ends with `.`, `!`, `?`, `:`, `;`, or `\n` |

**Implication:** Design your chunker to split at sentence or paragraph boundaries. Coherence score penalizes mid-sentence splits.

### From OntologyExtractor output (`OntologySchema`)

| Metric | Computation |
|---|---|
| `entity_type_count` | `len(ontology.entity_types)` |
| `relation_type_count` | `len(ontology.relation_types)` |

### From TripleExtractor output (`list[Triple]`)

| Metric | Computation |
|---|---|
| `triple_count` | `len(triples)` |
| `entity_extraction_f1` | Exact-match F1 against reference triples |
| `num_nodes` | Unique entities across all `subject` and `object` fields |
| `num_edges` | Unique undirected `(subject, object)` pairs |
| `average_degree` | Mean neighbor count per node |
| `num_connected_components` | Number of disconnected subgraphs |
| `giant_component_ratio` | Largest component size / total nodes |
| `edge_density` | `2 * edges / (nodes * (nodes - 1))` |

**Implications for your triple extractor:**

- **Entity normalization matters.** If the same entity appears as `"Acme Labs"`, `"ACME Labs"`, and `"Acme Laboratories"`, graph metrics will count them as separate nodes, inflating `num_nodes` and fragmenting connectivity.
- **`source_chunk_id` must be set correctly.** It links triples back to their source chunks and is used for provenance tracking.
- **`confidence` should be calibrated.** While not currently used in gating, it will likely be used for confidence-weighted metrics and filtering.
- **Aim for a connected graph.** A `giant_component_ratio` near 1.0 indicates good entity resolution. Many disconnected components suggests entity resolution is too conservative.

### From Retriever output (`list[dict]`)

| Metric | Computation |
|---|---|
| Faithfulness | Ragas: are claims in the answer supported by retrieved context? |
| Context Recall | Ragas: does the retrieved context cover the reference answer? |
| Context Precision | Ragas: is retrieved context relevant to the question? |
| Answer Relevancy | Ragas: is the answer relevant to the question? |
| Answer Correctness | Ragas: does the answer match the reference? |

**Implication:** The `"content"` field in each retriever result dict is what gets evaluated. Make it a clean, readable text passage, not a JSON blob or metadata dump.

### B-CUBED and CEAF (entity resolution quality)

These are available but must be called separately with predicted and reference cluster mappings:

```python
from spindle_eval.metrics.graph_metrics import bcubed_scores, ceaf_scores

scores = bcubed_scores(
    predicted_clusters={"entity_a": "cluster_1", "entity_b": "cluster_1", ...},
    reference_clusters={"entity_a": "cluster_1", "entity_b": "cluster_2", ...},
)
```

If your entity resolution produces cluster assignments, expose them so spindle-eval can compute these metrics.

---

## 9. Retriever and Generator flexibility

spindle-eval supports both callable and method-based components:

```python
# Both of these work for retriever:
def my_retriever(query: str, cfg: Any) -> list[dict]:
    ...

class MyRetriever:
    def retrieve(self, query: str, cfg: Any) -> list[dict]:
        ...

# Both of these work for generator:
def my_generator(query: str, contexts: list[dict], cfg: Any) -> str:
    ...

class MyGenerator:
    def generate(self, query: str, contexts: list[dict], cfg: Any) -> str:
        ...
```

The runner checks for callable first, then for the method name. Use whichever pattern fits your architecture.

---

## 10. Mock fallback behavior

When spindle is unavailable or returns incomplete components, spindle-eval uses mocks. Understanding what mocks do helps you know what behavior to replace:

| Component | Mock behavior |
|---|---|
| `preprocessor` | Returns N synthetic chunks with repeated text |
| `ontology_extractor` | Returns 4 entity types, 4 relation types (fixed) |
| `triple_extractor` | Returns 3 synthetic triples from first 3 chunks |
| `graph_store` | In-memory list; `query_subgraph` returns first 5 entities |
| `retriever` | Returns `top_k` synthetic context strings |
| `generator` | Returns `"Mock answer for: {query}..."` |

The runner logs `component_source` as either `"spindle"` or `"mocks"` so you can filter experiment results.

---

## 11. Golden dataset format

spindle-eval loads evaluation datasets from JSONL files. Each line is a JSON object:

```json
{
  "question": "Who founded Acme Labs?",
  "answer": "Acme Labs was founded by Jane Doe.",
  "contexts": ["Jane Doe founded Acme Labs in 2018."],
  "question_type": "factoid",
  "difficulty": "easy",
  "metadata": {"source": "seed"}
}
```

**Valid `question_type` values:**

| Type | Description |
|---|---|
| `factoid` | Single-fact lookup |
| `multi_hop_2` | Requires connecting 2 pieces of information |
| `multi_hop_3_plus` | Requires 3+ hops across the knowledge graph |
| `comparison` | Comparing two or more entities |
| `summarization` | Aggregating across multiple sources |
| `temporal` | Time-dependent reasoning |
| `negation` | Questions about what is NOT true |
| `unanswerable` | No answer exists in the knowledge base |

**Why this matters for spindle:** Your retriever and generator will be evaluated on all these question types. Multi-hop questions specifically test graph traversal quality. If your retriever only does vector similarity without graph traversal, `multi_hop_3_plus` scores will suffer.

---

## 12. Sweep parameters your configs must support

spindle-eval defines Optuna sweeps over these parameters. Your components **must** read these from the config rather than hard-coding them, or sweeps will have no effect.

### Chunk size sweep (`conf/sweep/chunk_size.yaml`)

```yaml
params:
  preprocessing.chunk_size: choice(128,256,384,512,640,768,896,1024,1280,1536,1792,2048)
  preprocessing.overlap: choice(0.0,0.1,0.2)
```

Your preprocessor must dynamically use `cfg.chunk_size` and `cfg.overlap`.

### Retrieval sweep (`conf/sweep/retrieval.yaml`)

```yaml
params:
  retrieval.top_k: choice(5,10,20)
  retrieval.traversal_depth: choice(1,2,3)
  retrieval.community_level: choice(C0,C1,C2)
```

Your retriever must respect all three parameters.

### Entity resolution threshold sweep (`conf/sweep/er_threshold.yaml`)

```yaml
params:
  extraction.entity_resolution_threshold: choice(0.7,0.75,0.8,0.85,0.9,0.95)
```

Your triple extractor must use `cfg.entity_resolution_threshold` for entity deduplication.

### General principle

If a parameter appears in a Hydra YAML file, your component should read it from the config object. This is the single most important rule for sweep compatibility. A sweep that silently does nothing is worse than no sweep at all — it wastes compute and produces misleading results.

---

## 13. CI regression detection

Every PR against spindle triggers a CI evaluation:

1. spindle-eval runs a quick eval (50 examples).
2. Current metrics are compared against baselines in `baselines/metrics.json`.
3. If any metric drops below its tolerance, the CI check fails.

**Default tolerances:**

| Metric | Max allowed regression |
|---|---|
| `faithfulness` | -0.02 |
| `answer_relevancy` | -0.03 |
| `context_precision` | -0.05 |

**What this means for spindle development:**

- Changes that degrade retrieval quality by more than 5% context precision will block the PR.
- Changes that degrade generation faithfulness by more than 2% will block the PR.
- Run the quick eval locally before pushing to catch regressions early:

```bash
python -m spindle_eval.runner evaluation=quick
```

- If a regression is intentional (e.g., trading precision for recall), update the baseline file and document the tradeoff.

---

## 14. Production monitoring hooks

spindle-eval provides two production monitoring utilities your pipeline should be aware of.

### Feedback loop

The feedback loop converts low-scoring production traces from Langfuse into new golden dataset entries:

```python
from spindle_eval.production.feedback_loop import TraceScore, append_failures_to_golden_dataset

# After collecting production traces with scores:
failures = [
    TraceScore(
        trace_id="trace_abc123",
        score=0.3,
        question="What is the revenue of Acme Labs?",
        answer="The revenue is $10M.",  # Incorrect answer from production
        contexts=["Acme Labs reported $50M in revenue."],
        metadata={"environment": "production"},
    )
]
append_failures_to_golden_dataset("golden_data/questions.jsonl", failures)
```

**For spindle:** If your pipeline logs traces to Langfuse (via the OpenTelemetry integration), include enough information in each trace to reconstruct the question, answer, and contexts. This enables the feedback loop.

### KG staleness monitoring

```python
from spindle_eval.production.staleness import DocumentExtractionState, stale_document_ids
from datetime import datetime, timedelta, timezone

states = [
    DocumentExtractionState(
        doc_id="doc_001",
        content_hash="abc123",
        last_extracted=datetime(2025, 1, 1, tzinfo=timezone.utc),
        policy_window=timedelta(days=7),
    ),
]
stale_ids = stale_document_ids(states)
```

**For spindle:** Track when each document was last extracted into the knowledge graph. Expose a method to query extraction timestamps so spindle-eval can detect stale documents. The staleness windows are:

| Source type | Window |
|---|---|
| `regulatory` | 1 day |
| `product` | 7 days |
| `general` | 14 days |

---

## 15. Dependency boundaries

spindle-eval deliberately keeps spindle as an **optional** dependency. The design philosophy:

- `pip install spindle` — pipeline only, no eval infrastructure.
- `pip install spindle-eval` — eval infrastructure, uses mocks if spindle is absent.
- `pip install spindle-eval[spindle]` — both packages together.

**For spindle's dependencies:**

- Do **not** depend on `mlflow`, `ragas`, `optuna`, `hydra-core`, or `langfuse` in spindle's core dependencies. These belong to spindle-eval.
- spindle can optionally depend on `spindle-eval` for the data model imports, but this should be an optional dependency at most.
- spindle's core should only have pipeline-relevant dependencies (LLM SDKs, graph DB clients, NLP libraries, etc.).

If you need the `Chunk`, `Triple`, and `OntologySchema` types without installing spindle-eval, define structurally compatible dataclasses in spindle and rely on Python's structural subtyping:

```python
# spindle/models.py — structurally compatible, no spindle-eval dependency
from dataclasses import dataclass
from typing import Any


@dataclass
class Chunk:
    text: str
    metadata: dict[str, Any]
    source_id: str


@dataclass
class Triple:
    subject: str
    predicate: str
    object: str
    confidence: float
    source_chunk_id: str


@dataclass
class OntologySchema:
    entity_types: list[str]
    relation_types: list[str]
```

Because the protocols use structural typing (`Protocol`), instances of these classes will satisfy spindle-eval's type checks as long as the field names and types match.

---

## 16. Recommended spindle package structure

```
spindle/
├── __init__.py                  # Exports get_eval_components
├── factory.py                   # Component factory for spindle-eval
├── models.py                    # Chunk, Triple, OntologySchema (or import from spindle-eval)
├── tracking.py                  # Tracker protocol + NoOpTracker
├── preprocessing/
│   ├── __init__.py
│   └── chunker.py               # Preprocessor implementation
├── ontology/
│   ├── __init__.py
│   └── extractor.py             # OntologyExtractor implementation
├── extraction/
│   ├── __init__.py
│   ├── triple_extractor.py      # TripleExtractor implementation
│   └── entity_resolution.py     # Entity dedup with configurable threshold
├── graph/
│   ├── __init__.py
│   └── store.py                 # GraphStore implementation
├── retrieval/
│   ├── __init__.py
│   ├── retriever.py             # Retriever implementation
│   └── strategies/              # local, global, hybrid, drift
├── generation/
│   ├── __init__.py
│   └── generator.py             # Generator implementation
└── pyproject.toml
```

---

## 17. End-to-end integration example

Here is a complete, minimal `factory.py` that wires everything together:

```python
"""Component factory for spindle-eval integration."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from spindle.models import Chunk, OntologySchema, Triple
from spindle.preprocessing.chunker import SpindleChunker
from spindle.ontology.extractor import SpindleOntologyExtractor
from spindle.extraction.triple_extractor import SpindleTripleExtractor
from spindle.graph.store import SpindleGraphStore
from spindle.retrieval.retriever import SpindleRetriever
from spindle.generation.generator import SpindleGenerator


def get_eval_components(cfg: DictConfig) -> dict[str, Any]:
    """Build and return pipeline components for spindle-eval.

    This is the single integration point between spindle and spindle-eval.
    The runner calls this function and uses the returned components to
    orchestrate the evaluation pipeline.
    """
    documents = _load_documents(cfg)
    graph_store = SpindleGraphStore()

    return {
        "preprocessor": SpindleChunker(documents),
        "ontology_extractor": SpindleOntologyExtractor(),
        "triple_extractor": SpindleTripleExtractor(),
        "graph_store": graph_store,
        "retriever": SpindleRetriever(graph_store),
        "generator": SpindleGenerator(),
    }


def _load_documents(cfg: DictConfig) -> list[str]:
    """Load source documents for preprocessing."""
    # Implement based on your document storage approach
    ...
```

And the corresponding `__init__.py`:

```python
# spindle/__init__.py
from spindle.factory import get_eval_components

__all__ = ["get_eval_components"]
```

### Running the evaluation

```bash
# Start tracking infrastructure
cd spindle-eval
./scripts/start-local-tracking.sh
source deploy/.env.local

# Run with default config
python -m spindle_eval.runner

# Run with config overrides
python -m spindle_eval.runner preprocessing=small_chunks retrieval=local

# Run a parameter sweep
python -m spindle_eval.runner --multirun sweep=chunk_size

# Run quick eval for CI
python -m spindle_eval.runner evaluation=quick
```

---

## 18. Common mistakes and how to avoid them

### 1. Hard-coding swept parameters

```python
# WRONG: sweep over entity_resolution_threshold has no effect
class TripleExtractor:
    def __call__(self, chunks, ontology, cfg):
        threshold = 0.85  # Hard-coded!
        ...

# RIGHT: reads from config
class TripleExtractor:
    def __call__(self, chunks, ontology, cfg):
        threshold = float(cfg.entity_resolution_threshold)
        ...
```

### 2. Missing `"content"` key in retriever output

```python
# WRONG: runner extracts item.get("content", "") and gets empty strings
return [{"text": chunk.text, "score": 0.9}]

# RIGHT: use "content" as the key
return [{"content": chunk.text, "score": 0.9}]
```

### 3. Inconsistent entity normalization

```python
# WRONG: same entity appears as multiple nodes in the graph
Triple(subject="Jane Doe", predicate="FOUNDED", object="Acme Labs", ...)
Triple(subject="jane doe", predicate="WORKS_AT", object="ACME Labs", ...)

# RIGHT: normalize before returning
Triple(subject="Jane Doe", predicate="FOUNDED", object="Acme Labs", ...)
Triple(subject="Jane Doe", predicate="WORKS_AT", object="Acme Labs", ...)
```

### 4. Returning wrong types from the factory

```python
# WRONG: returning an instance that will be called as a function
return {"preprocessor": SpindleChunker}  # Class, not instance

# RIGHT: return a callable instance or function
return {"preprocessor": SpindleChunker(documents)}  # Instance with __call__
```

### 5. Not handling missing config keys gracefully

```python
# WRONG: crashes when spindle-eval config doesn't have your new param
new_param = cfg.my_new_experimental_param  # AttributeError

# RIGHT: use getattr with a default
new_param = float(getattr(cfg, "my_new_experimental_param", 0.5))
```

### 6. Importing MLflow in spindle directly

```python
# WRONG: couples spindle to MLflow
import mlflow
mlflow.log_metric("extraction_time", elapsed)

# RIGHT: use the Tracker protocol
self._tracker.log_metric("extraction_time", elapsed)
```

### 7. Not setting `source_chunk_id` on triples

```python
# WRONG: breaks provenance tracking
Triple(subject="A", predicate="REL", object="B", confidence=0.9, source_chunk_id="")

# RIGHT: link back to the source chunk
Triple(subject="A", predicate="REL", object="B", confidence=0.9,
       source_chunk_id=chunk.source_id)
```

### 8. Chunker splitting mid-sentence

```python
# WRONG: chunk_boundary_coherence metric will be low
# "... the company was foun"  |  "ded in 2018 by ..."

# RIGHT: split at sentence or paragraph boundaries
# "... the company was founded in 2018."  |  "Jane Doe started ..."
```
