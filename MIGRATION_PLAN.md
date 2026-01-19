# Spindle Migration Plan

This document outlines the plan to refactor Spindle from a monolithic package (~20K LOC) into a focused core library (~12K LOC) with optional companion packages.

## Goals

1. **Slim core**: Keep only essential extraction functionality in `spindle`
2. **Modularity**: Extract separable concerns into standalone packages
3. **Flexibility**: Allow users to bring their own document loaders, vector stores, etc.
4. **Preserve KOS capabilities**: The semantic pipeline remains in core for building custom knowledge organization systems

## Target Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER APPLICATIONS                         │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                │               │               │
                ▼               ▼               ▼
        ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
        │spindle-server │ │spindle-ingest │ │spindle-vectors│
        │   (REST API)  │ │ (doc loading) │ │  (embeddings) │
        └───────┬───────┘ └───────┬───────┘ └───────┬───────┘
                │                 │                 │
                │         ┌──────┴──────┐           │
                │         │  Protocol:  │           │
                │         │DocumentChunk│           │
                │         └──────┬──────┘           │
                │                │                  │
                └────────────────┼──────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      SPINDLE (CORE)                              │
│  ┌────────────┐  ┌───────────────┐  ┌─────────────┐             │
│  │ extraction │  │entity_resolve │  │ graph_store │             │
│  └────────────┘  └───────────────┘  └─────────────┘             │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ pipeline (KOS): vocabulary, taxonomy, thesaurus, ontology  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌──────────────────────────────────────────────────┐           │
│  │ Protocols: EmbeddingProvider, SimilarityProvider │           │
│  └──────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                        ┌───────────────┐
                        │spindle-analytics│
                        │  (dashboard)  │
                        └───────────────┘
```

## Package Summary

| Package | Source Modules | Purpose | LOC |
|---------|---------------|---------|-----|
| `spindle` (core) | extraction, entity_resolution, graph_store, pipeline, baml_src | Triple extraction, KOS pipeline, and KG storage | ~12K |
| `spindle-analytics` | analytics, dashboard | Metrics visualization | ~1.9K |
| `spindle-server` | api | REST API for spindle services | ~3.2K |
| `spindle-ingest` | ingestion | Document loading pipeline | ~2.2K |
| `spindle-vectors` | vector_store | Embedding abstractions | ~1.2K |

---

## Phase 1: Extract `spindle-analytics`

**Modules**: `analytics/` + `dashboard/`
**New Repository**: `spindle-analytics`
**Estimated Effort**: Low (loosest coupling)

### 1.1 Create Repository Structure

```
spindle-analytics/
├── spindle_analytics/
│   ├── __init__.py
│   ├── metrics/              # from analytics/
│   │   ├── __init__.py
│   │   ├── schema.py
│   │   ├── store.py
│   │   └── views.py
│   └── dashboard/            # from dashboard/
│       ├── __init__.py
│       └── app.py
├── pyproject.toml
├── README.md
└── tests/
```

### 1.2 Refactoring Tasks

1. **Copy modules**: `spindle/analytics/` → `spindle_analytics/metrics/`
2. **Copy modules**: `spindle/dashboard/` → `spindle_analytics/dashboard/`
3. **Remove dependency on `ingestion.types`**:
   - Define local event types or accept generic dicts
   - Current dependency: `from spindle.ingestion.types import ...`
4. **Remove dependency on `observability`**:
   - Accept event data directly rather than subscribing to EventRecorder
   - Will be addressed fully in Phase 6
5. **Update imports**: Change all `from spindle.` to `from spindle_analytics.`

### 1.3 Interface with Core Spindle

```python
# spindle-analytics accepts event dicts, not spindle-specific types
class AnalyticsStore:
    def record_extraction_event(self, event: dict) -> None: ...
    def record_ingestion_event(self, event: dict) -> None: ...
    def record_resolution_event(self, event: dict) -> None: ...
```

### 1.4 Update Core Spindle

1. Remove `analytics/` and `dashboard/` directories
2. Remove analytics-related dependencies from `pyproject.toml`
3. Add `spindle-analytics` as optional dependency: `analytics = ["spindle-analytics"]`

### 1.5 README Content for spindle-analytics

```markdown
# spindle-analytics

Metrics collection and visualization dashboard for Spindle knowledge graph extraction.

## Current State

This package was extracted from the main Spindle repository. It contains:
- `metrics/`: Analytics schema, storage, and views for extraction/ingestion events
- `dashboard/`: Streamlit-based visualization dashboard

## Next Steps

1. [ ] Define stable event schema (currently depends on internal Spindle types)
2. [ ] Add pyproject.toml with proper dependencies (streamlit, pandas, sqlalchemy)
3. [ ] Create CLI entry point for dashboard
4. [ ] Add integration hooks for core Spindle to push events
5. [ ] Write tests for metric calculations
6. [ ] Document dashboard configuration options

## Integration with Spindle

Once complete, install alongside spindle:
\`\`\`bash
pip install spindle[analytics]  # or: pip install spindle spindle-analytics
\`\`\`
```

---

## Phase 2: Extract `spindle-server`

**Modules**: `api/`
**New Repository**: `spindle-server`
**Estimated Effort**: Medium (depends on core spindle)

### 2.1 Create Repository Structure

```
spindle-server/
├── spindle_server/
│   ├── __init__.py
│   ├── app.py                # FastAPI app factory
│   ├── config.py             # Server configuration
│   ├── dependencies.py       # Dependency injection
│   ├── models.py             # Pydantic request/response models
│   └── routers/
│       ├── __init__.py
│       ├── extraction.py
│       ├── ontology.py
│       ├── resolution.py
│       ├── process.py
│       ├── corpus.py
│       ├── ingestion.py      # Optional: requires spindle-ingest
│       └── pipeline.py       # Uses core spindle pipeline module
├── pyproject.toml
├── README.md
└── tests/
```

### 2.2 Refactoring Tasks

1. **Copy module**: `spindle/api/` → `spindle_server/`
2. **Update imports**:
   - Core spindle: `from spindle import SpindleExtractor, GraphStore, EntityResolver`
   - Keep these as required dependencies
3. **Make optional routers conditional**:
   ```python
   # spindle_server/app.py
   def create_app(
       enable_ingestion: bool = False,  # requires spindle-ingest
       enable_pipeline: bool = True,    # uses core spindle pipeline
       enable_analytics: bool = False,  # requires spindle-analytics
   ) -> FastAPI:
       app = FastAPI(title="Spindle API")

       # Core routers (always available)
       app.include_router(extraction_router)
       app.include_router(ontology_router)
       app.include_router(resolution_router)
       app.include_router(process_router)

       # Optional routers
       if enable_ingestion:
           from spindle_server.routers.ingestion import router
           app.include_router(router)
       ...
   ```
4. **Extract session management**: Move to dependency injection pattern

### 2.3 Update Core Spindle

1. Remove `api/` directory
2. Remove FastAPI dependencies from core `pyproject.toml`
3. Add `spindle-server` as optional dependency: `server = ["spindle-server"]`

### 2.4 README Content for spindle-server

```markdown
# spindle-server

REST API server for Spindle knowledge graph extraction services.

## Current State

This package was extracted from the main Spindle repository. It provides:
- FastAPI-based REST API
- Routers for: extraction, ontology, entity resolution, process extraction, pipeline
- Optional routers for: ingestion (requires spindle-ingest)

## Next Steps

1. [ ] Add pyproject.toml with dependencies (fastapi, uvicorn, python-multipart)
2. [ ] Make spindle-ingest router truly optional (lazy import)
3. [ ] Add OpenAPI documentation
4. [ ] Add authentication/authorization hooks
5. [ ] Add rate limiting and request validation
6. [ ] Create Docker deployment configuration
7. [ ] Write API integration tests

## Running the Server

\`\`\`bash
uvicorn spindle_server:create_app --factory --host 0.0.0.0 --port 8000
\`\`\`
```

---

## Phase 3: Extract `spindle-ingest`

**Modules**: `ingestion/`
**New Repository**: `spindle-ingest`
**Estimated Effort**: Medium-High (need to define document protocol)

### 3.1 Define Document Protocol in Core Spindle

Before extracting, define a protocol that allows any document loader to work with Spindle:

```python
# spindle/protocols.py (NEW FILE in core)
from typing import Protocol, Iterator, Any
from dataclasses import dataclass

@dataclass
class DocumentChunk:
    """Universal document chunk that any loader can produce."""
    content: str
    metadata: dict[str, Any]
    source_name: str
    chunk_index: int = 0
    total_chunks: int = 1

class DocumentLoader(Protocol):
    """Protocol for document loaders. Implement to integrate with Spindle."""

    def load(self, path: str) -> Iterator[DocumentChunk]:
        """Load document(s) from path and yield chunks."""
        ...

class DocumentSplitter(Protocol):
    """Protocol for text splitters."""

    def split(self, text: str, metadata: dict[str, Any]) -> list[DocumentChunk]:
        """Split text into chunks."""
        ...
```

### 3.2 Create Repository Structure

```
spindle-ingest/
├── spindle_ingest/
│   ├── __init__.py
│   ├── cli.py                # CLI entry point
│   ├── service.py            # Ingestion orchestration
│   ├── config.py             # Ingestion configuration
│   ├── loaders/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── pdf.py
│   │   ├── html.py
│   │   ├── markdown.py
│   │   └── csv.py
│   ├── splitters/
│   │   ├── __init__.py
│   │   ├── recursive.py
│   │   ├── semantic.py
│   │   └── token.py
│   ├── templates/
│   │   ├── __init__.py
│   │   ├── registry.py
│   │   └── defaults.py
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── catalog.py
│   │   └── corpus.py
│   └── adapters/             # NEW: adapters for other loaders
│       ├── __init__.py
│       ├── langchain.py      # Wrap LangChain loaders
│       └── docling.py        # Wrap Docling loaders
├── pyproject.toml
├── README.md
└── tests/
```

### 3.3 Refactoring Tasks

1. **Copy module**: `spindle/ingestion/` → `spindle_ingest/`
2. **Implement DocumentChunk output**: All loaders yield `DocumentChunk` objects
3. **Create adapters for external loaders**:
   ```python
   # spindle_ingest/adapters/langchain.py
   from langchain.document_loaders import BaseLoader
   from spindle.protocols import DocumentChunk, DocumentLoader

   class LangChainAdapter(DocumentLoader):
       def __init__(self, loader: BaseLoader):
           self.loader = loader

       def load(self, path: str) -> Iterator[DocumentChunk]:
           for i, doc in enumerate(self.loader.load()):
               yield DocumentChunk(
                   content=doc.page_content,
                   metadata=doc.metadata,
                   source_name=path,
                   chunk_index=i,
               )
   ```
4. **Create Docling adapter**:
   ```python
   # spindle_ingest/adapters/docling.py
   from docling.document_converter import DocumentConverter
   from spindle.protocols import DocumentChunk, DocumentLoader

   class DoclingAdapter(DocumentLoader):
       def __init__(self, converter: DocumentConverter | None = None):
           self.converter = converter or DocumentConverter()

       def load(self, path: str) -> Iterator[DocumentChunk]:
           result = self.converter.convert(path)
           # Convert Docling output to DocumentChunks
           ...
   ```
5. **Update CLI**: Keep `spindle-ingest` command but use new structure

### 3.4 Update Core Spindle

1. Add `spindle/protocols.py` with `DocumentChunk`, `DocumentLoader`, `DocumentSplitter`
2. Update `SpindleExtractor` to accept `DocumentChunk` directly:
   ```python
   def extract_from_chunks(
       self,
       chunks: Iterable[DocumentChunk],
       existing_triples: list[Triple] | None = None,
   ) -> Iterator[ExtractionResult]:
       for chunk in chunks:
           yield self.extract(
               text=chunk.content,
               source_name=chunk.source_name,
               existing_triples=existing_triples,
           )
   ```
3. Remove `ingestion/` directory from core
4. Add optional dependency: `ingest = ["spindle-ingest"]`

### 3.5 README Content for spindle-ingest

```markdown
# spindle-ingest

Document ingestion pipeline for Spindle knowledge graph extraction.

## Current State

This package was extracted from the main Spindle repository. It provides:
- CLI for batch document ingestion
- Document loaders for PDF, HTML, Markdown, CSV
- Text splitters (recursive, semantic, token-based)
- Template system for configurable ingestion pipelines
- Adapters for LangChain and Docling document loaders

## Next Steps

1. [ ] Add pyproject.toml with dependencies (langchain, pypdf, etc.)
2. [ ] Complete LangChain adapter for all loader types
3. [ ] Complete Docling adapter
4. [ ] Test adapters with real documents
5. [ ] Add streaming support for large document sets
6. [ ] Document template configuration format
7. [ ] Add progress reporting hooks

## Usage

### With built-in loaders
\`\`\`python
from spindle_ingest import PDFLoader, RecursiveSplitter

loader = PDFLoader()
splitter = RecursiveSplitter(chunk_size=1000)

for chunk in loader.load("document.pdf"):
    chunks = splitter.split(chunk.content, chunk.metadata)
\`\`\`

### With LangChain
\`\`\`python
from langchain.document_loaders import PyPDFLoader
from spindle_ingest.adapters import LangChainAdapter

loader = LangChainAdapter(PyPDFLoader("document.pdf"))
for chunk in loader.load("document.pdf"):
    # chunk is a DocumentChunk compatible with Spindle
    ...
\`\`\`

### With Docling
\`\`\`python
from spindle_ingest.adapters import DoclingAdapter

loader = DoclingAdapter()
for chunk in loader.load("document.pdf"):
    # chunk is a DocumentChunk compatible with Spindle
    ...
\`\`\`
```

---

## Phase 4: Extract `spindle-vectors`

**Modules**: `vector_store/`
**New Repository**: `spindle-vectors`
**Estimated Effort**: Low-Medium (need thin interface in core)

### 4.1 Define Embedding Protocol in Core Spindle

```python
# spindle/protocols.py (ADD to existing file)
from typing import Protocol, Sequence

class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embeddings for texts."""
        ...

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        ...

class SimilarityProvider(Protocol):
    """Protocol for similarity search."""

    def add(self, ids: list[str], texts: list[str], metadata: list[dict]) -> None:
        """Add items to the index."""
        ...

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Search for similar items, return (id, score) pairs."""
        ...
```

### 4.2 Create Repository Structure

```
spindle-vectors/
├── spindle_vectors/
│   ├── __init__.py
│   ├── chroma.py             # ChromaDB implementation
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── openai.py
│   │   ├── huggingface.py
│   │   ├── google.py
│   │   └── local.py          # sentence-transformers
│   └── graph_embeddings.py   # Node2Vec
├── pyproject.toml
├── README.md
└── tests/
```

### 4.3 Refactoring Tasks

1. **Copy module**: `spindle/vector_store/` → `spindle_vectors/`
2. **Implement protocols**: Ensure all classes implement `EmbeddingProvider` and/or `SimilarityProvider`
3. **Update entity_resolution**: Use protocol instead of concrete class
   ```python
   # spindle/entity_resolution/blocking.py
   from spindle.protocols import SimilarityProvider

   class SemanticBlocker:
       def __init__(self, similarity_provider: SimilarityProvider):
           self.provider = similarity_provider
   ```

### 4.4 Update Core Spindle

1. Add protocols to `spindle/protocols.py`
2. Update `entity_resolution` to use `SimilarityProvider` protocol
3. Remove `vector_store/` directory
4. Add optional dependency: `vectors = ["spindle-vectors"]`
5. Provide simple in-memory fallback for basic usage:
   ```python
   # spindle/similarity.py (NEW - simple fallback)
   from sklearn.metrics.pairwise import cosine_similarity

   class SimpleSimilarityProvider:
       """Basic in-memory similarity using TF-IDF. No external dependencies."""
       ...
   ```

### 4.5 README Content for spindle-vectors

```markdown
# spindle-vectors

Vector embeddings and similarity search for Spindle knowledge graph extraction.

## Current State

This package was extracted from the main Spindle repository. It provides:
- ChromaDB-based vector storage
- Embedding providers: OpenAI, HuggingFace, Google, local (sentence-transformers)
- Graph embeddings via Node2Vec

## Next Steps

1. [ ] Add pyproject.toml with optional dependencies for each provider
2. [ ] Implement EmbeddingProvider and SimilarityProvider protocols from spindle
3. [ ] Add FAISS backend option
4. [ ] Add Pinecone backend option
5. [ ] Document embedding provider configuration
6. [ ] Add benchmarks for different backends

## Usage

\`\`\`python
from spindle_vectors import ChromaVectorStore
from spindle_vectors.embeddings import OpenAIEmbeddings

# Create embedding provider
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Create vector store
store = ChromaVectorStore(
    collection_name="my_entities",
    embedding_function=embeddings,
)

# Use with Spindle entity resolution
from spindle import EntityResolver, ResolutionConfig

resolver = EntityResolver(
    config=ResolutionConfig(),
    similarity_provider=store,  # Implements SimilarityProvider protocol
)
\`\`\`
```

---

## Phase 5: Remove Observability Module

**Modules**: `observability/`
**Action**: Delete and use Langfuse directly
**Estimated Effort**: Low

### 5.1 Current State Analysis

The `observability/` module provides:
1. `ServiceEvent` dataclass for structured events
2. `EventRecorder` for dispatching events to observers
3. `EventLogStore` for SQLite persistence
4. Integration with Langfuse for LLM tracing

**What to keep**: The BAML Collector integration that extracts LLM metrics and sends to Langfuse.

### 5.2 Refactoring Tasks

1. **Keep Langfuse integration in extraction**:
   ```python
   # spindle/extraction/extractor.py
   from langfuse import Langfuse
   from baml_client.collector import Collector

   class SpindleExtractor:
       def __init__(
           self,
           langfuse_client: Langfuse | None = None,
           enable_tracing: bool = True,
       ):
           self.langfuse = langfuse_client
           self.enable_tracing = enable_tracing

       def extract(self, text: str, source_name: str, ...) -> ExtractionResult:
           collector = Collector() if self.enable_tracing else None

           # Run extraction with collector
           result = self._extract_with_collector(text, collector, ...)

           # Send metrics to Langfuse
           if self.langfuse and collector:
               self._record_to_langfuse(collector, source_name)

           return result

       def _record_to_langfuse(self, collector: Collector, source_name: str):
           """Extract metrics from BAML collector and send to Langfuse."""
           trace = self.langfuse.trace(name="spindle_extraction")
           trace.generation(
               name="extract_triples",
               model=collector.model,
               input={"source": source_name},
               output={"triple_count": len(result.triples)},
               usage={
                   "input_tokens": collector.input_tokens,
                   "output_tokens": collector.output_tokens,
               },
           )
   ```

2. **Remove observability module**:
   - Delete `spindle/observability/` directory
   - Remove `observability` imports from all modules
   - Remove `EventRecorder` usage (or replace with simple logging)

3. **Update modules that used observability**:
   - `extraction/`: Use Langfuse directly (as above)
   - `entity_resolution/`: Use Langfuse for matching traces
   - `graph_store/`: Use standard Python logging
   - `ingestion/`: Use standard Python logging or callbacks

4. **Simplify configuration**:
   ```python
   # spindle/configuration.py
   @dataclass
   class LangfuseSettings:
       """Langfuse configuration for LLM tracing."""
       enabled: bool = True
       public_key: str | None = None
       secret_key: str | None = None
       host: str = "https://cloud.langfuse.com"
   ```

### 5.3 Update Core Spindle

1. Delete `observability/` directory
2. Remove `EventRecorder`, `ServiceEvent`, `EventLogStore` from exports
3. Add `langfuse` as optional dependency
4. Update `configuration.py` to use `LangfuseSettings`
5. Document Langfuse setup in README

### 5.4 Migration Notes

```python
# BEFORE (with observability module)
from spindle.observability import get_event_recorder

recorder = get_event_recorder("extraction")
recorder.record("extraction.complete", {"triples": len(triples)})

# AFTER (direct Langfuse)
from langfuse import Langfuse

langfuse = Langfuse()
trace = langfuse.trace(name="extraction")
trace.event(name="extraction.complete", metadata={"triples": len(triples)})
```

---

## Implementation Order

```
Phase 1 ──────────────────────────────────────────────────────────►
         spindle-analytics (dashboard + analytics)

Phase 2 ──────────────────────────────────────────────────────────►
         spindle-server (api)

Phase 3 ──────────────────────────────────────────────────────────►
         spindle-ingest (ingestion + adapters)
         Define DocumentChunk protocol in core

Phase 4 ──────────────────────────────────────────────────────────►
         spindle-vectors (vector_store)
         Define EmbeddingProvider/SimilarityProvider protocols

Phase 5 ──────────────────────────────────────────────────────────►
         Remove observability, use Langfuse directly
```

**Parallel work possible**:
- Phase 1 and Phase 2 can run in parallel
- Phase 3 and Phase 4 can run in parallel (after protocols defined)
- Phase 5 can run anytime but best done last

---

## Post-Migration: Core Spindle Structure

```
spindle/
├── spindle/
│   ├── __init__.py           # Public API exports
│   ├── protocols.py          # DocumentChunk, EmbeddingProvider, SimilarityProvider
│   ├── configuration.py      # Simplified config (no ingestion, etc.)
│   ├── llm_config.py         # LLM authentication
│   ├── similarity.py         # Simple fallback SimilarityProvider
│   ├── extraction/
│   │   ├── __init__.py
│   │   ├── extractor.py      # SpindleExtractor (with Langfuse integration)
│   │   ├── recommender.py    # OntologyRecommender
│   │   ├── process.py        # Process graph extraction
│   │   ├── utils.py
│   │   └── helpers.py
│   ├── entity_resolution/
│   │   ├── __init__.py
│   │   ├── resolver.py       # EntityResolver
│   │   ├── blocking.py       # SemanticBlocker (uses SimilarityProvider protocol)
│   │   ├── matching.py       # SemanticMatcher
│   │   ├── merging.py
│   │   ├── config.py
│   │   └── models.py
│   ├── graph_store/
│   │   ├── __init__.py
│   │   ├── store.py          # GraphStore
│   │   ├── base.py
│   │   ├── backends/
│   │   │   └── kuzu.py
│   │   ├── nodes.py
│   │   ├── edges.py
│   │   └── triples.py
│   ├── pipeline/             # KOS semantic pipeline (kept in core)
│   │   ├── __init__.py
│   │   ├── orchestrator.py   # Main pipeline orchestrator
│   │   ├── config.py         # Pipeline configuration
│   │   ├── stages/           # Vocabulary, taxonomy, thesaurus, ontology stages
│   │   ├── models/           # VocabularyTerm, TaxonomyRelation, etc.
│   │   └── storage/          # KOS artifact persistence
│   ├── baml_src/             # Core extraction + pipeline BAML
│   │   ├── spindle.baml
│   │   ├── process.baml
│   │   ├── entity_resolution.baml
│   │   ├── pipeline.baml     # KOS pipeline functions
│   │   ├── clients.baml
│   │   └── generators.baml
│   └── baml_client/          # Auto-generated
├── tests/
├── pyproject.toml
└── README.md
```

**Estimated core LOC**: ~12,000 (down from ~20,000)

---

## Repository Checklist

Create these repositories:

- [ ] `spindle-analytics` - Metrics and dashboard
- [ ] `spindle-server` - REST API
- [ ] `spindle-ingest` - Document ingestion pipeline
- [ ] `spindle-vectors` - Vector embeddings and similarity

Each repository needs:
- [ ] Copy relevant source code
- [ ] Update imports (change `from spindle.` to local imports)
- [ ] Create `pyproject.toml` with dependencies
- [ ] Create `README.md` with current state and next steps
- [ ] Create minimal `tests/` directory structure
