<p align="center">
  <img src="assets/spindle_logo.svg" alt="Spindle Logo" width="400">
</p>


# Spindle

LLM-powered, ontology-first knowledge graph extraction from unstructured text.

## What It Does

Spindle is a multi-stage pipeline for building knowledge graphs from unstructured documents. It combines document preprocessing, a Knowledge Organization System (KOS), ontology synthesis, LLM-powered triple extraction, entity resolution, and graph storage — all orchestrated through composable pipeline stages.

### Core Features

- **Document Preprocessing**: Converts documents (PDF, HTML, etc.) via [Docling](https://github.com/DS4SD/docling), chunks with [Chonkie](https://github.com/chonkie-ai/chonkie), and resolves coreferences with [fastcoref](https://github.com/shon-otmazgin/fastcoref)
- **Knowledge Organization System (KOS)**: In-process SKOS/OWL/SHACL runtime backed by [pyoxigraph](https://github.com/oxigraph/oxigraph), with Aho-Corasick NER, ANN semantic search, and SPARQL queries
- **KOS Extraction**: Cold-start (LLM-based vocab/taxonomy/thesaurus) and incremental (three-pass NER cascade: Aho-Corasick → multi-step resolution → [GLiNER2](https://github.com/fastino-ai/GLiNER2)) extraction pipelines
- **Ontology Synthesis**: Transforms KOS artifacts into formal ontologies with SHACL validation
- **Triple Extraction**: LLM-powered extraction of knowledge graph triples with source metadata, evidence spans, and timestamps via BAML prompts
- **Entity Resolution**: Semantic entity deduplication using embeddings and LLM-based matching
- **Graph Storage**: Persistent graph database using embedded [Kùzu](https://kuzudb.com/)
- **Vector Store Integration**: ChromaDB-based vector storage with multiple embedding providers (OpenAI, HuggingFace, Google)
- **Provenance Tracking**: SQLite-backed provenance store linking extracted objects to source documents and evidence spans
- **Eval Bridge**: Integration layer for [spindle-eval](https://github.com/danielkentwood/spindle-eval) with Hydra-based configuration and composable stage definitions
- **Observability**: Structured event logging across all pipeline stages with optional SQLite persistence

## Quick Start

Prerequisites: Python 3.10+, Anthropic API key, and [`uv`](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/danielkentwood/spindle.git
cd spindle
uv venv
uv pip install -e ".[dev]"
```

Create a `.env` file with your API keys:
```bash
ANTHROPIC_API_KEY=sk-ant-...
# Optional: for embeddings and other providers
OPENAI_API_KEY=sk-...
HF_API_KEY=hf_...
GEMINI_API_KEY=AIza...
```

## Usage

### Pipeline Stages

Spindle is organised as composable pipeline stages. You can use them individually or wire them together via the eval bridge.

```python
from spindle import get_pipeline_definition

defn = get_pipeline_definition(
    cfg=my_config,
    kos_dir="kos/",
    ontology=my_ontology,
)

for stage_def in defn.stages:
    output = stage_def.stage.run(...)
```

### Simple Extraction

```python
from spindle import SpindleExtractor, create_ontology

entity_types = [{"name": "Person", "description": "A human"}]
relation_types = [{"name": "works_at", "description": "Employed by", "domain": "Person", "range": "Organization"}]
ontology = create_ontology(entity_types, relation_types)

extractor = SpindleExtractor(ontology)
result = extractor.extract(
    text="Alice Johnson leads research at TechCorp.",
    source_name="Company Blog",
)

for triple in result.triples:
    print(triple.subject.name, triple.predicate, triple.object.name)
```

### Graph Storage

```python
from spindle import SpindleExtractor, GraphStore

extractor = SpindleExtractor(ontology)
store = GraphStore("path/to/graph.db")

result = extractor.extract(text="...", source_name="...")
store.add_triples(result.triples)
```

### Entity Resolution

```python
from spindle import EntityResolver, ResolutionConfig

resolver = EntityResolver(ResolutionConfig())
result = resolver.resolve_entities(store.get_all_nodes())
store.add_edges(result.same_as_edges)
```

### KOS Service

```python
from spindle.kos import KOSService

kos = KOSService(kos_dir="kos/")
hits = kos.search_ahocorasick("Alice works at TechCorp in New York")
concepts = kos.resolve("TechCorp")
```

### Preprocessing

```python
from spindle.preprocessing import SpindlePreprocessor

preprocessor = SpindlePreprocessor(cfg=preprocessing_config)
chunks = preprocessor.run(documents=["path/to/document.pdf"])
```

### Configuration

Spindle uses [Hydra](https://hydra.cc/) for configuration. YAML config groups live under `spindle/conf/`:

```
spindle/conf/
├── preprocessing/       # spindle_default.yaml, spindle_fast.yaml
├── kos_extraction/      # cold_start.yaml, incremental.yaml
├── ontology_synthesis/  # default.yaml
├── retrieval/           # local.yaml, hybrid.yaml, global.yaml
└── generation/          # default.yaml
```

Load programmatically: `from spindle.configuration import load_config_from_file`

## Architecture

```
Documents
    │
    ▼
PreprocessingStage (Docling → Chonkie → fastcoref)
    │
    ▼
KOSExtractionStage (cold-start LLM | incremental 3-pass NER)
    │
    ▼
KOSService (pyoxigraph + Aho-Corasick/ANN indices) ←→ ProvenanceStore
    │
    ▼
OntologySynthesisStage (KOS → formal ontology + SHACL)
    │
    ▼
RetrievalStage (KOS + GraphStore + ChromaDB)
    │
    ▼
GenerationStage (SpindleExtractor → triples)
    │
    ▼
EntityResolutionStage (semantic deduplication, post-batch)
```

## Project Layout

```
spindle/
├── spindle/
│   ├── preprocessing/         # Docling ingestion, chunking, coreference
│   ├── kos/                   # Knowledge Organization System (pyoxigraph)
│   ├── stages/                # Pipeline stage wrappers for spindle-eval
│   ├── extraction/            # SpindleExtractor and ontology helpers
│   ├── entity_resolution/     # Semantic entity deduplication
│   ├── graph_store/           # Kùzu-backed graph database
│   ├── vector_store/          # ChromaDB vector storage and embeddings
│   ├── provenance/            # SQLite provenance store
│   ├── observability/         # Structured event logging
│   ├── conf/                  # Hydra YAML config groups
│   ├── api/                   # FastAPI REST endpoints
│   ├── baml_src/              # BAML prompt definitions
│   ├── baml_client/           # Auto-generated BAML client
│   └── eval_bridge.py         # spindle-eval integration
├── tests/                     # Test suite
└── docs/
    ├── v1/                    # Legacy v1 documentation
    └── v2/                    # v2 design docs and notes
```

## Documentation

...coming soon!

## Testing

```bash
# Unit tests (no API required)
uv run pytest tests/ -m "not integration"

# Integration tests (requires API access)
uv run pytest tests/ -m integration

# With coverage
uv run pytest tests/ --cov=spindle --cov-report=html
```

## Optional Dependencies

```bash
# Embedding models (sentence-transformers)
uv pip install -e ".[embeddings]"

# Embedding API providers (OpenAI, HuggingFace, Google)
uv pip install -e ".[embeddings-api]"

# Eval framework integration (Hydra, OmegaConf)
uv pip install -e ".[eval]"

# All extras
uv pip install -e ".[dev,embeddings,embeddings-api,eval]"
```

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `baml-py` | LLM prompt orchestration |
| `kuzu` | Embedded graph database |
| `chromadb` | Vector storage |
| `pyoxigraph` | In-process RDF/SPARQL for KOS |
| `pyahocorasick` | Fast NER via Aho-Corasick |
| `hnswlib` | ANN vector search for KOS concepts |
| `pyshacl` | SHACL validation |
| `docling` | Document conversion (PDF, HTML, etc.) |
| `chonkie` | Recursive semantic chunking |
| `fastcoref` | Coreference resolution |
| `gliner2` | Open NER, relation extraction, classification, and structured extraction |
| `deepdiff` | Change detection for incremental ingestion |
| `hydra-core` | Config composition |
| `omegaconf` | Config composition |
| `spindle-eval` | Evaluation framework |

## Contributing & License

Contribution guidelines and license details are tracked in the `/docs` folder. PRs welcome.

**License**: MIT
