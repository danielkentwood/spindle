<p align="center">
  <img src="assets/spindle_logo.svg" alt="Spindle Logo" width="400">
</p>


# Spindle

LLM-powered ontology-first extraction of knowledge graph triples from text.

## What It Does

Spindle is a comprehensive toolkit for building knowledge graphs from unstructured text. It combines LLM-powered extraction with graph storage, entity resolution, and observability features.

### Core Features

- **Ontology Recommendation**: Automatically recommends ontologies (entities, relations, attributes) from raw text using BAML-driven LLM prompts
- **Triple Extraction**: Extracts knowledge graph triples with source metadata, evidence spans, and timestamps
- **Entity Resolution**: Semantic entity deduplication using embeddings and LLM-based matching to keep entities consistent across documents
- **Graph Storage**: Persistent graph database using embedded Kùzu with support for nodes, edges, and embeddings
- **Vector Store Integration**: ChromaDB-based vector storage with support for multiple embedding providers (OpenAI, HuggingFace, Google)
- **Document Ingestion Pipeline**: CLI-driven ingestion system with template-based document processing, chunking, and graph construction
- **Analytics Dashboard**: Streamlit-based dashboard for visualizing ingestion metrics, extraction statistics, and entity resolution results
- **Observability**: Structured event logging across ingestion, extraction, and storage with optional SQLite persistence
- **Configuration Management**: Unified configuration system for storage paths, graph databases, vector stores, and observability settings

## Quick Start

Prerequisites: Python 3.9+, Anthropic API key, and [`uv`](https://github.com/astral-sh/uv) (preferred installer).

```bash
git clone <repository-url>
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

See `docs/QUICKSTART.md` for detailed setup instructions and `docs/ENV_SETUP.md` if you hit environment issues.

## Basic Usage

### Simple Extraction

```python
from spindle import SpindleExtractor

extractor = SpindleExtractor()  # auto-recommends ontology on first extract
result = extractor.extract(
    text="Alice Johnson leads research at TechCorp.",
    source_name="Company Blog"
)

for triple in result.triples:
    print(triple.subject.name, triple.predicate, triple.object.name)
```

### With Graph Storage

```python
from spindle import SpindleExtractor, GraphStore

extractor = SpindleExtractor()
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

### Document Ingestion Pipeline

```bash
# Initialize configuration
uv run spindle-ingest config init

# Ingest documents
uv run spindle-ingest ingest --input documents/ --template default

# View analytics dashboard
uv run spindle-dashboard --database sqlite:///spindle_storage/analytics.db
```

### Key Features

- **Ontology Scope Control**: Use `ontology_scope="minimal" | "balanced" | "comprehensive"` to control extraction granularity
- **Ontology Recommender**: Use `OntologyRecommender` for explicit recommendations and conservative ontology extension
- **Graph Persistence**: Store and query triples with `GraphStore()` backed by Kùzu
- **Vector Search**: Use `ChromaVectorStore` for semantic search over extracted content
- **Template-Based Ingestion**: Define custom document processing templates (see `docs/INGESTION_TEMPLATES.md`)

### Configuration

- Scaffold a runtime config with `uv run spindle-ingest config init`
- Edit the generated `config.py` to customize storage paths, graph DB, vector store, and observability settings
- Pass config to tooling via `--config /path/to/config.py`
- Load programmatically: `from spindle.configuration import load_config_from_file`
- See `docs/CONFIGURATION.md` for the full schema and usage patterns

## Project Layout

```
spindle/
├── spindle/                    # Main package
│   ├── extraction/             # Core extraction and ontology recommendation
│   ├── graph_store/            # Kùzu-backed graph database
│   ├── entity_resolution/      # Semantic entity deduplication
│   ├── vector_store/           # Vector storage and embeddings
│   ├── ingestion/              # Document ingestion pipeline
│   │   ├── loaders/            # Document loaders
│   │   ├── splitters/          # Text splitting strategies
│   │   ├── templates/          # Ingestion templates
│   │   └── observers/          # Observability hooks
│   ├── analytics/              # Analytics and metrics
│   ├── observability/          # Event logging and persistence
│   ├── dashboard/              # Streamlit analytics dashboard
│   ├── baml_client/            # BAML runtime client
│   ├── baml_src/               # BAML function definitions
│   ├── notebooks/              # Jupyter notebook examples 
├── tests/                      # Test suite       
```

## Documentation

- **[Quick Start Guide](docs/QUICKSTART.md)**: Get up and running in minutes
- **[Configuration](docs/CONFIGURATION.md)**: Unified configuration system
- **[Graph Store](docs/GRAPH_STORE.md)**: Graph database operations and queries
- **[Entity Resolution](docs/ENTITY_RESOLUTION.md)**: Semantic entity deduplication
- **[Ingestion Templates](docs/INGESTION_TEMPLATES.md)**: Custom document processing
- **[Ontology Recommender](docs/ONTOLOGY_RECOMMENDER.md)**: Automatic ontology generation
- **[Observability](docs/OBSERVABILITY.md)**: Event logging and monitoring
- **[Analytics](docs/INGESTION_ANALYTICS.md)**: Metrics and dashboard usage

## Testing

```bash
# Unit tests (no API required)
uv run pytest tests/ -m "not integration"

# Integration tests (requires API access)
uv run pytest tests/ -m integration

# With coverage
uv run pytest tests/ --cov=spindle --cov-report=html
```

Coverage reports and further details: `docs/TESTING.md` and `docs/TESTING_QUICK_REF.md`.

## Optional Dependencies

Install additional features with extras:

```bash
# Embedding models (sentence-transformers)
uv pip install -e ".[embeddings]"

# Embedding API providers (OpenAI, HuggingFace, Google)
uv pip install -e ".[embeddings-api]"

# All extras
uv pip install -e ".[dev,embeddings,embeddings-api]"
```

## Contributing & License

Contribution guidelines and license details are tracked in the `/docs` folder. PRs welcome.

**License**: MIT

