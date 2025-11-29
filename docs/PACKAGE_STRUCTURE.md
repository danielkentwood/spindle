# Package Structure

## Overview

Spindle is organized as a proper Python package for better maintainability and distribution.

## Directory Structure

```
spindle/
├── spindle/                         # Main package
│   ├── __init__.py                  # Public API exports
│   ├── extraction/                  # Extraction package
│   │   ├── __init__.py             # Public API exports
│   │   ├── extractor.py            # SpindleExtractor class
│   │   ├── recommender.py           # OntologyRecommender class
│   │   ├── process.py               # Process extraction functions
│   │   ├── utils.py                # Public utility functions
│   │   └── helpers.py              # Internal helper functions
│   ├── graph_store/                 # Graph database abstraction
│   │   ├── __init__.py              # Public API exports (GraphStore)
│   │   ├── store.py                 # GraphStore facade class
│   │   ├── base.py                  # GraphStoreBackend abstract base class
│   │   ├── backends/                 # Backend implementations
│   │   │   ├── __init__.py         # Backend exports
│   │   │   └── kuzu.py             # Kùzu backend implementation
│   │   ├── nodes.py                 # Node operation utilities
│   │   ├── edges.py                 # Edge operation utilities
│   │   ├── triples.py               # Triple integration utilities
│   │   ├── resolution.py            # Entity resolution support
│   │   ├── embeddings.py           # Graph embedding computation
│   │   └── utils.py                 # Shared utilities
│   ├── vector_store/                # Embedding + vector DB integrations
│   │   ├── __init__.py              # Public API exports
│   │   ├── base.py                  # VectorStore abstract base class
│   │   ├── chroma.py                # ChromaVectorStore implementation
│   │   ├── embeddings.py            # Embedding function factories
│   │   └── graph_embeddings.py     # GraphEmbeddingGenerator class
│   ├── ingestion/                   # Document ingestion pipeline
│   │   ├── cli.py                   # CLI interface (spindle-ingest command)
│   │   ├── service.py               # Ingestion orchestration
│   │   ├── pipeline/                # Pipeline execution
│   │   ├── loaders/                  # Document loaders
│   │   ├── splitters/                # Text splitting strategies
│   │   ├── templates/                # Ingestion templates
│   │   ├── observers/                # Observability hooks
│   │   ├── storage/                  # Storage utilities
│   │   └── graph/                   # Document graph construction
│   ├── analytics/                    # Analytics and metrics
│   ├── observability/               # Service event primitives + persistence helpers
│   ├── dashboard/                   # Streamlit analytics dashboard
│   ├── entity_resolution/           # Semantic entity deduplication
│   ├── baml_src/                    # Authoritative BAML schemas
│   │   ├── clients.baml             # LLM client configurations
│   │   ├── generators.baml          # Code generation settings
│   │   ├── entity_resolution.baml   # Entity resolution prompts
│   │   ├── process.baml             # Process extraction
│   │   └── spindle.baml             # Extraction + recommendation endpoints
│   ├── baml_client/                 # Auto-generated BAML Python client (do not edit)
│   ├── notebooks/                   # Exploratory notebooks (optional tooling)
│   ├── configuration.py             # Unified configuration system
│   ├── llm_config.py                # LLM authentication and configuration
│   └── llm_pricing.py                # LLM pricing utilities
├── demos/                           # Runnable examples (invoke with uv run)
│   ├── example.py                   # Basic extraction example
│   ├── example_entity_resolution.py # Semantic entity deduplication
│   ├── example_analytics_dashboard.py # Analytics visualization
│   ├── example_api_usage.py         # REST API usage examples
│   └── ingestion_benchmark.py        # Ingestion pipeline performance
├── tests/                           # Pytest suite
│   ├── conftest.py                  # Shared fixtures + marks
│   ├── fixtures/                    # Sample ontologies/text fixtures
│   ├── test_extractor.py            # SpindleExtractor unit tests
│   ├── test_graph_store.py          # GraphStore behaviour
│   ├── test_embeddings.py           # Vector store + embedding helpers
│   ├── test_recommender.py          # OntologyRecommender logic
│   ├── test_process_extraction.py   # Process graph extraction tests
│   ├── test_analytics.py            # Analytics and metrics tests
│   ├── test_api.py                  # REST API endpoint tests
│   ├── test_configuration.py        # Configuration system tests
│   ├── test_entity_resolution.py    # Entity resolution tests
│   ├── test_helpers.py              # Helper function tests
│   ├── test_ingestion_pipeline.py   # Ingestion pipeline tests
│   ├── test_ingestion_templates.py  # Template system tests
│   ├── test_observability_events.py # Observability event tests
│   ├── test_serialization.py        # Serialization utility tests
│   └── test_integration.py          # Real LLM integration tests (requires API keys)
├── docs/                            # Additional documentation
│   ├── QUICKSTART.md                # Getting started guide
│   ├── ENV_SETUP.md                 # Environment setup and Vertex AI
│   ├── CONFIGURATION.md              # Configuration system
│   ├── GRAPH_STORE.md               # Graph database usage
│   ├── ENTITY_RESOLUTION.md         # Entity resolution guide
│   ├── INGESTION_TEMPLATES.md       # Template system
│   ├── INGESTION_ANALYTICS.md       # Analytics schema
│   ├── OBSERVABILITY.md             # Event logging
│   ├── ONTOLOGY_RECOMMENDER.md      # Ontology recommendation
│   ├── TESTING.md                   # Testing guide
│   ├── TESTING_QUICK_REF.md         # Testing quick reference
│   └── PACKAGE_STRUCTURE.md         # This file
├── spindle_storage/                 # Default unified storage root (generated by config init)
├── htmlcov/                         # Coverage reports (generated)
├── pyproject.toml                   # Project metadata + dependencies
├── requirements.txt                 # Locked dependency snapshot (optional)
├── requirements-dev.txt             # Dev-only dependencies snapshot
├── setup.py                         # Legacy setuptools entry point
└── README.md                        # Top-level overview
```

> Storage directories are configurable. Run `uv run spindle-ingest config init`
> to generate `config.py` and point Spindle at a custom root. See
> `docs/CONFIGURATION.md` for details.

## Architecture Diagram

```mermaid
flowchart LR
    subgraph Services
        ingestion[Ingestion CLI & Pipeline]
        extractor[Extractor & Ontology Recommender]
        entity[Entity Resolution]
        vectorSvc[Vector Store API]
        obs[Observability Recorder]
        analyticsSvc[Ingestion Analytics]
    end

    subgraph Storage Solutions
        graphDB[(GraphStore<br/>Kùzu DB)]
        vectorDB[(Vector Store<br/>Chroma / Embeddings)]
        catalogDB[(Document Catalog<br/>SQLite)]
        docCache[(Document Cache Directory)]
        eventLog[(Event Log Store<br/>SQLite)]
        analyticsDB[(Analytics DB<br/>SQLite)]
    end

    ingestion --> extractor
    ingestion --> analyticsSvc
    ingestion --> obs
    ingestion --> docCache
    ingestion --> catalogDB

    extractor --> entity
    extractor --> graphDB
    extractor --> vectorSvc
    extractor --> obs

    entity --> extractor
    entity --> obs

    vectorSvc --> vectorDB
    vectorSvc --> obs

    obs --> eventLog
    obs --> analyticsDB

    analyticsSvc --> analyticsDB
```

## Module Organization

### `spindle/` Package

The main package contains all the core functionality:

#### `spindle/__init__.py`
- Exports all public API functions and classes
- Handles optional imports (e.g., GraphStore requires kuzu)
- Defines `__version__` and `__all__`

Public API:
```python
# Main classes
SpindleExtractor
OntologyRecommender
GraphStore              # Optional (requires kuzu)
VectorStore             # Optional (requires chromadb + embeddings extras)
ChromaVectorStore

# Process extraction
extract_process_graph   # Extract process DAGs from text

# Factory functions
create_ontology
create_source_metadata

# Serialization
triples_to_dict
dict_to_triples
ontology_to_dict
recommendation_to_dict
extension_to_dict

# Query/filter
get_supporting_text
filter_triples_by_source
parse_extraction_datetime
filter_triples_by_date_range

# Embedding helpers
create_openai_embedding_function
create_huggingface_embedding_function
get_default_embedding_function
```

#### `spindle/extraction/` Package
Core extraction functionality organized into modules:
- `extractor.py`: `SpindleExtractor` class for triple extraction
- `recommender.py`: `OntologyRecommender` class for ontology recommendation
- `process.py`: `extract_process_graph()` function for extracting process DAGs from text
- `utils.py`: Public utility functions (ontology creation, serialization, filtering)
- `helpers.py`: Internal helper functions (span processing, event recording, process graph utilities)
- `__init__.py`: Public API exports maintaining backward compatibility

#### `spindle/graph_store/`
Graph database abstraction (optional, requires kuzu):
- `store.py`: `GraphStore` facade class for graph database operations
- `base.py`: `GraphStoreBackend` abstract base class for backend implementations
- `backends/kuzu.py`: Kùzu backend implementation
- `nodes.py`, `edges.py`: Node and edge operation utilities
- `triples.py`: Triple integration utilities
- `resolution.py`: Entity resolution support methods
- `embeddings.py`: Graph embedding computation utilities
- `utils.py`: Shared utilities (path resolution, event recording)
- `__init__.py`: Public API exports maintaining backward compatibility
- Supports multiple backends (Kùzu now, Neo4j/others in future)
- CRUD operations for nodes and edges
- Query operations (pattern matching, source filtering, date ranges)
- Cypher query support

#### `spindle/vector_store/`
Vector embeddings and similarity search utilities (optional extras):
- `base.py`: `VectorStore` abstract base class for embedding backends
- `chroma.py`: `ChromaVectorStore` implementation for local retrieval
- `embeddings.py`: Embedding function factories (`create_openai_embedding_function`, `create_huggingface_embedding_function`, `create_gemini_embedding_function`, `get_default_embedding_function`)
- `graph_embeddings.py`: `GraphEmbeddingGenerator` class for Node2Vec-based graph embeddings
- `__init__.py`: Public API exports maintaining backward compatibility
- Optional integration with `GraphStore.compute_graph_embeddings`

#### `spindle/observability/`
Service-wide event logging support:
- `events.py`: `ServiceEvent` dataclass, `EventRecorder`, and global recorder helpers
- `storage.py`: `EventLogStore` (SQLite persistence), replay helpers, and observer attachment utilities
- `__init__.py`: Public exports for recorder, observers, and storage helpers

### `spindle/baml_src/` BAML Schemas

BAML (Basically, A Made-up Language) schema definitions:

- `clients.baml`: LLM client configurations (Claude Sonnet 4)
- `generators.baml`: Code generation settings
- `spindle.baml`: Function definitions for extraction and ontology recommendation
- `process.baml`: Process graph extraction function definitions

### `spindle/baml_client/` Generated Code

Auto-generated Python client from BAML schemas:
- **Do not edit manually** - regenerated from BAML files
- `types.py`: Type definitions (Triple, Ontology, etc.)
- Client functions for calling BAML-defined functions

### `demos/` Example Scripts

Runnable examples demonstrating various features:
- All examples import from `spindle` package
- Run from repo root with uv: `uv run python demos/example.py`
- Additional demos cover auto-ontology, scope comparison, and ontology extension flows

### `tests/` Test Suite

Comprehensive test coverage:
- Unit tests for all major functionality
- Integration tests (require API key)
- Fixtures in `conftest.py`
- Run with: `uv run pytest tests/`

## Installation

### Development Mode

Recommended for active development:

```bash
uv pip install -e ".[dev]"
```

This installs the package in "editable" mode - changes to the code are immediately available.

### With Optional Dependencies

```bash
# Add embeddings backends (local sentence-transformers)
uv pip install -e ".[dev,embeddings]"

# Add remote embedding APIs (OpenAI, Gemini, Hugging Face)
uv pip install -e ".[dev,embeddings-api]"

# Full toolbox (dev + local + remote embeddings)
uv pip install -e ".[dev,embeddings,embeddings-api]"
```

### Direct Requirements

```bash
uv pip install -r requirements.txt
```

## Imports

All public APIs are imported through the `spindle` package:

```python
# Core functionality
from spindle import SpindleExtractor, create_ontology

# Ontology recommendation
from spindle import OntologyRecommender

# Process extraction (from extraction subpackage)
from spindle.extraction import extract_process_graph

# Graph database (optional)
from spindle import GraphStore

# Utilities
from spindle import (
    triples_to_dict,
    filter_triples_by_source,
    parse_extraction_datetime
)
```

## Adding New Functionality

### Adding a New Module

1. Create the module in `spindle/` directory
2. Import and export public APIs in `spindle/__init__.py`
3. Update `__all__` list in `__init__.py`
4. Add tests in `tests/`
5. Document in relevant docs

### Adding a New Function

1. Add function to appropriate module (`extraction/` package or `graph_store.py`)
2. Export from `spindle/__init__.py` if public
3. Add to `__all__` list if public
4. Write tests
5. Document in docstring and README

## Package Distribution

The package is configured for distribution via PyPI:

- `setup.py`: Package metadata and configuration
- `MANIFEST.in`: Specifies additional files to include
- Version: Defined in `spindle/__init__.py`

To build distribution:
```bash
uv run python -m build  # install with `uv pip install build` if missing
```

## Best Practices

1. **Import from Package**: Always use `from spindle import ...`
2. **Public API**: Only import from `spindle/__init__.py` exports
3. **Private Functions**: Prefix with `_` (e.g., `_find_span_indices`)
4. **Optional Dependencies**: Handle import errors gracefully (see GraphStore)
5. **Type Hints**: Use type hints from `baml_client.types`
6. **Documentation**: Update docstrings and README for new features

## Maintenance

### Regenerating BAML Client

If you modify BAML schemas:

```bash
# Regenerate Python client
uv run baml-cli generate
```

This updates `spindle/baml_client/` directory.

### Running Tests

```bash
# All tests
uv run pytest tests/

# Specific test file
uv run pytest tests/test_extractor.py -v

# Without integration tests (no API key needed)
uv run pytest tests/ -m "not integration"

# With coverage
uv run pytest tests/ --cov=spindle --cov-report=term-missing
```

### Checking Package Structure

```bash
# Verify imports
uv run python -c "from spindle import *; print('OK')"

# List package contents
uv run python -c "import spindle; print(dir(spindle))"

# Check version
uv run python -c "import spindle; print(spindle.__version__)"
```

## Migration from Old Structure

The old structure had `spindle.py` and `graph_store.py` at the root. These have been moved to the `spindle/` package directory:

- `spindle.py` → `spindle/extraction/` package
- `graph_store.py` → `spindle/graph_store.py`

All imports remain the same thanks to the package structure!

