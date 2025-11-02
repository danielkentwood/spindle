# Package Structure

## Overview

Spindle is organized as a proper Python package for better maintainability and distribution.

## Directory Structure

```
spindle/
├── spindle/                         # Main package
│   ├── __init__.py                  # Package exports (public API)
│   ├── extractor.py                 # Core extraction functionality
│   └── graph_store.py               # Graph database persistence
├── baml_src/                        # BAML schema definitions
│   ├── clients.baml                 # LLM client configurations
│   ├── generators.baml              # Code generation config
│   └── spindle.baml                 # Extraction function definitions
├── baml_client/                     # Auto-generated BAML Python client
│   └── types.py                     # Generated type definitions
├── demos/                           # Example scripts
│   ├── example.py                   # Basic extraction example
│   ├── example_graph_store.py       # GraphStore usage example
│   └── ...                          # Other examples
├── tests/                           # Test suite
│   ├── conftest.py                  # Shared test fixtures
│   ├── test_extractor.py            # Extractor tests
│   ├── test_graph_store.py          # GraphStore tests
│   └── ...                          # Other test files
├── docs/                            # Documentation
│   ├── GRAPH_STORE.md               # GraphStore usage guide
│   ├── TESTING.md                   # Testing documentation
│   └── ...                          # Other documentation
├── setup.py                         # Package installation configuration
├── MANIFEST.in                      # Package data specification
├── requirements.txt                 # Python dependencies
├── requirements-dev.txt             # Development dependencies
└── README.md                        # Main documentation
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
GraphStore

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
```

#### `spindle/extractor.py`
Core extraction functionality:
- `SpindleExtractor`: Main extraction class
- `OntologyRecommender`: Automatic ontology recommendation
- Helper functions for ontology creation and manipulation
- Serialization utilities
- Filter and query functions

#### `spindle/graph_store.py`
Graph database persistence (optional, requires kuzu):
- `GraphStore`: Main class for graph database operations
- CRUD operations for nodes and edges
- Query operations (pattern matching, source filtering, date ranges)
- Cypher query support

### `baml_src/` BAML Schemas

BAML (Basically, A Made-up Language) schema definitions:

- `clients.baml`: LLM client configurations (Claude Sonnet 4)
- `generators.baml`: Code generation settings
- `spindle.baml`: Function definitions for extraction and ontology recommendation

### `baml_client/` Generated Code

Auto-generated Python client from BAML schemas:
- **Do not edit manually** - regenerated from BAML files
- `types.py`: Type definitions (Triple, Ontology, etc.)
- Client functions for calling BAML-defined functions

### `demos/` Example Scripts

Runnable examples demonstrating various features:
- All examples import from `spindle` package
- Run from repo root: `python demos/example.py`

### `tests/` Test Suite

Comprehensive test coverage:
- Unit tests for all major functionality
- Integration tests (require API key)
- Fixtures in `conftest.py`
- Run with: `pytest tests/`

## Installation

### Development Mode

Recommended for active development:

```bash
pip install -e .
```

This installs the package in "editable" mode - changes to the code are immediately available.

### With Optional Dependencies

```bash
# Install with GraphStore support
pip install -e ".[graph]"

# Install with development tools
pip install -e ".[dev]"

# Install everything
pip install -e ".[graph,dev]"
```

### Direct Requirements

```bash
pip install -r requirements.txt
```

## Imports

All public APIs are imported through the `spindle` package:

```python
# Core functionality
from spindle import SpindleExtractor, create_ontology

# Ontology recommendation
from spindle import OntologyRecommender

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

1. Add function to appropriate module (`extractor.py` or `graph_store.py`)
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
python setup.py sdist bdist_wheel
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
baml-cli generate
```

This updates `baml_client/` directory.

### Running Tests

```bash
# All tests
pytest tests/

# Specific test file
pytest tests/test_extractor.py -v

# Without integration tests (no API key needed)
pytest tests/ -m "not integration"

# With coverage
pytest tests/ --cov=spindle --cov-report=term-missing
```

### Checking Package Structure

```bash
# Verify imports
python -c "from spindle import *; print('OK')"

# List package contents
python -c "import spindle; print(dir(spindle))"

# Check version
python -c "import spindle; print(spindle.__version__)"
```

## Migration from Old Structure

The old structure had `spindle.py` and `graph_store.py` at the root. These have been moved to the `spindle/` package directory:

- `spindle.py` → `spindle/extractor.py`
- `graph_store.py` → `spindle/graph_store.py`

All imports remain the same thanks to the package structure!

