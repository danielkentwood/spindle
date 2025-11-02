# Migration Guide: Package Restructuring

## Overview

Spindle has been restructured into a proper Python package. The functionality remains the same, but the internal structure has been reorganized for better maintainability and distribution.

## What Changed

### Old Structure
```
spindle/
├── spindle.py           # All extraction code
├── graph_store.py       # GraphStore code
├── example.py           # Example in root
└── tests/               # Tests
```

### New Structure
```
spindle/
├── spindle/             # Package directory
│   ├── __init__.py      # Package exports
│   ├── extractor.py     # Extraction functionality (was spindle.py)
│   └── graph_store.py   # GraphStore functionality
├── demos/               # Example scripts
│   └── example.py       # Moved here
├── tests/               # Tests (updated imports)
└── setup.py             # New: pip installation support
```

## For Users

### No Code Changes Required!

All imports remain the same:

```python
# These still work exactly as before
from spindle import SpindleExtractor, create_ontology
from spindle import OntologyRecommender
from spindle import GraphStore
```

### Installation

You can now install Spindle as a package:

```bash
# Development mode (recommended)
pip install -e .

# With all dependencies including GraphStore
pip install -e ".[graph]"

# Or just requirements
pip install -r requirements.txt
```

## For Developers

### Module Structure

- `spindle/extractor.py`: Core extraction functionality
  - `SpindleExtractor`
  - `OntologyRecommender`
  - Helper functions
  
- `spindle/graph_store.py`: Graph database persistence
  - `GraphStore`
  
- `spindle/__init__.py`: Package exports (imports everything from submodules)

### Tests

All tests have been updated to use `from spindle import ...`. No changes needed if you're using the public API.

### Old Files

The old `spindle.py` and `graph_store.py` files at the root are kept temporarily for backward compatibility. They can be safely removed once you've verified everything works with the new structure.

To remove them:
```bash
rm spindle.py graph_store.py
```

## Benefits of New Structure

1. **Proper Package**: Can be installed via `pip install -e .`
2. **Better Organization**: Clear separation between modules
3. **Distribution Ready**: Can be published to PyPI
4. **Import Consistency**: All imports go through `spindle` package
5. **Maintainability**: Easier to find and modify specific functionality

## Testing the Migration

Run the test suite to verify everything works:

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_helpers.py -v

# Run without integration tests
pytest tests/ -m "not integration"
```

Run a demo script:

```bash
python demos/example.py
```

Test imports:

```python
from spindle import (
    SpindleExtractor,
    OntologyRecommender,
    GraphStore,
    create_ontology
)
print("All imports successful!")
```

## Troubleshooting

### Import Errors

If you see import errors, ensure you're in the repo root and the package is properly structured:

```bash
cd /path/to/spindle
ls spindle/  # Should show __init__.py, extractor.py, graph_store.py
python -c "from spindle import SpindleExtractor; print('OK')"
```

### Old Imports

If you have code that directly imports from the old files:

```python
# Old (will break when old files are removed)
from spindle import SpindleExtractor  # This worked by accident

# New (correct - works with package structure)
from spindle import SpindleExtractor  # This is the intended way
```

The imports are actually the same! The package structure just makes them official.

## Questions?

If you encounter any issues with the migration, please:
1. Check that you're using imports from `spindle` package
2. Verify the package structure is intact
3. Run tests to identify specific issues
4. Check this guide for common problems

The migration should be seamless for all existing code that imports from `spindle`.

