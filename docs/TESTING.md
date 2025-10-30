# Testing Guide for Spindle

This document describes the testing strategy and how to run tests for Spindle.

## Test Structure

```
tests/
├── __init__.py
├── conftest.py                  # Pytest configuration and shared fixtures
├── fixtures/
│   ├── __init__.py
│   ├── sample_texts.py          # Sample text data for tests
│   └── sample_ontologies.py     # Sample ontology definitions
├── test_helpers.py              # Tests for helper functions
├── test_serialization.py        # Tests for serialization functions
├── test_extractor.py            # Tests for SpindleExtractor
├── test_recommender.py          # Tests for OntologyRecommender
└── test_integration.py          # Integration tests (require API key)
```

## Test Types

### Unit Tests (Fast, No API Calls)

Unit tests mock all LLM calls using `pytest-mock` and `unittest.mock`. They test:

- **Helper Functions** (`test_helpers.py`)
  - `_find_span_indices()` - Character span matching with various edge cases
  - `create_ontology()` - Ontology creation from dictionaries
  - `filter_triples_by_source()` - Source-based filtering
  - `parse_extraction_datetime()` - DateTime parsing
  - `filter_triples_by_date_range()` - Date range filtering

- **Serialization** (`test_serialization.py`)
  - `triples_to_dict()` / `dict_to_triples()` - Triple serialization
  - `ontology_to_dict()` - Ontology serialization
  - `recommendation_to_dict()` - Recommendation serialization
  - `extension_to_dict()` - Extension serialization
  - Round-trip serialization tests

- **SpindleExtractor** (`test_extractor.py`)
  - Initialization with/without ontology
  - Extraction with mocked BAML calls
  - Post-processing logic (datetime setting, span indices)
  - Auto-ontology recommendation flow

- **OntologyRecommender** (`test_recommender.py`)
  - Recommendation with different scopes
  - Extension analysis
  - Ontology extension logic
  - Combined operations

### Integration Tests (Slow, Require API Key)

Integration tests make actual LLM calls and are marked with `@pytest.mark.integration`:

- End-to-end extraction with real LLM
- Real ontology recommendation
- Multi-source extraction workflows
- Entity consistency across extractions

## Running Tests

### Setup

1. Ensure you're in the `kgx` conda environment:
```bash
conda activate kgx
```

2. Install testing dependencies:
```bash
pip install -r requirements-dev.txt
```

### Run All Unit Tests (Fast)

```bash
# Run all unit tests (excludes integration tests)
pytest tests/ -m "not integration"

# Or simply (integration tests are skipped by default if no API key)
pytest tests/
```

### Run Specific Test Files

```bash
# Test helper functions only
pytest tests/test_helpers.py -v

# Test serialization only
pytest tests/test_serialization.py -v

# Test extractor only
pytest tests/test_extractor.py -v

# Test recommender only
pytest tests/test_recommender.py -v
```

### Run with Coverage

```bash
# Generate coverage report
pytest tests/ -m "not integration" --cov=spindle --cov-report=term-missing

# Generate HTML coverage report
pytest tests/ -m "not integration" --cov=spindle --cov-report=html

# View HTML report (opens in browser)
open htmlcov/index.html
```

### Run Integration Tests

Integration tests require an `ANTHROPIC_API_KEY` to be set:

```bash
# Set API key (if not in .env)
export ANTHROPIC_API_KEY=your_key_here

# Run integration tests only
pytest tests/ -m integration -v

# Run all tests including integration tests
pytest tests/ -v
```

**Note:** Integration tests make actual API calls and will incur costs. Use sparingly during development.

### Run Specific Tests

```bash
# Run a specific test class
pytest tests/test_helpers.py::TestFindSpanIndices -v

# Run a specific test method
pytest tests/test_helpers.py::TestFindSpanIndices::test_exact_match -v
```

## Test Execution Options

### Verbose Output

```bash
pytest tests/ -v          # Verbose (show test names)
pytest tests/ -vv         # More verbose (show full diffs)
```

### Stop on First Failure

```bash
pytest tests/ -x          # Stop after first failure
pytest tests/ --maxfail=3 # Stop after 3 failures
```

### Run Failed Tests Only

```bash
pytest tests/ --lf        # Run last failed tests
pytest tests/ --ff        # Run failed tests first, then others
```

### Parallel Execution (Optional)

Install `pytest-xdist` for parallel test execution:

```bash
pip install pytest-xdist

# Run tests in parallel (4 workers)
pytest tests/ -n 4
```

## Writing New Tests

### Unit Test Template

```python
"""Tests for my_module."""

import pytest
from unittest.mock import patch, MagicMock

from spindle import MyClass

class TestMyClass:
    """Tests for MyClass."""
    
    def test_my_method(self, simple_ontology):
        """Test description."""
        # Arrange
        obj = MyClass(simple_ontology)
        
        # Act
        result = obj.my_method()
        
        # Assert
        assert result is not None
```

### Integration Test Template

```python
"""Integration tests for my_module."""

import pytest
from spindle import MyClass

@pytest.mark.integration
class TestMyClassIntegration:
    """Integration tests for MyClass."""
    
    def test_real_llm_call(self, skip_if_no_api_key):
        """Test with real LLM call."""
        obj = MyClass()
        result = obj.call_llm()
        
        assert result is not None
```

### Using Fixtures

Fixtures are defined in `conftest.py` and can be used by adding them as test function parameters:

```python
def test_with_fixtures(simple_ontology, sample_triples, sample_text):
    """Test using predefined fixtures."""
    # simple_ontology, sample_triples, and sample_text are automatically provided
    pass
```

## Mocking BAML Calls

BAML functions are mocked in unit tests to avoid API calls:

```python
from unittest.mock import patch

@patch('spindle.b.ExtractTriples')
def test_extract(mock_baml_extract, simple_ontology):
    """Test extraction with mocked BAML call."""
    # Setup mock return value
    mock_baml_extract.return_value = ExtractionResult(
        triples=[],
        reasoning="Test"
    )
    
    # Call the function
    extractor = SpindleExtractor(simple_ontology)
    result = extractor.extract(text="test", source_name="test")
    
    # Verify mock was called
    mock_baml_extract.assert_called_once()
```

## Continuous Integration

When setting up CI/CD:

1. Run unit tests on every PR:
```bash
pytest tests/ -m "not integration" --cov=spindle --cov-report=xml
```

2. Run integration tests on main branch only (or scheduled):
```bash
pytest tests/ -m integration
```

3. Fail if coverage drops below threshold (e.g., 80%):
```bash
pytest tests/ -m "not integration" --cov=spindle --cov-report=term --cov-fail-under=80
```

## Debugging Tests

### Run with Debugging

```bash
# Drop into debugger on failure
pytest tests/ --pdb

# Drop into debugger on error
pytest tests/ --pdb --pdbcls=IPython.terminal.debugger:Pdb
```

### Print Output

```bash
# Show print statements
pytest tests/ -s

# Show captured output for failed tests
pytest tests/ --capture=no
```

### Increase Logging

```bash
# Show log output
pytest tests/ --log-cli-level=DEBUG
```

## Coverage Goals

- **Helper Functions**: >90% coverage
- **Core Classes**: >80% coverage
- **Integration Tests**: Coverage not required (focus on correctness)

## Common Issues

### Issue: Tests fail with "ANTHROPIC_API_KEY not set"

**Solution:** Integration tests require an API key. Either:
- Skip them: `pytest tests/ -m "not integration"`
- Set the key: `export ANTHROPIC_API_KEY=your_key_here`

### Issue: Mocked tests not working

**Solution:** Ensure you're patching the correct import path:
```python
# Correct: patch where it's used
@patch('spindle.b.ExtractTriples')

# Incorrect: patch where it's defined
@patch('baml_client.b.ExtractTriples')
```

### Issue: Fixture not found

**Solution:** Ensure fixture is defined in `conftest.py` or imported properly.

## Best Practices

1. **Keep unit tests fast** - Mock all external calls
2. **Use descriptive test names** - `test_extract_with_existing_triples`
3. **One assertion per test** - Or related assertions only
4. **Use fixtures** - Avoid repeating test setup
5. **Test edge cases** - Empty inputs, None values, errors
6. **Run tests before committing** - `pytest tests/ -m "not integration"`
7. **Monitor coverage** - Aim for >80% on critical functions

