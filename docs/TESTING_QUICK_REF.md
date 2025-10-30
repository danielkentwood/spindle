# Testing Quick Reference

## Most Common Commands

```bash
# Run all unit tests (development default)
pytest tests/ -m "not integration"

# Run with coverage
pytest tests/ --cov=spindle --cov-report=term-missing

# Run specific test file
pytest tests/test_helpers.py -v

# Run specific test
pytest tests/test_helpers.py::TestFindSpanIndices::test_exact_match -v

# Run integration tests (requires API key)
pytest tests/ -m integration
```

## Test Structure

```
tests/
├── test_helpers.py          # Helper function tests (44 tests)
├── test_serialization.py    # Serialization tests (22 tests)
├── test_extractor.py        # SpindleExtractor tests (13 tests)
├── test_recommender.py      # OntologyRecommender tests (17 tests)
└── test_integration.py      # Integration tests (12 tests)
```

## Coverage

Current: **91%**

Target: **>80%** for core functions

## Quick Checks

```bash
# Before committing
pytest tests/ -m "not integration" && echo "✓ All tests passed"

# Check coverage threshold
pytest tests/ --cov=spindle --cov-fail-under=80

# Generate HTML coverage report
pytest tests/ --cov=spindle --cov-report=html
open htmlcov/index.html
```

## Test Categories

| Category | Count | Speed | API Calls |
|----------|-------|-------|-----------|
| Unit Tests | 84 | <1s | No (mocked) |
| Integration | 12 | ~30s | Yes (real LLM) |

## Fixtures Available

Common fixtures in `conftest.py`:
- `simple_ontology` - Basic ontology (Person, Organization)
- `complex_ontology` - Extended ontology (5 entity types)
- `sample_triple` - Single triple with metadata
- `sample_triples` - List of triples
- `mock_extraction_result` - Mocked BAML response
- `mock_ontology_recommendation` - Mocked recommendation
- `sample_text` - Simple test text

## Writing New Tests

### Unit Test Template

```python
from unittest.mock import patch

@patch('spindle.b.ExtractTriples')
def test_my_feature(mock_baml, simple_ontology):
    """Test my feature."""
    # Setup mock
    mock_baml.return_value = ExtractionResult(triples=[], reasoning="Test")
    
    # Test code
    extractor = SpindleExtractor(simple_ontology)
    result = extractor.extract(text="test", source_name="test")
    
    # Assertions
    assert result is not None
    mock_baml.assert_called_once()
```

### Integration Test Template

```python
@pytest.mark.integration
def test_real_extraction(skip_if_no_api_key):
    """Test with real LLM."""
    extractor = SpindleExtractor()
    result = extractor.extract(text="Alice works at TechCorp", source_name="Test")
    assert len(result.triples) > 0
```

## Debugging

```bash
# Show print statements
pytest tests/test_helpers.py -s

# Drop into debugger on failure
pytest tests/ --pdb

# Verbose output
pytest tests/ -vv

# Show log output
pytest tests/ --log-cli-level=DEBUG
```

## Common Issues

**"API key not set"**
→ Skip integration tests: `pytest tests/ -m "not integration"`

**"Mock not working"**
→ Check patch path: `@patch('spindle.b.ExtractTriples')`

**"Fixture not found"**
→ Check `conftest.py` or import

## More Info

- Full Guide: `docs/TESTING.md`
- Summary: `docs/TESTING_SUMMARY.md`
- Example CI/CD: `.github/workflows/tests.yml.example`

