# Package Refactoring Test Fix

## Issue

After refactoring Spindle into a proper Python package structure, 24 tests were failing with:
```
AttributeError: module 'spindle' has no attribute 'b'
```

## Root Cause

The tests were mocking `spindle.b` (the BAML client), but after the refactoring:
- The BAML client `b` is imported in `spindle/extractor.py`
- It's not exported from `spindle/__init__.py` (correctly, as it's internal)
- Tests needed to mock `spindle.extractor.b` instead

## Solution

Updated all mock decorators in test files:

**Before:**
```python
@patch('spindle.b.ExtractTriples')
@patch('spindle.b.RecommendOntology')
@patch('spindle.b.AnalyzeOntologyExtension')
```

**After:**
```python
@patch('spindle.extractor.b.ExtractTriples')
@patch('spindle.extractor.b.RecommendOntology')
@patch('spindle.extractor.b.AnalyzeOntologyExtension')
```

## Files Modified

- `tests/test_extractor.py`: Updated 10 mock decorators
- `tests/test_recommender.py`: Updated 14 mock decorators

## Command Used

```bash
sed -i '' "s/@patch('spindle\.b\./@patch('spindle.extractor.b./g" \
    tests/test_extractor.py tests/test_recommender.py
```

## Verification

After the fix, all tests should pass:
```bash
pytest tests/ --cov=spindle --cov-report=term-missing
```

## Lesson Learned

When refactoring module structure:
1. Internal imports (like `b` from `baml_client`) should not be re-exported
2. Tests that mock internal dependencies need to be updated to reflect the new module paths
3. Mock paths must match the actual import location, not the public API

## Related Changes

This fix is part of the broader package restructuring that:
- Created `spindle/` package directory
- Moved `spindle.py` → `spindle/extractor.py`
- Moved `graph_store.py` → `spindle/graph_store.py`
- Added `spindle/__init__.py` for public API exports
- Maintained backward compatibility for all public imports

