# GraphStore Pandas Dependency Fix

**Date:** October 30, 2025

## Problem

The `GraphStore.add_triples()` method appeared to be adding triples successfully (returning the correct count), but when querying the database with `query_by_pattern()` or `get_node()`, no results were returned. The `get_statistics()` method also showed 0 edges/nodes even after adding triples.

## Root Cause

The issue was that **pandas was not included in the project dependencies**. 

Kùzu's `Connection.execute()` method returns a `QueryResult` object, and the code uses the `get_as_df()` method to convert results to pandas DataFrames:

```python
rows = result.get_as_df()  # Requires pandas + numpy
```

When pandas was not installed, `get_as_df()` would raise an `ImportError: No module named 'numpy'`, but this exception was being silently caught and suppressed in the error handling code:

```python
try:
    # ... query code ...
    rows = result.get_as_df()
    # ... process rows ...
except Exception as e:
    return None  # Silently swallows the pandas import error!
```

This meant:
1. **Adding triples** appeared to succeed (returned True) because the `CREATE` operations executed successfully
2. **Querying triples** failed silently and returned empty results because `get_as_df()` threw an ImportError that was caught and suppressed

## Solution

Added `pandas>=2.0.0` to the project dependencies in `pyproject.toml`:

```toml
dependencies = [
    "baml-py==0.211.2",
    "python-dotenv==1.0.0",
    "kuzu>=0.7.0",
    "pandas>=2.0.0",  # Required for Kùzu's get_as_df() method
]
```

## Impact

- All GraphStore query operations now work correctly
- Triples can be successfully stored and retrieved
- Tests pass successfully

## Affected Components

- `GraphStore.get_node()` - Returns node data
- `GraphStore.get_edge()` - Returns edge data
- `GraphStore.query_by_pattern()` - Pattern matching queries
- `GraphStore.query_by_source()` - Source-based filtering
- `GraphStore.query_by_date_range()` - Date-based filtering
- `GraphStore.query_cypher()` - Raw Cypher queries
- `GraphStore.get_triples()` - Export triples
- `GraphStore.get_statistics()` - Graph statistics

## Additional Improvements

Fixed a bug in `tests/test_graph_store.py` where the pytestmark skip condition was incorrectly checking for `pytest.importorskip`:

```python
# Before (broken):
pytestmark = pytest.mark.skipif(
    not hasattr(pytest, "importorskip") or pytest.importorskip("kuzu", reason="kuzu not installed"),
    reason="GraphStore tests require kuzu"
)

# After (fixed):
try:
    import kuzu
    KUZU_AVAILABLE = True
except ImportError:
    KUZU_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not KUZU_AVAILABLE,
    reason="GraphStore tests require kuzu"
)
```

## Testing

After the fix, all core GraphStore tests pass:
- `test_add_triples` - PASSED
- `test_add_edge_from_triple` - PASSED  
- `test_get_triples` - PASSED
- `test_query_by_pattern_*` - PASSED
- And many more...

## Recommendation for Future

Consider better error handling that logs warnings or raises more informative exceptions rather than silently returning `None` or `False`. This would make debugging easier in the future.

