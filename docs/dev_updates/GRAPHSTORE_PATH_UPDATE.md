# GraphStore Path Structure Update

## Summary

Updated GraphStore to automatically organize all graph databases in a centralized `/graphs` directory at the project root, with each graph in its own subdirectory.

## Changes

### Before
- Graphs were created in the current working directory
- `GraphStore("test_db")` would create `./test_db` in whatever directory you ran the code from
- No consistent organization of graph files

### After
- All graphs are stored in `/graphs/<graph_name>/` at the project root
- `GraphStore("test_db")` creates `/graphs/test_db/data`
- Consistent, predictable location for all graph files
- Independent of current working directory

## Directory Structure

```
/graphs/
  ├── my_graph/
  │   └── data         # Kùzu database files
  ├── test_db/
  │   └── data
  └── spindle_graph/   # Default graph
      └── data
```

## API Changes

### GraphStore Constructor

**Before:**
```python
store = GraphStore("./my_graph.db")  # Created in current directory
```

**After:**
```python
store = GraphStore("my_graph")  # Creates /graphs/my_graph/
# .db extension is automatically removed
store = GraphStore("my_graph.db")  # Still creates /graphs/my_graph/
```

### Environment Variable

**Before:**
```bash
KUZU_DB_PATH=./my_knowledge_graph.db
```

**After:**
```bash
KUZU_DB_PATH=my_knowledge_graph  # Just the name
```

### Default Location

**Before:** `./spindle_graph.db` (in current directory)

**After:** `/graphs/spindle_graph/` (at project root)

## Implementation Details

### Path Resolution

Added `_resolve_graph_path()` method that:
1. Gets the workspace root (project root directory)
2. Creates `/graphs/<graph_name>/` directory structure
3. Removes `.db` extensions from graph names
4. Returns path to database file within the graph directory

### Files Modified

1. **spindle/graph_store.py**
   - Added `_resolve_graph_path()` method
   - Updated `__init__()` to use path resolution
   - Updated `create_graph()` to use path resolution
   - Updated docstrings and examples

2. **docs/GRAPH_STORE.md**
   - Updated Configuration section
   - Added Directory Structure section
   - Updated examples

3. **demos/example_graph_store.py**
   - Updated to use graph names instead of temporary directories
   - Removed tempfile dependency

4. **README.md**
   - Updated GraphStore configuration section
   - Updated API reference for GraphStore

5. **.gitignore**
   - Added `/graphs/` to ignore graph database files
   - Added `*.db` and `*.wal` patterns

## Benefits

1. **Predictable Location**: All graphs in one place, easy to find
2. **Clean Organization**: Each graph has its own subdirectory
3. **Working Directory Independent**: Works the same regardless of where code runs
4. **Easy Cleanup**: Can delete entire `/graphs` directory to remove all databases
5. **Better .gitignore**: Single directory pattern to exclude from version control

## Migration Notes

Existing code using explicit paths like `./my_graph.db` will now create graphs in `/graphs/my_graph/` instead of the current directory. This is intentional and provides better organization.

To continue using custom paths (advanced use case), you would need to modify the code, but the new structure is recommended for all use cases.

## Testing

Created and ran comprehensive tests verifying:
- Default graph creation at `/graphs/spindle_graph/`
- Named graph creation at `/graphs/<name>/`
- `.db` extension removal
- `create_graph()` method with new names
- All graphs properly organized in `/graphs` directory

All tests passed successfully.

## Date

October 30, 2025

