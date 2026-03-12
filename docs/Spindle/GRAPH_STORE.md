## Graph Store

`GraphStore` is Spindle's facade for persistent graph operations, backed by Kuzu by default.

## Responsibilities

- Persist extracted triples as nodes/edges.
- Preserve metadata attached to triples.
- Provide node/edge/triple query helpers.
- Support entity-resolution utilities (duplicate clusters, canonical entities).

## Default path

When no explicit path or config is provided, `GraphStore()` places the
database under the auto-detected stores root:

```
<stores_root>/graphs/spindle_graph/graph.db
```

Named graphs (e.g. `GraphStore(db_path="my_graph")`) live alongside:

```
<stores_root>/graphs/my_graph/graph.db
```

The stores root is `<git_root>/stores` when running inside a git repository,
or `<cwd>/stores` otherwise.

## Basic usage

```python
from spindle import GraphStore

# Uses <stores_root>/graphs/spindle_graph/graph.db by default
store = GraphStore()
store.add_triples(result.triples)

print(len(store.nodes()))
print(len(store.edges()))

store.close()
```

## Configuration-aware usage

Passing a `SpindleConfig` pins the graph path to
`cfg.storage.graph_store_path` (or `cfg.graph_store.db_path_override` if
set) and creates the directory tree on first use:

```python
from spindle.configuration import default_config
from spindle import GraphStore

cfg = default_config()
store = GraphStore(config=cfg)
```

To point the default graph at a custom root:

```python
from spindle.configuration import SpindleConfig
from spindle import GraphStore

cfg = SpindleConfig.with_root("/data/my_project/stores")
store = GraphStore(config=cfg)
# Writes to /data/my_project/stores/graphs/spindle_graph/graph.db
```

## Integration points

- Works with `EntityResolver` as the graph input.
- Can be paired with a `VectorStore` for embedding-assisted workflows.
- Can optionally write to `ProvenanceStore` during graph mutations.
