## Graph Store

`GraphStore` is Spindle's facade for persistent graph operations, backed by Kuzu by default.

## Responsibilities

- Persist extracted triples as nodes/edges.
- Preserve metadata attached to triples.
- Provide node/edge/triple query helpers.
- Support entity-resolution utilities (duplicate clusters, canonical entities).

## Basic usage

```python
from spindle import GraphStore

store = GraphStore(db_path="spindle_graph")
store.add_triples(result.triples)

print(len(store.nodes()))
print(len(store.edges()))

store.close()
```

## Configuration-aware usage

```python
from spindle.configuration import SpindleConfig
from spindle import GraphStore

cfg = SpindleConfig.with_root("spindle_storage")
store = GraphStore(config=cfg)
```

With `config`, default graph path is derived from `cfg.storage.graph_store_path`
(unless overridden by `cfg.graph_store.db_path_override`).

## Integration points

- Works with `EntityResolver` as the graph input.
- Can be paired with a `VectorStore` for embedding-assisted workflows.
- Can optionally write to `ProvenanceStore` during graph mutations.
