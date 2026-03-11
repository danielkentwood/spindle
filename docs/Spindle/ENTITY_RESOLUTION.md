## Entity Resolution

Entity resolution deduplicates semantically equivalent nodes and edges in a graph.

## Core API

Main orchestrator: `EntityResolver`.

```python
from spindle import EntityResolver, ResolutionConfig, GraphStore
from spindle.vector_store import ChromaVectorStore

resolver = EntityResolver(config=ResolutionConfig())
graph_store = GraphStore("spindle_graph")
vector_store = ChromaVectorStore()

result = resolver.resolve_entities(
    graph_store=graph_store,
    vector_store=vector_store,
    apply_to_nodes=True,
    apply_to_edges=True,
    context="domain-specific hints",
)
```

## Pipeline phases

1. Blocking: cluster similar candidates with embedding-based grouping.
2. Matching: compare candidates within blocks.
3. Merging links: add `SAME_AS` edges and compute duplicate clusters.

## API endpoints

- `POST /api/resolution/resolve` (stateless)
- `POST /api/resolution/session/{session_id}/resolve` (session-based)

Both expect graph + vector-store context either directly or via session state.

## Practical notes

- Resolution quality depends on embedding quality and matching config.
- On very small graphs, the resolver may return no merges (expected behavior).
