"""Stage wrapper for entity resolution."""

from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from spindle.entity_resolution.resolver import EntityResolver
    from spindle.graph_store.store import GraphStore
    from spindle.vector_store.chroma import ChromaVectorStore


class EntityResolutionStage:
    """Wraps EntityResolver for spindle-eval Stage protocol.

    Entity resolution is a post-batch operation on the accumulated graph,
    not a per-document pipeline step.  It is available for callers to
    construct and invoke separately from ``get_pipeline_definition()``.

    Args:
        resolver: EntityResolver instance (created with defaults if not provided).
        graph_store: GraphStore containing the knowledge graph.
        vector_store: VectorStore for computing embeddings.
        tracker: Optional tracker for metrics emission.
    """

    name: str = "entity_resolution"

    def __init__(
        self,
        resolver: Optional["EntityResolver"] = None,
        graph_store: Optional["GraphStore"] = None,
        vector_store: Optional["ChromaVectorStore"] = None,
        tracker: Optional[Any] = None,
    ) -> None:
        self._resolver = resolver
        self._graph = graph_store
        self._vector = vector_store
        self._tracker = tracker

    def run(
        self,
        apply_to_nodes: bool = True,
        apply_to_edges: bool = True,
        context: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run entity resolution on the accumulated graph.

        Returns ``{"status": "skipped"}`` when graph_store or vector_store
        are not available.

        Args:
            apply_to_nodes: Whether to resolve node duplicates.
            apply_to_edges: Whether to resolve edge duplicates.
            context: Optional domain context string.

        Returns:
            Dict from ``ResolutionResult.to_dict()`` or skip sentinel.
        """
        if self._graph is None or self._vector is None:
            return {"status": "skipped"}

        resolver = self._resolver or self._create_resolver()
        result = resolver.resolve_entities(
            graph_store=self._graph,
            vector_store=self._vector,
            apply_to_nodes=apply_to_nodes,
            apply_to_edges=apply_to_edges,
            context=context,
        )
        return result.to_dict()

    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "apply_to_nodes": {"type": "boolean"},
                "apply_to_edges": {"type": "boolean"},
                "context": {"type": "string"},
            },
        }

    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "total_nodes_processed": {"type": "integer"},
                "same_as_edges_created": {"type": "integer"},
            },
        }

    def __call__(self, **kwargs: Any) -> Dict[str, Any]:
        return self.run(**kwargs)

    def _create_resolver(self) -> "EntityResolver":
        from spindle.entity_resolution.resolver import EntityResolver
        return EntityResolver(tracker=self._tracker)
