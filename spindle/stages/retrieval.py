"""Stage wrapper for knowledge graph + vector store retrieval."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from spindle.graph_store.store import GraphStore
    from spindle.vector_store.chroma import ChromaVectorStore


class RetrievalStage:
    """Combined retrieval over GraphStore and ChromaVectorStore.

    Supports three modes (controlled by ``mode`` param):
    - ``"local"``  — KOS Aho-Corasick + ANN only
    - ``"global"`` — ChromaDB vector search
    - ``"hybrid"`` — KOS + ChromaDB fused results

    Args:
        graph_store: GraphStore instance (optional).
        vector_store: ChromaVectorStore instance (optional).
        kos_service: KOSService for local search (optional).
        mode: Default retrieval mode.
        top_k: Default number of results.
    """

    name: str = "retrieval"

    def __init__(
        self,
        graph_store: Optional["GraphStore"] = None,
        vector_store: Optional["ChromaVectorStore"] = None,
        kos_service: Optional[Any] = None,
        mode: str = "hybrid",
        top_k: int = 10,
    ) -> None:
        self._graph = graph_store
        self._vector = vector_store
        self._kos = kos_service
        self._mode = mode
        self._top_k = top_k

    def run(
        self,
        query: str,
        mode: Optional[str] = None,
        top_k: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Retrieve relevant context for a query string.

        Args:
            query: Natural language or keyword query.
            mode: Override instance-level mode.
            top_k: Override instance-level top_k.

        Returns:
            Dict with ``graph``, ``vector``, and/or ``kos`` result lists.
        """
        effective_mode = mode or self._mode
        effective_top_k = top_k or self._top_k

        results: Dict[str, Any] = {}

        if effective_mode in ("local", "hybrid") and self._kos is not None:
            results["kos"] = self._kos.search_ann(query, top_k=effective_top_k)

        if effective_mode in ("global", "hybrid") and self._vector is not None:
            results["vector"] = _vector_query(self._vector, query, effective_top_k)

        if self._graph is not None:
            results["graph"] = _graph_query(self._graph, query, effective_top_k)

        return results

    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }

    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "kos": {"type": "array"},
                "vector": {"type": "array"},
                "graph": {"type": "array"},
            },
        }

    def __call__(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        return self.run(query, **kwargs)


def _vector_query(vector_store: Any, query: str, top_k: int) -> List[Dict[str, Any]]:
    try:
        results = vector_store.query(query_texts=[query], n_results=top_k)
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        return [{"text": d, "metadata": m} for d, m in zip(docs, metas)]
    except Exception:
        return []


def _graph_query(graph_store: Any, query: str, top_k: int) -> List[Dict[str, Any]]:
    try:
        triples = graph_store.get_triples(limit=top_k)
        return [t if isinstance(t, dict) else {"triple": str(t)} for t in triples]
    except Exception:
        return []
