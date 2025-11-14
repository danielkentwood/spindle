"""
GraphStore: Graph database abstraction for storing and querying knowledge graphs.

This module provides a GraphStore class and backend abstraction for storing
and retrieving knowledge graph triples. The design supports multiple backends
(Kùzu now, Neo4j/others in future).

Key Features:
- Abstract base class for extensibility
- Kùzu implementation for embedded graph storage
- Full CRUD operations for nodes and edges
- Pattern-based querying with wildcards
- Source and date-based filtering
- Direct Cypher query support
- Seamless Triple import/export
"""

from spindle.graph_store.store import GraphStore
from spindle.graph_store.base import GraphStoreBackend

# Optionally export backend for advanced users
try:
    from spindle.graph_store.backends.kuzu import KuzuBackend
    _KUZU_AVAILABLE = True
except ImportError:
    KuzuBackend = None
    _KUZU_AVAILABLE = False

__all__ = [
    "GraphStore",
    "GraphStoreBackend",
]

# Conditionally export KuzuBackend if available
if _KUZU_AVAILABLE:
    __all__.append("KuzuBackend")

