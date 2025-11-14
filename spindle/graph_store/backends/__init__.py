"""
Graph store backend implementations.

This module provides concrete implementations of GraphStoreBackend
for various graph database systems.
"""

try:
    from spindle.graph_store.backends.kuzu import KuzuBackend
    _KUZU_AVAILABLE = True
except ImportError:
    KuzuBackend = None
    _KUZU_AVAILABLE = False

__all__ = []

if _KUZU_AVAILABLE:
    __all__.append("KuzuBackend")

