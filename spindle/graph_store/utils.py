"""
Shared utilities for graph store operations.

This module provides backend-agnostic utility functions used across
graph store implementations.
"""

from typing import Any, Dict
from pathlib import Path
import os

from spindle.observability import get_event_recorder

GRAPH_STORE_RECORDER = get_event_recorder("graph_store")


def record_graph_event(name: str, payload: Dict[str, Any]) -> None:
    """Record a graph store event for observability."""
    GRAPH_STORE_RECORDER.record(name=name, payload=payload)


def resolve_graph_path(db_path: str) -> str:
    """Resolve a database path for the graph store.

    Absolute paths are used as-is (a ``graph.db`` filename is appended when
    the path points to an existing directory).

    Relative names (e.g. ``"my_graph"``) are placed under the auto-detected
    stores root::

        <stores_root>/graphs/<name>/graph.db

    where ``<stores_root>`` is determined by
    :py:func:`~spindle.configuration.find_stores_root`.

    Args:
        db_path: Graph name or path.

    Returns:
        Absolute path string for the Kùzu database directory/file.
    """
    path_obj = Path(db_path)

    # Absolute paths are used directly.
    if path_obj.is_absolute():
        if os.path.isdir(db_path):
            return str(path_obj / "graph.db")
        return str(path_obj)

    # Relative names → stores/graphs/<name>/graph.db
    from spindle.configuration import find_stores_root

    graphs_dir = find_stores_root() / "graphs"

    graph_name = path_obj.name
    if graph_name.endswith(".db"):
        graph_name = graph_name[:-3]

    graph_dir = graphs_dir / graph_name
    graph_dir.mkdir(parents=True, exist_ok=True)

    return str(graph_dir / "graph.db")
