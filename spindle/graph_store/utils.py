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
    """
    Resolve database path for graph store.
    
    For absolute paths to existing directories (e.g., from test fixtures), 
    append a database file name. For relative names, create in workspace 
    /graphs/<name>/ directory.
    
    Args:
        db_path: Graph name or path
    
    Returns:
        Absolute path for graph database file
    """
    path_obj = Path(db_path)
    
    # If it's an absolute path
    if path_obj.is_absolute():
        # If it's an existing directory, append a database file name
        if os.path.isdir(db_path):
            return str(path_obj / "graph.db")
        # Otherwise, use it as-is (might be a file path or non-existent path)
        return str(path_obj)
    
    # Get workspace root (project root)
    workspace_root = Path(__file__).parent.parent.parent.absolute()
    
    # Create graphs directory path
    graphs_dir = workspace_root / "graphs"
    
    # Extract graph name from path
    # If it's just a name (no slashes), use it directly
    # If it's a path, extract the base name
    graph_name = path_obj.name
    if graph_name.endswith('.db'):
        graph_name = graph_name[:-3]
    
    # Create graph directory: /graphs/<graph_name>/
    graph_dir = graphs_dir / graph_name
    graph_dir.mkdir(parents=True, exist_ok=True)
    
    # Return path to database file within the directory
    return str(graph_dir / "graph.db")

