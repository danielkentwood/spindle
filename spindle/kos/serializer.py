"""Turtle-Star file I/O and graph merge utilities for the KOS."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import List, Optional


_KOS_GRAPH = "http://spindle.dev/ns/graph/kos"
_ONTOLOGY_GRAPH = "http://spindle.dev/ns/graph/ontology"
_SHAPES_GRAPH = "http://spindle.dev/ns/graph/shapes"
_SCHEME_GRAPH = "http://spindle.dev/ns/graph/scheme"

# Named graph URIs keyed by filename stem
GRAPH_URIS = {
    "kos": _KOS_GRAPH,
    "ontology": _ONTOLOGY_GRAPH,
    "shapes": _SHAPES_GRAPH,
    "scheme": _SCHEME_GRAPH,
}


def load_file_into_store(store: object, path: Path, graph_uri: str) -> int:
    """Parse a Turtle/Turtle-Star file into an Oxigraph store.

    Args:
        store: ``pyoxigraph.Store`` instance.
        path: Path to the ``.ttl`` or ``.ttls`` file.
        graph_uri: Named graph URI string (e.g. ``spndl:graph/kos``).

    Returns:
        Number of triples loaded.

    Raises:
        ImportError: If pyoxigraph is not installed.
    """
    try:
        import pyoxigraph
    except ImportError as exc:
        raise ImportError(
            "pyoxigraph is required for KOS functionality. "
            "Install it with: uv pip install pyoxigraph"
        ) from exc

    if not path.exists():
        return 0

    graph = pyoxigraph.NamedNode(graph_uri)
    content = path.read_bytes()
    # Use standard Turtle for all KOS files (.ttls is just our extension for SKOS/OWL Turtle)
    fmt = pyoxigraph.RdfFormat.TURTLE
    before = len(list(store.quads_for_pattern(None, None, None, graph)))
    store.load(BytesIO(content), fmt, base_iri=None, to_graph=graph)
    after = len(list(store.quads_for_pattern(None, None, None, graph)))
    return after - before


def serialize_store_to_file(store: object, graph_uri: str, output_path: Path) -> None:
    """Serialize a named graph from an Oxigraph store to a Turtle-Star file.

    Args:
        store: ``pyoxigraph.Store`` instance.
        graph_uri: Named graph URI to serialize.
        output_path: Path where the Turtle-Star file will be written.

    Raises:
        ImportError: If pyoxigraph is not installed.
    """
    try:
        import pyoxigraph
    except ImportError as exc:
        raise ImportError(
            "pyoxigraph is required for KOS serialization."
        ) from exc

    graph = pyoxigraph.NamedNode(graph_uri)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    buf = BytesIO()
    store.dump(buf, pyoxigraph.RdfFormat.TURTLE, from_graph=graph)
    output_path.write_bytes(buf.getvalue())


def merge_staging_into_kos(
    kos_dir: Path,
    staging_files: Optional[List[Path]] = None,
) -> int:
    """Merge staging Turtle-Star files into the consolidated kos.ttls.

    Implements a graph union: loads all staging files plus the existing kos.ttls
    into a temporary store, then serializes the merged KOS graph back to kos.ttls.

    Args:
        kos_dir: Root KOS directory (contains kos.ttls and staging/).
        staging_files: Specific staging files to merge.  Defaults to all
                       .ttls files in kos_dir/staging/.

    Returns:
        Total number of triples in the merged kos.ttls graph.

    Raises:
        ImportError: If pyoxigraph is not installed.
    """
    try:
        import pyoxigraph
    except ImportError as exc:
        raise ImportError(
            "pyoxigraph is required for KOS merge operations."
        ) from exc

    if staging_files is None:
        staging_dir = kos_dir / "staging"
        staging_files = list(staging_dir.glob("*.ttls")) if staging_dir.exists() else []

    tmp_store = pyoxigraph.Store()
    kos_graph = pyoxigraph.NamedNode(_KOS_GRAPH)

    # Load existing kos.ttls
    kos_path = kos_dir / "kos.ttls"
    load_file_into_store(tmp_store, kos_path, _KOS_GRAPH)

    # Merge each staging file
    for path in staging_files:
        load_file_into_store(tmp_store, path, _KOS_GRAPH)

    triple_count = len(list(tmp_store.quads_for_pattern(None, None, None, kos_graph)))
    serialize_store_to_file(tmp_store, _KOS_GRAPH, kos_path)
    return triple_count
