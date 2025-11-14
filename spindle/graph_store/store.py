"""
GraphStore: Facade class for graph database operations.

This module provides the main GraphStore class that delegates to backend
implementations, maintaining backward compatibility with the original API.
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from spindle.baml_client.types import Triple
from spindle.configuration import GraphStoreSettings, SpindleConfig

from spindle.graph_store.base import GraphStoreBackend
from spindle.graph_store.backends.kuzu import KuzuBackend
from spindle.graph_store.utils import record_graph_event, resolve_graph_path
from spindle.graph_store.triples import triple_to_edge_metadata
from spindle.graph_store.nodes import extract_nodes_from_triple
from spindle.graph_store.resolution import (
    get_duplicate_clusters,
    get_canonical_entity,
    query_with_resolution
)
from spindle.graph_store.embeddings import (
    compute_graph_embeddings,
    update_node_embeddings
)

if TYPE_CHECKING:
    from spindle.vector_store import VectorStore


class GraphStore:
    """
    Persistent graph database for storing and querying Spindle knowledge graphs.
    
    Uses a backend implementation (default: Kùzu) to provide efficient storage and
    querying of knowledge graph triples with full metadata preservation.
    
    Example:
        >>> from spindle import GraphStore
        >>> 
        >>> # Create or connect to database (stored in /graphs/spindle_graph/)
        >>> store = GraphStore()  # Uses default "spindle_graph"
        >>> 
        >>> # Or specify graph name (stored in /graphs/my_graph/)
        >>> store = GraphStore(db_path="my_graph")
        >>> 
        >>> # Add triples from extraction
        >>> store.add_triples(extraction_result.triples)
        >>> 
        >>> # Query by pattern
        >>> results = store.query_by_pattern(predicate="works_at")
        >>> 
        >>> # Close when done
        >>> store.close()
        >>> 
        >>> # Or use as context manager
        >>> with GraphStore("project_kg") as store:
        ...     store.add_triples(triples)
    """
    
    def __init__(
        self,
        db_path: str = "spindle_graph",
        vector_store: Optional['VectorStore'] = None,
        *,
        config: Optional[SpindleConfig] = None,
        backend: Optional[GraphStoreBackend] = None,
    ):
        """
        Initialize GraphStore with backend.
        
        Args:
            db_path: Graph name or path. If just a name (e.g., "my_graph"),
                    creates /graphs/my_graph/ directory. If an absolute path,
                    uses that path directly. Defaults to "spindle_graph".
            vector_store: Optional VectorStore instance for storing embeddings.
                         Required for compute_graph_embeddings() method.
                         Embeddings are computed on-demand, not automatically.
            config: Optional SpindleConfig that supplies default storage paths.
                    When provided and ``db_path`` is left as default, the
                    database will be persisted to ``config.storage.graph_store_path``.
            backend: Optional GraphStoreBackend instance. If not provided,
                    defaults to KuzuBackend.
        """
        self._spindle_config = config
        graph_settings = GraphStoreSettings()
        if self._spindle_config:
            self._spindle_config.storage.ensure_directories()
            graph_settings = self._spindle_config.graph_store
            default_path = (
                graph_settings.db_path_override
                or self._spindle_config.storage.graph_store_path
            )
            if db_path == "spindle_graph":
                db_path = str(default_path)
            elif graph_settings.db_path_override is not None:
                db_path = str(graph_settings.db_path_override)
            if graph_settings.snapshot_dir and graph_settings.auto_snapshot:
                graph_settings.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._graph_settings = graph_settings
        
        # Convert to absolute path in /graphs directory structure
        self.db_path = resolve_graph_path(db_path)
        self.vector_store = vector_store
        
        # Initialize backend (default to KuzuBackend)
        if backend is None:
            self._backend = KuzuBackend(self.db_path)
        else:
            self._backend = backend
        
        self._emit_event(
            "init.complete",
            {
                "db_path": self.db_path,
                "vector_store_enabled": self.vector_store is not None,
            },
        )
    
    def _emit_event(self, name: str, payload: Optional[Dict[str, Any]] = None) -> None:
        """Emit an observability event."""
        event_payload: Dict[str, Any] = {
            "db_path": self.db_path,
        }
        if payload:
            event_payload.update(payload)
        record_graph_event(name, event_payload)
    
    # ========== Graph Management ==========
    
    def create_graph(self, db_path: Optional[str] = None):
        """
        Initialize a new graph database at the specified path.
        
        This closes any existing connection and creates a fresh database.
        
        Args:
            db_path: Optional graph name or path. If just a name (e.g., "my_graph"),
                    creates /graphs/my_graph/ directory. If not provided, uses current path.
        """
        # Close existing connection
        if self._backend:
            self._backend.close()
        
        # Update path if provided
        if db_path:
            self.db_path = resolve_graph_path(db_path)
        
        # Reinitialize backend
        self._backend = KuzuBackend(self.db_path)
    
    def delete_graph(self):
        """
        Delete the entire graph database.
        
        This removes the database file and its parent directory (if empty).
        WARNING: This operation is irreversible!
        """
        self._emit_event("delete_graph.start", {})
        # Close connection
        if self._backend:
            self._backend.close()
            self._backend = None
        
        # Remove database file
        if os.path.exists(self.db_path):
            if os.path.isfile(self.db_path):
                os.remove(self.db_path)
            elif os.path.isdir(self.db_path):
                shutil.rmtree(self.db_path)
            
            # Try to remove parent directory if it's empty
            parent_dir = os.path.dirname(self.db_path)
            try:
                if os.path.exists(parent_dir) and not os.listdir(parent_dir):
                    os.rmdir(parent_dir)
            except OSError:
                # Directory not empty or other error, that's fine
                pass
        self._emit_event("delete_graph.complete", {})
    
    # ========== Node Operations ==========
    
    def add_node(
        self,
        name: str,
        entity_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        description: str = "",
        custom_atts: Optional[Dict[str, Any]] = None,
        vector_index: Optional[str] = None
    ) -> bool:
        """Add a single node to the graph."""
        return self._backend.add_node(
            name, entity_type, metadata, description, custom_atts, vector_index
        )
    
    def add_nodes(self, nodes: List[Dict[str, Any]]) -> int:
        """Add multiple nodes in bulk."""
        count = 0
        for node in nodes:
            name = node.get("name")
            entity_type = node.get("type", "Unknown")
            metadata = node.get("metadata", {})
            description = node.get("description", "")
            custom_atts = node.get("custom_atts")
            vector_index = node.get("vector_index")
            
            if name and self.add_node(
                name, entity_type, metadata,
                description=description,
                custom_atts=custom_atts,
                vector_index=vector_index
            ):
                count += 1
        
        return count
    
    def add_nodes_from_triple(
        self,
        triple: Triple,
        subject_vector_index: Optional[str] = None,
        object_vector_index: Optional[str] = None
    ) -> tuple[bool, bool]:
        """Extract and add subject and object nodes from a triple."""
        subject_node, object_node = extract_nodes_from_triple(
            triple, subject_vector_index, object_vector_index
        )
        
        # Map 'type' to 'entity_type' for add_node signature
        subject_name = subject_node.pop("name")
        subject_entity_type = subject_node.pop("type", "Unknown")
        object_name = object_node.pop("name")
        object_entity_type = object_node.pop("type", "Unknown")
        
        subject_added = self.add_node(
            subject_name,
            subject_entity_type,
            metadata=subject_node.get("metadata"),
            description=subject_node.get("description", ""),
            custom_atts=subject_node.get("custom_atts"),
            vector_index=subject_node.get("vector_index")
        )
        object_added = self.add_node(
            object_name,
            object_entity_type,
            metadata=object_node.get("metadata"),
            description=object_node.get("description", ""),
            custom_atts=object_node.get("custom_atts"),
            vector_index=object_node.get("vector_index")
        )
        
        return (subject_added, object_added)
    
    def get_node(self, name: str) -> Optional[Dict[str, Any]]:
        """Retrieve a node by name."""
        return self._backend.get_node(name)
    
    def nodes(self) -> List[Dict[str, Any]]:
        """Retrieve all nodes in the graph."""
        return self._backend.nodes()
    
    def update_node(self, name: str, updates: Dict[str, Any]) -> bool:
        """Update node properties."""
        return self._backend.update_node(name, updates)
    
    def delete_node(self, name: str) -> bool:
        """Delete a node and all its edges."""
        return self._backend.delete_node(name)
    
    # ========== Edge Operations ==========
    
    def add_edge(
        self,
        subject: str,
        predicate: str,
        obj: str,
        metadata: Optional[Dict[str, Any]] = None,
        vector_index: Optional[str] = None
    ) -> Dict[str, Any]:
        """Add a single edge to the graph with intelligent evidence merging."""
        return self._backend.add_edge(subject, predicate, obj, metadata, vector_index)
    
    def add_edges(self, edges: List[Dict[str, Any]]) -> int:
        """Add multiple edges in bulk."""
        count = 0
        for edge in edges:
            subject = edge.get("subject")
            predicate = edge.get("predicate")
            obj = edge.get("object")
            metadata = edge.get("metadata", {})
            vector_index = edge.get("vector_index")
            
            if subject and predicate and obj:
                result = self.add_edge(subject, predicate, obj, metadata, vector_index=vector_index)
                if result.get("success"):
                    count += 1
        
        return count
    
    def add_edge_from_triple(self, triple: Triple, vector_index: Optional[str] = None) -> bool:
        """Create an edge from a Triple object with new nested evidence format."""
        # First ensure nodes exist
        self.add_nodes_from_triple(triple)
        
        # Convert triple to edge metadata format
        metadata = triple_to_edge_metadata(triple)
        
        # Use entity names for edge creation
        result = self.add_edge(
            triple.subject.name,
            triple.predicate,
            triple.object.name,
            metadata,
            vector_index=vector_index
        )
        return result.get("success", False)
    
    def get_edge(
        self,
        subject: str,
        predicate: str,
        obj: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Retrieve edges matching the exact pattern."""
        return self._backend.get_edge(subject, predicate, obj)
    
    def edges(self) -> List[Dict[str, Any]]:
        """Retrieve all edges in the graph."""
        return self._backend.edges()
    
    def update_edge(
        self,
        subject: str,
        predicate: str,
        obj: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update edge properties."""
        return self._backend.update_edge(subject, predicate, obj, updates)
    
    def delete_edge(self, subject: str, predicate: str, obj: str) -> bool:
        """Delete all edges matching the pattern."""
        return self._backend.delete_edge(subject, predicate, obj)
    
    # ========== Triple Integration ==========
    
    def add_triples(self, triples: List[Triple]) -> int:
        """Bulk import triples from Spindle extraction."""
        self._emit_event(
            "add_triples.start",
            {
                "requested": len(triples),
            },
        )
        count = 0
        try:
            for triple in triples:
                if self.add_edge_from_triple(triple):
                    count += 1
        except Exception as exc:
            self._emit_event(
                "add_triples.error",
                {
                    "requested": len(triples),
                    "added": count,
                    "error": str(exc),
                },
            )
            raise
        self._emit_event(
            "add_triples.complete",
            {
                "requested": len(triples),
                "added": count,
            },
        )
        return count
    
    def get_triples(self) -> List[Triple]:
        """Export all edges as Triple objects with full Entity information."""
        import json
        from spindle.baml_client.types import Entity, SourceMetadata, CharacterSpan, AttributeValue
        
        try:
            result = self._backend.query_cypher(
                "MATCH (s:Entity)-[r:Relationship]->(o:Entity) "
                "RETURN s.name, s.type, s.description, s.custom_atts, "
                "r.predicate, o.name, o.type, o.description, o.custom_atts, "
                "r.supporting_evidence, r.metadata"
            )
            
            triples = []
            
            for row in result:
                # Parse subject entity
                subject_custom_atts = json.loads(row["s.custom_atts"]) if row.get("s.custom_atts") else {}
                subject = Entity(
                    name=row["s.name"],
                    type=row["s.type"],
                    description=row.get("s.description", "") or "",
                    custom_atts={
                        attr_name: AttributeValue(value=attr_data["value"], type=attr_data["type"])
                        for attr_name, attr_data in subject_custom_atts.items()
                    }
                )
                
                # Parse object entity
                object_custom_atts = json.loads(row["o.custom_atts"]) if row.get("o.custom_atts") else {}
                obj = Entity(
                    name=row["o.name"],
                    type=row["o.type"],
                    description=row.get("o.description", "") or "",
                    custom_atts={
                        attr_name: AttributeValue(value=attr_data["value"], type=attr_data["type"])
                        for attr_name, attr_data in object_custom_atts.items()
                    }
                )
                
                # Parse supporting evidence (new nested format)
                evidence_list = json.loads(row["r.supporting_evidence"]) if row.get("r.supporting_evidence") else []
                
                # Create one Triple per source for backward compatibility
                for evidence_source in evidence_list:
                    source_nm = evidence_source.get("source_nm", "")
                    source_url = evidence_source.get("source_url", "")
                    spans_data = evidence_source.get("spans", [])
                    
                    # Convert spans and extract extraction_datetime from first span
                    spans = []
                    extraction_datetime = ""
                    for span_data in spans_data:
                        spans.append(CharacterSpan(
                            text=span_data.get("text", ""),
                            start=span_data.get("start"),
                            end=span_data.get("end")
                        ))
                        # Use first span's datetime
                        if not extraction_datetime:
                            extraction_datetime = span_data.get("extraction_datetime", "")
                    
                    # Create Triple object
                    triple = Triple(
                        subject=subject,
                        predicate=row["r.predicate"],
                        object=obj,
                        source=SourceMetadata(
                            source_name=source_nm,
                            source_url=source_url
                        ),
                        supporting_spans=spans,
                        extraction_datetime=extraction_datetime
                    )
                    triples.append(triple)
            
            return triples
        except Exception as e:
            return []
    
    # ========== Query Operations ==========
    
    def query_by_pattern(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Query edges by pattern with wildcards."""
        self._emit_event(
            "query_by_pattern.start",
            {
                "subject": subject,
                "predicate": predicate,
                "object": obj,
            },
        )
        
        try:
            edges = self._backend.query_by_pattern(subject, predicate, obj)
        except Exception as e:
            self._emit_event(
                "query_by_pattern.error",
                {
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj,
                    "error": str(e),
                },
            )
            return []
        
        self._emit_event(
            "query_by_pattern.complete",
            {
                "subject": subject,
                "predicate": predicate,
                "object": obj,
                "result_count": len(edges),
            },
        )
        return edges
    
    def query_by_source(self, source_name: str) -> List[Dict[str, Any]]:
        """Query edges from a specific source."""
        return self._backend.query_by_source(source_name)
    
    def query_by_date_range(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Query edges by extraction date range."""
        return self._backend.query_by_date_range(start, end)
    
    def query_cypher(self, cypher_query: str) -> List[Dict[str, Any]]:
        """Execute a raw Cypher query."""
        return self._backend.query_cypher(cypher_query)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return self._backend.get_statistics()
    
    # ========== Entity Resolution Support ==========
    
    def get_duplicate_clusters(self) -> List[List[str]]:
        """Find connected components of entities via SAME_AS edges."""
        return get_duplicate_clusters(self._backend)
    
    def get_canonical_entity(self, name: str) -> Optional[str]:
        """Get the canonical (primary) entity name for a given entity."""
        return get_canonical_entity(self._backend, name)
    
    def query_with_resolution(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
        resolve_duplicates: bool = True
    ) -> List[Dict[str, Any]]:
        """Query edges by pattern with optional duplicate resolution."""
        return query_with_resolution(
            self._backend, subject, predicate, obj, resolve_duplicates
        )
    
    # ========== Graph Embedding Methods ==========
    
    def compute_graph_embeddings(
        self,
        vector_store: Optional['VectorStore'] = None,
        dimensions: int = 128,
        walk_length: int = 80,
        num_walks: int = 10,
        p: float = 1.0,
        q: float = 1.0,
        workers: int = 1
    ) -> Dict[str, str]:
        """Compute Node2Vec embeddings for all nodes in the graph and store them."""
        if vector_store is None:
            vector_store = self.vector_store
        if vector_store is None:
            raise ValueError("vector_store is required for computing embeddings")
        
        # Compute embeddings (pass self since GraphEmbeddingGenerator expects GraphStore)
        vector_index_map = compute_graph_embeddings(
            self,
            vector_store,
            dimensions=dimensions,
            walk_length=walk_length,
            num_walks=num_walks,
            p=p,
            q=q,
            workers=workers
        )
        
        # Update all nodes with their vector_index values
        self.update_node_embeddings(vector_index_map)
        
        return vector_index_map
    
    def _extract_graph_for_embeddings(self):
        """Extract graph structure for embedding computation."""
        try:
            from spindle.vector_store import GraphEmbeddingGenerator
        except ImportError:
            raise ImportError(
                "Graph extraction requires optional dependencies. "
                "Ensure all dependencies are installed: pip install networkx"
            )
        
        return GraphEmbeddingGenerator.extract_graph_structure(self)
    
    def update_node_embeddings(self, embeddings: Dict[str, str]) -> int:
        """Update nodes with their computed vector_index values."""
        return update_node_embeddings(self, embeddings)
    
    def close(self):
        """Close database connection."""
        if self._backend:
            self._backend.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
    
    # Expose backend for advanced use cases
    @property
    def db(self):
        """Access to backend database object (for compatibility)."""
        if hasattr(self._backend, 'db'):
            return self._backend.db
        return None
    
    @property
    def conn(self):
        """Access to backend connection object (for compatibility)."""
        if hasattr(self._backend, 'conn'):
            return self._backend.conn
        return None

