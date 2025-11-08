"""
GraphStore: Kùzu-based graph database persistence for Spindle knowledge graphs.

This module provides a GraphStore class that enables persistent storage and
querying of knowledge graph triples extracted by Spindle using the Kùzu
embedded graph database.

Key Features:
- Embedded Kùzu database (no separate server needed)
- Full CRUD operations for nodes and edges
- Pattern-based querying with wildcards
- Source and date-based filtering
- Direct Cypher query support
- Seamless Triple import/export
"""

import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import kuzu
except ImportError:
    raise ImportError(
        "Kùzu is required for graph database functionality. "
        "Install it with: pip install kuzu>=0.7.0"
    )

from spindle.baml_client.types import Triple, Entity, SourceMetadata, CharacterSpan, AttributeValue

# Import VectorStore with optional dependency handling
try:
    from spindle.vector_store import VectorStore
    _VECTOR_STORE_AVAILABLE = True
except ImportError:
    VectorStore = None
    _VECTOR_STORE_AVAILABLE = False

from spindle.observability import get_event_recorder

GRAPH_STORE_RECORDER = get_event_recorder("graph_store")


def _record_graph_event(name: str, payload: Dict[str, Any]) -> None:
    GRAPH_STORE_RECORDER.record(name=name, payload=payload)


class GraphStore:
    """
    Persistent graph database for storing and querying Spindle knowledge graphs.
    
    Uses Kùzu embedded graph database to provide efficient storage and querying
    of knowledge graph triples with full metadata preservation.
    
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
    
    def __init__(self, db_path: str = "spindle_graph", vector_store: Optional['VectorStore'] = None):
        """
        Initialize GraphStore with Kùzu database.
        
        Args:
            db_path: Graph name or path. If just a name (e.g., "my_graph"),
                    creates /graphs/my_graph/ directory. If an absolute path,
                    uses that path directly. Defaults to "spindle_graph".
            vector_store: Optional VectorStore instance for storing embeddings.
                         Required for compute_graph_embeddings() method.
                         Embeddings are computed on-demand, not automatically.
        """
        # Convert to absolute path in /graphs directory structure
        self.db_path = self._resolve_graph_path(db_path)
        self.db = None
        self.conn = None
        self.vector_store = vector_store
        
        # Initialize database and schema
        self._initialize_database()

        self._emit_event(
            "init.complete",
            {
                "db_path": self.db_path,
                "vector_store_enabled": self.vector_store is not None,
            },
        )

    def _emit_event(self, name: str, payload: Optional[Dict[str, Any]] = None) -> None:
        event_payload: Dict[str, Any] = {
            "db_path": self.db_path,
        }
        if payload:
            event_payload.update(payload)
        _record_graph_event(name, event_payload)
    
    def _resolve_graph_path(self, db_path: str) -> str:
        """
        Resolve database path for Kùzu.
        
        Kùzu expects a path to a database file (not a directory). For absolute 
        paths to existing directories (e.g., from test fixtures), append a 
        database file name. For relative names, create in workspace /graphs/<name>/ 
        directory.
        
        Args:
            db_path: Graph name or path
        
        Returns:
            Absolute path for Kùzu database file
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
        workspace_root = Path(__file__).parent.parent.absolute()
        
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
    
    def _initialize_database(self):
        """Initialize Kùzu database and create schema if needed."""
        self._emit_event("database.initialize.start", {})
        try:
            # Create database
            self.db = kuzu.Database(self.db_path)
            self.conn = kuzu.Connection(self.db)

            # Create schema (if tables don't exist, they'll be created)
            self._create_schema()
        except Exception as exc:
            self._emit_event(
                "database.initialize.error",
                {
                    "error": str(exc),
                },
            )
            raise
        self._emit_event("database.initialize.complete", {})
    
    def _create_schema(self):
        """Create node and relationship tables with proper schema."""
        # Create Entity node table with description and custom_atts
        # Note: Kùzu requires checking if table exists before creating
        try:
            self.conn.execute(
                "CREATE NODE TABLE IF NOT EXISTS Entity("
                "name STRING, "
                "type STRING, "
                "description STRING, "
                "custom_atts STRING, "
                "metadata STRING, "
                "id STRING, "
                "vector_index STRING, "
                "PRIMARY KEY(name)"
                ")"
            )
        except Exception as e:
            # Table might already exist
            pass
        
        # Create Relationship edge table
        try:
            self.conn.execute(
                "CREATE REL TABLE IF NOT EXISTS Relationship("
                "FROM Entity TO Entity, "
                "predicate STRING, "
                "supporting_evidence STRING, "
                "metadata STRING, "
                "id STRING, "
                "vector_index STRING"
                ")"
            )
        except Exception as e:
            # Table might already exist
            pass
    
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
        if self.conn:
            self.close()
        
        # Update path if provided
        if db_path:
            self.db_path = self._resolve_graph_path(db_path)
        
        # Reinitialize
        self._initialize_database()
    
    def delete_graph(self):
        """
        Delete the entire graph database.
        
        This removes the database file and its parent directory (if empty).
        WARNING: This operation is irreversible!
        """
        self._emit_event("delete_graph.start", {})
        # Close connection
        if self.conn:
            self.conn = None
        if self.db:
            self.db = None
        
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
        """
        Add a single node to the graph.
        
        Args:
            name: Entity name (must be unique, will be converted to uppercase)
            entity_type: Type of entity (e.g., "Person", "Organization")
            metadata: Optional dictionary of additional metadata
            description: Entity description
            custom_atts: Optional dictionary of custom attributes with type metadata
            vector_index: Optional UID of the embedding in the vector store
        
        Returns:
            True if node was added, False if it already exists
        """
        if metadata is None:
            metadata = {}
        if custom_atts is None:
            custom_atts = {}
        
        # Convert name to uppercase
        name = name.upper()
        
        # Check if node already exists
        if self.get_node(name) is not None:
            return False
        
        # Generate unique ID
        node_id = str(uuid.uuid4())
        
        metadata_json = json.dumps(metadata)
        custom_atts_json = json.dumps(custom_atts)
        
        try:
            self.conn.execute(
                "CREATE (e:Entity {name: $name, type: $type, description: $description, "
                "custom_atts: $custom_atts, metadata: $metadata, id: $id, vector_index: $vector_index})",
                parameters={
                    "name": name,
                    "type": entity_type,
                    "description": description,
                    "custom_atts": custom_atts_json,
                    "metadata": metadata_json,
                    "id": node_id,
                    "vector_index": vector_index or ""
                }
            )
            return True
        except Exception as e:
            # Failed to create node
            return False
    
    def add_nodes(self, nodes: List[Dict[str, Any]]) -> int:
        """
        Add multiple nodes in bulk.
        
        Args:
            nodes: List of node dictionaries with keys: 'name', 'type', 'metadata', 'vector_index'
        
        Returns:
            Number of nodes successfully added
        """
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
    ) -> Tuple[bool, bool]:
        """
        Extract and add subject and object nodes from a triple.
        
        Args:
            triple: Triple object containing subject and object Entity objects
            subject_vector_index: Optional vector index UID for subject embedding
            object_vector_index: Optional vector index UID for object embedding
        
        Returns:
            Tuple of (subject_added, object_added) booleans
        """
        # Extract subject entity information
        subject_metadata = {
            "sources": [triple.source.source_name],
            "first_seen": triple.extraction_datetime
        }
        
        # Convert AttributeValue objects to serializable dicts
        subject_custom_atts = {
            attr_name: {"value": attr_val.value, "type": attr_val.type}
            for attr_name, attr_val in triple.subject.custom_atts.items()
        }
        
        subject_added = self.add_node(
            name=triple.subject.name,
            entity_type=triple.subject.type,
            description=triple.subject.description,
            custom_atts=subject_custom_atts,
            metadata=subject_metadata,
            vector_index=subject_vector_index
        )
        
        # Extract object entity information
        object_metadata = {
            "sources": [triple.source.source_name],
            "first_seen": triple.extraction_datetime
        }
        
        # Convert AttributeValue objects to serializable dicts
        object_custom_atts = {
            attr_name: {"value": attr_val.value, "type": attr_val.type}
            for attr_name, attr_val in triple.object.custom_atts.items()
        }
        
        object_added = self.add_node(
            name=triple.object.name,
            entity_type=triple.object.type,
            description=triple.object.description,
            custom_atts=object_custom_atts,
            metadata=object_metadata,
            vector_index=object_vector_index
        )
        
        return (subject_added, object_added)
    
    def get_node(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a node by name.
        
        Args:
            name: Entity name (will be converted to uppercase for lookup)
        
        Returns:
            Dictionary with node properties including description, custom_atts, id, and vector_index, or None if not found
        """
        # Convert name to uppercase
        name = name.upper()
        
        try:
            result = self.conn.execute(
                "MATCH (e:Entity {name: $name}) RETURN e.name, e.type, e.description, e.custom_atts, e.metadata, e.id, e.vector_index",
                parameters={"name": name}
            )
            
            rows = result.get_as_df()
            if len(rows) == 0:
                return None
            
            row = rows.iloc[0]
            return {
                "name": row["e.name"],
                "type": row["e.type"],
                "description": row["e.description"] if row["e.description"] else "",
                "custom_atts": json.loads(row["e.custom_atts"]) if row["e.custom_atts"] else {},
                "metadata": json.loads(row["e.metadata"]) if row["e.metadata"] else {},
                "id": row["e.id"] if row["e.id"] else None,
                "vector_index": row["e.vector_index"] if row["e.vector_index"] else None
            }
        except Exception as e:
            return None
    
    def nodes(self) -> List[Dict[str, Any]]:
        """
        Retrieve all nodes in the graph.
        
        Returns:
            List of node dictionaries, each with keys: name, type, description,
            custom_atts, metadata, id, and vector_index
        """
        try:
            result = self.conn.execute(
                "MATCH (e:Entity) RETURN e.name, e.type, e.description, e.custom_atts, e.metadata, e.id, e.vector_index"
            )
            
            rows = result.get_as_df()
            nodes = []
            
            for _, row in rows.iterrows():
                nodes.append({
                    "name": row["e.name"],
                    "type": row["e.type"],
                    "description": row["e.description"] if row["e.description"] else "",
                    "custom_atts": json.loads(row["e.custom_atts"]) if row["e.custom_atts"] else {},
                    "metadata": json.loads(row["e.metadata"]) if row["e.metadata"] else {},
                    "id": row["e.id"] if row["e.id"] else None,
                    "vector_index": row["e.vector_index"] if row["e.vector_index"] else None
                })
            
            return nodes
        except Exception as e:
            return []
    
    def update_node(self, name: str, updates: Dict[str, Any]) -> bool:
        """
        Update node properties.
        
        Args:
            name: Entity name (will be converted to uppercase for lookup)
            updates: Dictionary of properties to update (can include 'type', 'metadata', 'vector_index')
        
        Returns:
            True if node was updated, False if not found
        """
        # Convert name to uppercase
        name = name.upper()
        
        # Check if node exists
        if self.get_node(name) is None:
            return False
        
        # Build update query
        set_clauses = []
        params = {"name": name}
        
        if "type" in updates:
            set_clauses.append("e.type = $type")
            params["type"] = updates["type"]
        
        if "metadata" in updates:
            set_clauses.append("e.metadata = $metadata")
            params["metadata"] = json.dumps(updates["metadata"])
        
        if "vector_index" in updates:
            set_clauses.append("e.vector_index = $vector_index")
            params["vector_index"] = updates["vector_index"] or ""
        
        if not set_clauses:
            return True  # Nothing to update
        
        query = f"MATCH (e:Entity {{name: $name}}) SET {', '.join(set_clauses)}"
        
        try:
            self.conn.execute(query, parameters=params)
            return True
        except Exception as e:
            return False
    
    def delete_node(self, name: str) -> bool:
        """
        Delete a node and all its edges.
        
        Args:
            name: Entity name (will be converted to uppercase for lookup)
        
        Returns:
            True if node was deleted, False if not found
        """
        # Convert name to uppercase
        name = name.upper()
        
        # Check if node exists
        if self.get_node(name) is None:
            return False
        
        try:
            # In Kùzu, we need to delete edges first, then the node
            # Delete outgoing edges
            self.conn.execute(
                "MATCH (e:Entity {name: $name})-[r:Relationship]->() DELETE r",
                parameters={"name": name}
            )
            
            # Delete incoming edges
            self.conn.execute(
                "MATCH ()-[r:Relationship]->(e:Entity {name: $name}) DELETE r",
                parameters={"name": name}
            )
            
            # Delete the node
            self.conn.execute(
                "MATCH (e:Entity {name: $name}) DELETE e",
                parameters={"name": name}
            )
            return True
        except Exception as e:
            return False
    
    # ========== Edge Operations ==========
    
    def _merge_evidence(
        self,
        existing_evidence: List[Dict[str, Any]],
        new_source_nm: str,
        new_source_url: str,
        new_spans: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Merge new evidence into existing evidence with deduplication.
        
        Deduplication Rules:
        - Same source + same span text → Skip, return message
        - Same source + different span → Add span to existing source
        - New source → Add new source entry
        
        Args:
            existing_evidence: List of existing source evidence dicts
            new_source_nm: New source name
            new_source_url: New source URL
            new_spans: List of new span dicts with 'text', 'start', 'end', 'extraction_datetime'
        
        Returns:
            Tuple of (merged_evidence_list, status_message)
        """
        # Find if source already exists
        source_idx = None
        for idx, evidence in enumerate(existing_evidence):
            if evidence.get("source_nm") == new_source_nm:
                source_idx = idx
                break
        
        # If source doesn't exist, add it as new
        if source_idx is None:
            existing_evidence.append({
                "source_nm": new_source_nm,
                "source_url": new_source_url,
                "spans": new_spans
            })
            return (existing_evidence, f"Added new source: {new_source_nm}")
        
        # Source exists - check spans for duplicates
        existing_spans = existing_evidence[source_idx]["spans"]
        existing_span_texts = {span["text"] for span in existing_spans}
        
        added_count = 0
        duplicate_count = 0
        
        for new_span in new_spans:
            if new_span["text"] in existing_span_texts:
                duplicate_count += 1
            else:
                existing_spans.append(new_span)
                existing_span_texts.add(new_span["text"])
                added_count += 1
        
        # Update the source's spans
        existing_evidence[source_idx]["spans"] = existing_spans
        
        # Create status message
        if duplicate_count > 0 and added_count == 0:
            return (existing_evidence, f"All spans already exist for source: {new_source_nm}")
        elif added_count > 0 and duplicate_count > 0:
            return (existing_evidence, f"Added {added_count} new span(s) to source: {new_source_nm} ({duplicate_count} duplicate(s) skipped)")
        else:
            return (existing_evidence, f"Added {added_count} new span(s) to source: {new_source_nm}")
    
    def add_edge(
        self,
        subject: str,
        predicate: str,
        obj: str,
        metadata: Optional[Dict[str, Any]] = None,
        vector_index: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Add a single edge to the graph with intelligent evidence merging.
        
        Args:
            subject: Subject entity name (will be converted to uppercase)
            predicate: Relationship type (will be converted to uppercase)
            obj: Object entity name (will be converted to uppercase)
            metadata: Optional dictionary with 'supporting_evidence' (new nested format)
                     Format: {'supporting_evidence': [{'source_nm': '...', 'source_url': '...', 
                              'spans': [{'text': '...', 'start': 0, 'end': 10, 'extraction_datetime': '...'}]}]}
            vector_index: Optional UID of the embedding in the vector store
        
        Returns:
            Dictionary with 'success' (bool) and 'message' (str) keys
        """
        if metadata is None:
            metadata = {}
        
        # Convert subject, predicate, and object to uppercase
        subject = subject.upper()
        predicate = predicate.upper()
        obj = obj.upper()
        
        # Extract supporting evidence (new nested format)
        new_evidence = metadata.get("supporting_evidence", [])
        
        # Store remaining metadata
        extra_metadata = {k: v for k, v in metadata.items() 
                         if k not in ["supporting_evidence"]}
        
        # Check if edge already exists
        existing_edges = self.get_edge(subject, predicate, obj)
        
        if existing_edges is not None and len(existing_edges) > 0:
            # Edge exists - merge evidence
            existing_edge = existing_edges[0]
            existing_evidence = existing_edge.get("supporting_evidence", [])
            
            # Merge each new source's evidence
            all_messages = []
            for new_source in new_evidence:
                merged_evidence, message = self._merge_evidence(
                    existing_evidence,
                    new_source.get("source_nm", ""),
                    new_source.get("source_url", ""),
                    new_source.get("spans", [])
                )
                existing_evidence = merged_evidence
                all_messages.append(message)
            
            # Update the edge with merged evidence
            # Also update vector_index if provided
            try:
                set_clauses = [
                    "r.supporting_evidence = $evidence",
                    "r.metadata = $metadata"
                ]
                params = {
                    "subject": subject,
                    "predicate": predicate,
                    "obj": obj,
                    "evidence": json.dumps(existing_evidence),
                    "metadata": json.dumps(extra_metadata)
                }
                
                if vector_index is not None:
                    set_clauses.append("r.vector_index = $vector_index")
                    params["vector_index"] = vector_index
                
                query = (
                    "MATCH (s:Entity {name: $subject})-[r:Relationship {predicate: $predicate}]->"
                    "(o:Entity {name: $obj}) "
                    f"SET {', '.join(set_clauses)}"
                )
                
                self.conn.execute(query, parameters=params)
                return {"success": True, "message": "; ".join(all_messages)}
            except Exception as e:
                return {"success": False, "message": f"Failed to update edge: {str(e)}"}
        else:
            # Edge doesn't exist - create new
            # First verify nodes exist
            subject_node = self.get_node(subject)
            obj_node = self.get_node(obj)
            
            if subject_node is None or obj_node is None:
                return {"success": False, "message": f"Nodes do not exist"}
            
            # Generate unique ID
            edge_id = str(uuid.uuid4())
            
            try:
                self.conn.execute(
                    "MATCH (s:Entity {name: $subject}), (o:Entity {name: $obj}) "
                    "CREATE (s)-[r:Relationship {"
                    "predicate: $predicate, "
                    "supporting_evidence: $evidence, "
                    "metadata: $metadata, "
                    "id: $id, "
                    "vector_index: $vector_index"
                    "}]->(o)",
                    parameters={
                        "subject": subject,
                        "obj": obj,
                        "predicate": predicate,
                        "evidence": json.dumps(new_evidence),
                        "metadata": json.dumps(extra_metadata),
                        "id": edge_id,
                        "vector_index": vector_index or ""
                    }
                )
                return {"success": True, "message": "Created new edge"}
            except Exception as e:
                return {"success": False, "message": f"Failed to create edge: {str(e)}"}
    
    def add_edges(self, edges: List[Dict[str, Any]]) -> int:
        """
        Add multiple edges in bulk.
        
        Args:
            edges: List of edge dictionaries with keys: 'subject', 'predicate',
                  'object', 'metadata', 'vector_index'
        
        Returns:
            Number of edges successfully added
        """
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
        """
        Create an edge from a Triple object with new nested evidence format.
        
        Args:
            triple: Triple object from Spindle extraction
            vector_index: Optional UID of the embedding in the vector store
        
        Returns:
            True if edge was added successfully
        """
        # First ensure nodes exist
        self.add_nodes_from_triple(triple)
        
        # Transform to new nested evidence format
        # Move extraction_datetime into each span
        extraction_datetime = triple.extraction_datetime or ""
        
        supporting_evidence = [{
            "source_nm": triple.source.source_name,
            "source_url": triple.source.source_url or "",
            "spans": [
                {
                    "text": span.text,
                    "start": span.start,
                    "end": span.end,
                    "extraction_datetime": extraction_datetime
                }
                for span in triple.supporting_spans
            ]
        }]
        
        # Prepare metadata with new format
        metadata = {
            "supporting_evidence": supporting_evidence
        }
        
        # Use entity names for edge creation
        result = self.add_edge(triple.subject.name, triple.predicate, triple.object.name, metadata, vector_index=vector_index)
        return result.get("success", False)
    
    def get_edge(
        self,
        subject: str,
        predicate: str,
        obj: str
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieve edges matching the exact pattern.
        
        Note: Now returns single edge with consolidated evidence from all sources.
        
        Args:
            subject: Subject entity name (will be converted to uppercase for lookup)
            predicate: Relationship type (will be converted to uppercase for lookup)
            obj: Object entity name (will be converted to uppercase for lookup)
        
        Returns:
            List with single edge dictionary (for consistency) or None if not found
        """
        # Convert parameters to uppercase
        subject = subject.upper()
        predicate = predicate.upper()
        obj = obj.upper()
        
        try:
            result = self.conn.execute(
                "MATCH (s:Entity {name: $subject})-[r:Relationship {predicate: $predicate}]->(o:Entity {name: $obj}) "
                "RETURN s.name, r.predicate, o.name, r.supporting_evidence, r.metadata, r.id, r.vector_index",
                parameters={"subject": subject, "predicate": predicate, "obj": obj}
            )
            
            rows = result.get_as_df()
            if len(rows) == 0:
                return None
            
            edges = []
            for _, row in rows.iterrows():
                edges.append({
                    "subject": row["s.name"],
                    "predicate": row["r.predicate"],
                    "object": row["o.name"],
                    "supporting_evidence": json.loads(row["r.supporting_evidence"]) if row["r.supporting_evidence"] else [],
                    "metadata": json.loads(row["r.metadata"]) if row["r.metadata"] else {},
                    "id": row["r.id"] if row["r.id"] else None,
                    "vector_index": row["r.vector_index"] if row["r.vector_index"] else None
                })
            
            return edges
        except Exception as e:
            return None
    
    def edges(self) -> List[Dict[str, Any]]:
        """
        Retrieve all edges in the graph.
        
        Returns:
            List of edge dictionaries, each with keys: subject, predicate, object,
            supporting_evidence, metadata, id, and vector_index
        """
        try:
            result = self.conn.execute(
                "MATCH (s:Entity)-[r:Relationship]->(o:Entity) "
                "RETURN s.name, r.predicate, o.name, r.supporting_evidence, r.metadata, r.id, r.vector_index"
            )
            
            rows = result.get_as_df()
            edges = []
            
            for _, row in rows.iterrows():
                edges.append({
                    "subject": row["s.name"],
                    "predicate": row["r.predicate"],
                    "object": row["o.name"],
                    "supporting_evidence": json.loads(row["r.supporting_evidence"]) if row["r.supporting_evidence"] else [],
                    "metadata": json.loads(row["r.metadata"]) if row["r.metadata"] else {},
                    "id": row["r.id"] if row["r.id"] else None,
                    "vector_index": row["r.vector_index"] if row["r.vector_index"] else None
                })
            
            return edges
        except Exception as e:
            return []
    
    def update_edge(
        self,
        subject: str,
        predicate: str,
        obj: str,
        updates: Dict[str, Any]
    ) -> bool:
        """
        Update edge properties.
        
        Args:
            subject: Subject entity name (will be converted to uppercase for lookup)
            predicate: Relationship type (will be converted to uppercase for lookup)
            obj: Object entity name (will be converted to uppercase for lookup)
            updates: Dictionary of properties to update (can include 'supporting_evidence', 'metadata', 'vector_index')
        
        Returns:
            True if edge was updated, False if none found
        """
        # Convert parameters to uppercase
        subject = subject.upper()
        predicate = predicate.upper()
        obj = obj.upper()
        
        # Check if edge exists
        if self.get_edge(subject, predicate, obj) is None:
            return False
        
        # Build update query
        set_clauses = []
        params = {"subject": subject, "predicate": predicate, "obj": obj}
        
        if "supporting_evidence" in updates:
            set_clauses.append("r.supporting_evidence = $evidence")
            params["evidence"] = json.dumps(updates["supporting_evidence"])
        
        if "metadata" in updates:
            set_clauses.append("r.metadata = $metadata")
            params["metadata"] = json.dumps(updates["metadata"])
        
        if "vector_index" in updates:
            set_clauses.append("r.vector_index = $vector_index")
            params["vector_index"] = updates["vector_index"] or ""
        
        if not set_clauses:
            return True  # Nothing to update
        
        query = (
            f"MATCH (s:Entity {{name: $subject}})-[r:Relationship {{predicate: $predicate}}]->"
            f"(o:Entity {{name: $obj}}) SET {', '.join(set_clauses)}"
        )
        
        try:
            self.conn.execute(query, parameters=params)
            return True
        except Exception as e:
            return False
    
    def delete_edge(self, subject: str, predicate: str, obj: str) -> bool:
        """
        Delete all edges matching the pattern.
        
        Args:
            subject: Subject entity name (will be converted to uppercase for lookup)
            predicate: Relationship type (will be converted to uppercase for lookup)
            obj: Object entity name (will be converted to uppercase for lookup)
        
        Returns:
            True if edges were deleted, False if none found
        """
        # Convert parameters to uppercase
        subject = subject.upper()
        predicate = predicate.upper()
        obj = obj.upper()
        
        # Check if edges exist
        if self.get_edge(subject, predicate, obj) is None:
            return False
        
        try:
            self.conn.execute(
                "MATCH (s:Entity {name: $subject})-[r:Relationship {predicate: $predicate}]->"
                "(o:Entity {name: $obj}) DELETE r",
                parameters={"subject": subject, "predicate": predicate, "obj": obj}
            )
            return True
        except Exception as e:
            return False
    
    # ========== Triple Integration ==========
    
    def add_triples(self, triples: List[Triple]) -> int:
        """
        Bulk import triples from Spindle extraction.
        
        Args:
            triples: List of Triple objects
        
        Returns:
            Number of triples successfully added
        """
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
        """
        Export all edges as Triple objects with full Entity information.
        
        Note: Since edges now consolidate evidence from multiple sources,
        this creates one Triple per source within each edge for backward compatibility.
        
        Returns:
            List of Triple objects representing all edges in the graph
        """
        try:
            result = self.conn.execute(
                "MATCH (s:Entity)-[r:Relationship]->(o:Entity) "
                "RETURN s.name, s.type, s.description, s.custom_atts, "
                "r.predicate, o.name, o.type, o.description, o.custom_atts, "
                "r.supporting_evidence, r.metadata"
            )
            
            rows = result.get_as_df()
            triples = []
            
            for _, row in rows.iterrows():
                # Parse subject entity
                subject_custom_atts = json.loads(row["s.custom_atts"]) if row["s.custom_atts"] else {}
                subject = Entity(
                    name=row["s.name"],
                    type=row["s.type"],
                    description=row["s.description"] if row["s.description"] else "",
                    custom_atts={
                        attr_name: AttributeValue(value=attr_data["value"], type=attr_data["type"])
                        for attr_name, attr_data in subject_custom_atts.items()
                    }
                )
                
                # Parse object entity
                object_custom_atts = json.loads(row["o.custom_atts"]) if row["o.custom_atts"] else {}
                obj = Entity(
                    name=row["o.name"],
                    type=row["o.type"],
                    description=row["o.description"] if row["o.description"] else "",
                    custom_atts={
                        attr_name: AttributeValue(value=attr_data["value"], type=attr_data["type"])
                        for attr_name, attr_data in object_custom_atts.items()
                    }
                )
                
                # Parse supporting evidence (new nested format)
                evidence_list = json.loads(row["r.supporting_evidence"]) if row["r.supporting_evidence"] else []
                
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
        """
        Query edges by pattern with wildcards.
        
        Args:
            subject: Optional subject name (None = wildcard, will be converted to uppercase if provided)
            predicate: Optional predicate name (None = wildcard, will be converted to uppercase if provided)
            obj: Optional object name (None = wildcard, will be converted to uppercase if provided)
        
        Returns:
            List of matching edge dictionaries with nested evidence structure
        """
        self._emit_event(
            "query_by_pattern.start",
            {
                "subject": subject,
                "predicate": predicate,
                "object": obj,
            },
        )

        # Convert parameters to uppercase if provided
        if subject is not None:
            subject = subject.upper()
        if predicate is not None:
            predicate = predicate.upper()
        if obj is not None:
            obj = obj.upper()
        
        # Build query with optional filters
        where_clauses = []
        params = {}
        
        if subject is not None:
            where_clauses.append("s.name = $subject")
            params["subject"] = subject
        
        if predicate is not None:
            where_clauses.append("r.predicate = $predicate")
            params["predicate"] = predicate
        
        if obj is not None:
            where_clauses.append("o.name = $obj")
            params["obj"] = obj
        
        where_clause = " AND ".join(where_clauses) if where_clauses else "true"
        
        query = (
            f"MATCH (s:Entity)-[r:Relationship]->(o:Entity) "
            f"WHERE {where_clause} "
            f"RETURN s.name, r.predicate, o.name, r.supporting_evidence, r.metadata, r.id, r.vector_index"
        )
        
        try:
            result = self.conn.execute(query, parameters=params)
            rows = result.get_as_df()

            edges = []
            for _, row in rows.iterrows():
                edges.append({
                    "subject": row["s.name"],
                    "predicate": row["r.predicate"],
                    "object": row["o.name"],
                    "supporting_evidence": json.loads(row["r.supporting_evidence"]) if row["r.supporting_evidence"] else [],
                    "metadata": json.loads(row["r.metadata"]) if row["r.metadata"] else {},
                    "id": row.get("r.id") if "r.id" in row else None,
                    "vector_index": row.get("r.vector_index") if "r.vector_index" in row else None
                })
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
        """
        Query edges from a specific source.
        
        Note: Since sources are now nested within supporting_evidence,
        this queries all edges and filters by searching within the evidence structure.
        
        Args:
            source_name: Source name to filter by
        
        Returns:
            List of matching edge dictionaries
        """
        try:
            # Get all edges
            result = self.conn.execute(
                "MATCH (s:Entity)-[r:Relationship]->(o:Entity) "
                "RETURN s.name, r.predicate, o.name, r.supporting_evidence, r.metadata, r.id, r.vector_index"
            )
            
            rows = result.get_as_df()
            edges = []
            
            for _, row in rows.iterrows():
                supporting_evidence = json.loads(row["r.supporting_evidence"]) if row["r.supporting_evidence"] else []
                
                # Check if any source in the evidence matches
                has_source = any(
                    source.get("source_nm") == source_name 
                    for source in supporting_evidence
                )
                
                if has_source:
                    edges.append({
                        "subject": row["s.name"],
                        "predicate": row["r.predicate"],
                        "object": row["o.name"],
                        "supporting_evidence": supporting_evidence,
                        "metadata": json.loads(row["r.metadata"]) if row["r.metadata"] else {},
                        "id": row.get("r.id") if "r.id" in row else None,
                        "vector_index": row.get("r.vector_index") if "r.vector_index" in row else None
                    })
            
            return edges
        except Exception as e:
            return []
    
    def query_by_date_range(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Query edges by extraction date range.
        
        Note: Since extraction_datetime is now at the span level,
        this queries edges that have at least one span within the date range.
        
        Args:
            start: Optional start datetime (inclusive)
            end: Optional end datetime (inclusive)
        
        Returns:
            List of matching edge dictionaries
        """
        try:
            # Get all edges
            result = self.conn.execute(
                "MATCH (s:Entity)-[r:Relationship]->(o:Entity) "
                "RETURN s.name, r.predicate, o.name, r.supporting_evidence, r.metadata, r.id, r.vector_index"
            )
            
            rows = result.get_as_df()
            edges = []
            
            for _, row in rows.iterrows():
                supporting_evidence = json.loads(row["r.supporting_evidence"]) if row["r.supporting_evidence"] else []
                
                # Check if any span in any source matches the date range
                has_matching_date = False
                for source in supporting_evidence:
                    for span in source.get("spans", []):
                        dt_str = span.get("extraction_datetime", "")
                        if not dt_str:
                            continue
                        
                        try:
                            # Try parsing ISO 8601 format
                            dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
                            
                            # Make start/end timezone-aware if they aren't already
                            if start and start.tzinfo is None:
                                from datetime import timezone
                                start_aware = start.replace(tzinfo=timezone.utc)
                            else:
                                start_aware = start
                            
                            if end and end.tzinfo is None:
                                from datetime import timezone
                                end_aware = end.replace(tzinfo=timezone.utc)
                            else:
                                end_aware = end
                            
                            # Check if within date range
                            if start_aware and dt < start_aware:
                                continue
                            if end_aware and dt > end_aware:
                                continue
                            
                            has_matching_date = True
                            break
                        except (ValueError, AttributeError):
                            # Skip if datetime parsing fails
                            continue
                    
                    if has_matching_date:
                        break
                
                if has_matching_date:
                    edges.append({
                        "subject": row["s.name"],
                        "predicate": row["r.predicate"],
                        "object": row["o.name"],
                        "supporting_evidence": supporting_evidence,
                        "metadata": json.loads(row["r.metadata"]) if row["r.metadata"] else {},
                        "id": row.get("r.id") if "r.id" in row else None,
                        "vector_index": row.get("r.vector_index") if "r.vector_index" in row else None
                    })
            
            return edges
        except Exception as e:
            return []
    
    def query_cypher(self, cypher_query: str) -> List[Dict[str, Any]]:
        """
        Execute a raw Cypher query.
        
        Args:
            cypher_query: Cypher query string
        
        Returns:
            List of result dictionaries (keys depend on query)
        """
        try:
            result = self.conn.execute(cypher_query)
            rows = result.get_as_df()
            
            # Convert DataFrame to list of dictionaries
            return rows.to_dict('records')
        except Exception as e:
            return []
    
    # ========== Utility Methods ==========
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get graph statistics.
        
        Returns:
            Dictionary with node count, edge count, sources, etc.
        """
        stats = {
            "node_count": 0,
            "edge_count": 0,
            "sources": [],
            "predicates": [],
            "date_range": None
        }
        
        try:
            # Count nodes
            result = self.conn.execute("MATCH (e:Entity) RETURN count(e) as count")
            rows = result.get_as_df()
            if len(rows) > 0:
                stats["node_count"] = int(rows.iloc[0]["count"])
            
            # Count edges
            result = self.conn.execute("MATCH ()-[r:Relationship]->() RETURN count(r) as count")
            rows = result.get_as_df()
            if len(rows) > 0:
                stats["edge_count"] = int(rows.iloc[0]["count"])
            
            # Get unique sources from nested evidence
            result = self.conn.execute(
                "MATCH ()-[r:Relationship]->() RETURN r.supporting_evidence as evidence"
            )
            rows = result.get_as_df()
            sources_set = set()
            all_dates = []
            
            for _, row in rows.iterrows():
                evidence_list = json.loads(row["evidence"]) if row["evidence"] else []
                for evidence_source in evidence_list:
                    source_nm = evidence_source.get("source_nm", "")
                    if source_nm:
                        sources_set.add(source_nm)
                    
                    # Collect dates from spans
                    for span in evidence_source.get("spans", []):
                        dt_str = span.get("extraction_datetime", "")
                        if dt_str:
                            all_dates.append(dt_str)
            
            stats["sources"] = sorted(list(sources_set))
            
            # Get unique predicates
            result = self.conn.execute(
                "MATCH ()-[r:Relationship]->() RETURN DISTINCT r.predicate as predicate"
            )
            rows = result.get_as_df()
            stats["predicates"] = rows["predicate"].tolist() if len(rows) > 0 else []
            
            # Get date range from collected dates
            if all_dates:
                sorted_dates = sorted(all_dates)
                stats["date_range"] = {
                    "earliest": sorted_dates[0],
                    "latest": sorted_dates[-1]
                }
        except Exception as e:
            pass
        
        return stats
    
    # ========== Graph Embedding Methods ==========
    
    def compute_graph_embeddings(
        self,
        vector_store: 'VectorStore',
        dimensions: int = 128,
        walk_length: int = 80,
        num_walks: int = 10,
        p: float = 1.0,
        q: float = 1.0,
        workers: int = 1
    ) -> Dict[str, str]:
        """
        Compute Node2Vec embeddings for all nodes in the graph and store them.
        
        This method:
        1. Extracts the graph structure from the database
        2. Computes Node2Vec embeddings that capture structural relationships
        3. Stores embeddings in the provided VectorStore
        4. Updates all nodes with their vector_index values
        
        Args:
            vector_store: VectorStore instance to store embeddings in
            dimensions: Dimensionality of embedding vectors (default: 128)
            walk_length: Length of each random walk (default: 80)
            num_walks: Number of random walks per node (default: 10)
            p: Return parameter - controls likelihood of revisiting a node (default: 1.0)
            q: In-out parameter - controls exploration vs exploitation (default: 1.0)
            workers: Number of worker threads (default: 1)
        
        Returns:
            Dictionary mapping node names to vector_index UIDs in VectorStore
        
        Raises:
            ImportError: If required dependencies (networkx, node2vec) are not installed
            ValueError: If vector_store is None
        """
        if vector_store is None:
            raise ValueError("vector_store is required for computing embeddings")
        
        try:
            from spindle.vector_store import GraphEmbeddingGenerator
        except ImportError:
            raise ImportError(
                "Graph embedding computation requires optional dependencies. "
                "Ensure all dependencies are installed: pip install node2vec networkx"
            )
        
        # Compute embeddings
        vector_index_map = GraphEmbeddingGenerator.compute_and_store_embeddings(
            store=self,
            vector_store=vector_store,
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
        """
        Extract graph structure for embedding computation.
        
        This is a private helper method that uses GraphEmbeddingGenerator
        to extract the graph structure.
        
        Returns:
            NetworkX Graph object
        
        Raises:
            ImportError: If required dependencies are not installed
        """
        try:
            from spindle.vector_store import GraphEmbeddingGenerator
        except ImportError:
            raise ImportError(
                "Graph extraction requires optional dependencies. "
                "Ensure all dependencies are installed: pip install networkx"
            )
        
        return GraphEmbeddingGenerator.extract_graph_structure(self)
    
    def update_node_embeddings(self, embeddings: Dict[str, str]) -> int:
        """
        Update nodes with their computed vector_index values.
        
        Args:
            embeddings: Dictionary mapping node names to vector_index UIDs
        
        Returns:
            Number of nodes successfully updated
        """
        updated_count = 0
        
        for node_name, vector_index in embeddings.items():
            try:
                success = self.update_node(
                    node_name,
                    updates={"vector_index": vector_index}
                )
                if success:
                    updated_count += 1
            except Exception:
                # Skip nodes that can't be updated
                continue
        
        return updated_count
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn = None
        if self.db:
            self.db = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False

