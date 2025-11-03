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

import os
import json
import shutil
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

try:
    import kuzu
except ImportError:
    raise ImportError(
        "Kùzu is required for graph database functionality. "
        "Install it with: pip install kuzu>=0.7.0"
    )

from baml_client.types import Triple, SourceMetadata, CharacterSpan


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
    
    def __init__(self, db_path: str = "spindle_graph"):
        """
        Initialize GraphStore with Kùzu database.
        
        Args:
            db_path: Graph name or path. If just a name (e.g., "my_graph"),
                    creates /graphs/my_graph/ directory. If an absolute path,
                    uses that path directly. Defaults to "spindle_graph".
        """
        # Convert to absolute path in /graphs directory structure
        self.db_path = self._resolve_graph_path(db_path)
        self.db = None
        self.conn = None
        
        # Initialize database and schema
        self._initialize_database()
    
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
        # Create database
        self.db = kuzu.Database(self.db_path)
        self.conn = kuzu.Connection(self.db)
        
        # Create schema (if tables don't exist, they'll be created)
        self._create_schema()
    
    def _create_schema(self):
        """Create node and relationship tables with proper schema."""
        # Create Entity node table
        # Note: Kùzu requires checking if table exists before creating
        try:
            self.conn.execute(
                "CREATE NODE TABLE IF NOT EXISTS Entity("
                "name STRING, "
                "type STRING, "
                "metadata STRING, "
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
                "metadata STRING"
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
    
    # ========== Node Operations ==========
    
    def add_node(
        self,
        name: str,
        entity_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a single node to the graph.
        
        Args:
            name: Entity name (must be unique)
            entity_type: Type of entity (e.g., "Person", "Organization")
            metadata: Optional dictionary of additional metadata
        
        Returns:
            True if node was added, False if it already exists
        """
        if metadata is None:
            metadata = {}
        
        # Check if node already exists
        if self.get_node(name) is not None:
            return False
        
        metadata_json = json.dumps(metadata)
        
        try:
            self.conn.execute(
                "CREATE (e:Entity {name: $name, type: $type, metadata: $metadata})",
                parameters={"name": name, "type": entity_type, "metadata": metadata_json}
            )
            return True
        except Exception as e:
            # Failed to create node
            return False
    
    def add_nodes(self, nodes: List[Dict[str, Any]]) -> int:
        """
        Add multiple nodes in bulk.
        
        Args:
            nodes: List of node dictionaries with keys: 'name', 'type', 'metadata'
        
        Returns:
            Number of nodes successfully added
        """
        count = 0
        for node in nodes:
            name = node.get("name")
            entity_type = node.get("type", "Unknown")
            metadata = node.get("metadata", {})
            
            if name and self.add_node(name, entity_type, metadata):
                count += 1
        
        return count
    
    def add_nodes_from_triple(self, triple: Triple) -> Tuple[bool, bool]:
        """
        Extract and add subject and object nodes from a triple.
        
        Args:
            triple: Triple object containing subject and object entities
        
        Returns:
            Tuple of (subject_added, object_added) booleans
        """
        # Extract entity types from triple if available
        # Note: BAML Triple doesn't have subject_type/object_type fields,
        # so we infer from the triple's structure or use "Entity" as default
        
        subject_metadata = {
            "sources": [triple.source.source_name],
            "first_seen": triple.extraction_datetime
        }
        
        object_metadata = {
            "sources": [triple.source.source_name],
            "first_seen": triple.extraction_datetime
        }
        
        subject_added = self.add_node(
            name=triple.subject,
            entity_type="Entity",  # Default type
            metadata=subject_metadata
        )
        
        object_added = self.add_node(
            name=triple.object,
            entity_type="Entity",  # Default type
            metadata=object_metadata
        )
        
        return (subject_added, object_added)
    
    def get_node(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a node by name.
        
        Args:
            name: Entity name
        
        Returns:
            Dictionary with node properties or None if not found
        """
        try:
            result = self.conn.execute(
                "MATCH (e:Entity {name: $name}) RETURN e.name, e.type, e.metadata",
                parameters={"name": name}
            )
            
            rows = result.get_as_df()
            if len(rows) == 0:
                return None
            
            row = rows.iloc[0]
            return {
                "name": row["e.name"],
                "type": row["e.type"],
                "metadata": json.loads(row["e.metadata"]) if row["e.metadata"] else {}
            }
        except Exception as e:
            return None
    
    def update_node(self, name: str, updates: Dict[str, Any]) -> bool:
        """
        Update node properties.
        
        Args:
            name: Entity name
            updates: Dictionary of properties to update (can include 'type', 'metadata')
        
        Returns:
            True if node was updated, False if not found
        """
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
            name: Entity name
        
        Returns:
            True if node was deleted, False if not found
        """
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
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add a single edge to the graph with intelligent evidence merging.
        
        Args:
            subject: Subject entity name
            predicate: Relationship type
            obj: Object entity name
            metadata: Optional dictionary with 'supporting_evidence' (new nested format)
                     Format: {'supporting_evidence': [{'source_nm': '...', 'source_url': '...', 
                              'spans': [{'text': '...', 'start': 0, 'end': 10, 'extraction_datetime': '...'}]}]}
        
        Returns:
            Dictionary with 'success' (bool) and 'message' (str) keys
        """
        if metadata is None:
            metadata = {}
        
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
            try:
                self.conn.execute(
                    "MATCH (s:Entity {name: $subject})-[r:Relationship {predicate: $predicate}]->"
                    "(o:Entity {name: $obj}) "
                    "SET r.supporting_evidence = $evidence, r.metadata = $metadata",
                    parameters={
                        "subject": subject,
                        "predicate": predicate,
                        "obj": obj,
                        "evidence": json.dumps(existing_evidence),
                        "metadata": json.dumps(extra_metadata)
                    }
                )
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
            
            try:
                self.conn.execute(
                    "MATCH (s:Entity {name: $subject}), (o:Entity {name: $obj}) "
                    "CREATE (s)-[r:Relationship {"
                    "predicate: $predicate, "
                    "supporting_evidence: $evidence, "
                    "metadata: $metadata"
                    "}]->(o)",
                    parameters={
                        "subject": subject,
                        "obj": obj,
                        "predicate": predicate,
                        "evidence": json.dumps(new_evidence),
                        "metadata": json.dumps(extra_metadata)
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
                  'object', 'metadata'
        
        Returns:
            Number of edges successfully added
        """
        count = 0
        for edge in edges:
            subject = edge.get("subject")
            predicate = edge.get("predicate")
            obj = edge.get("object")
            metadata = edge.get("metadata", {})
            
            if subject and predicate and obj:
                result = self.add_edge(subject, predicate, obj, metadata)
                if result.get("success"):
                    count += 1
        
        return count
    
    def add_edge_from_triple(self, triple: Triple) -> bool:
        """
        Create an edge from a Triple object with new nested evidence format.
        
        Args:
            triple: Triple object from Spindle extraction
        
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
        
        result = self.add_edge(triple.subject, triple.predicate, triple.object, metadata)
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
            subject: Subject entity name
            predicate: Relationship type
            obj: Object entity name
        
        Returns:
            List with single edge dictionary (for consistency) or None if not found
        """
        try:
            result = self.conn.execute(
                "MATCH (s:Entity {name: $subject})-[r:Relationship {predicate: $predicate}]->(o:Entity {name: $obj}) "
                "RETURN s.name, r.predicate, o.name, r.supporting_evidence, r.metadata",
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
                    "metadata": json.loads(row["r.metadata"]) if row["r.metadata"] else {}
                })
            
            return edges
        except Exception as e:
            return None
    
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
            subject: Subject entity name
            predicate: Relationship type
            obj: Object entity name
            updates: Dictionary of properties to update
        
        Returns:
            True if edge was updated, False if none found
        """
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
            subject: Subject entity name
            predicate: Relationship type
            obj: Object entity name
        
        Returns:
            True if edges were deleted, False if none found
        """
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
        count = 0
        for triple in triples:
            if self.add_edge_from_triple(triple):
                count += 1
        return count
    
    def get_triples(self) -> List[Triple]:
        """
        Export all edges as Triple objects.
        
        Note: Since edges now consolidate evidence from multiple sources,
        this creates one Triple per source within each edge for backward compatibility.
        
        Returns:
            List of Triple objects representing all edges in the graph
        """
        try:
            result = self.conn.execute(
                "MATCH (s:Entity)-[r:Relationship]->(o:Entity) "
                "RETURN s.name, r.predicate, o.name, r.supporting_evidence, r.metadata"
            )
            
            rows = result.get_as_df()
            triples = []
            
            for _, row in rows.iterrows():
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
                        subject=row["s.name"],
                        predicate=row["r.predicate"],
                        object=row["o.name"],
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
            subject: Optional subject name (None = wildcard)
            predicate: Optional predicate name (None = wildcard)
            obj: Optional object name (None = wildcard)
        
        Returns:
            List of matching edge dictionaries with nested evidence structure
        """
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
            f"RETURN s.name, r.predicate, o.name, r.supporting_evidence, r.metadata"
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
                    "metadata": json.loads(row["r.metadata"]) if row["r.metadata"] else {}
                })
            
            return edges
        except Exception as e:
            return []
    
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
                "RETURN s.name, r.predicate, o.name, r.supporting_evidence, r.metadata"
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
                        "metadata": json.loads(row["r.metadata"]) if row["r.metadata"] else {}
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
                "RETURN s.name, r.predicate, o.name, r.supporting_evidence, r.metadata"
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
                        "metadata": json.loads(row["r.metadata"]) if row["r.metadata"] else {}
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

