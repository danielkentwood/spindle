"""
KuzuBackend: Kùzu database implementation of GraphStoreBackend.

This module provides the Kùzu-specific implementation of the graph store backend,
using the Kùzu embedded graph database for persistent storage.
"""

import json
import os
import shutil
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import kuzu
except ImportError:
    raise ImportError(
        "Kùzu is required for graph database functionality. "
        "Install it with: pip install kuzu>=0.7.0"
    )

from spindle.graph_store.base import GraphStoreBackend
from spindle.graph_store.edges import merge_evidence
from spindle.graph_store.utils import record_graph_event


class KuzuBackend(GraphStoreBackend):
    """
    Kùzu-based implementation of GraphStoreBackend.
    
    Uses the Kùzu embedded graph database to provide efficient storage and
    querying of knowledge graph triples with full metadata preservation.
    """
    
    def __init__(self, db_path: str):
        """
        Initialize Kùzu backend.
        
        Args:
            db_path: Path to the Kùzu database file
        """
        self.db_path = db_path
        self.db = None
        self.conn = None
        self.initialize(db_path)
    
    def initialize(self, db_path: str) -> None:
        """Initialize the database connection and create schema if needed."""
        record_graph_event("database.initialize.start", {"db_path": db_path})
        try:
            # Create database
            self.db = kuzu.Database(db_path)
            self.conn = kuzu.Connection(self.db)
            
            # Create schema (if tables don't exist, they'll be created)
            self._create_schema()
        except Exception as exc:
            record_graph_event(
                "database.initialize.error",
                {
                    "db_path": db_path,
                    "error": str(exc),
                },
            )
            raise
        record_graph_event("database.initialize.complete", {"db_path": db_path})
    
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
    
    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn = None
        if self.db:
            self.db = None
    
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
    
    def get_node(self, name: str) -> Optional[Dict[str, Any]]:
        """Retrieve a node by name."""
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
        """Retrieve all nodes in the graph."""
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
        """Update node properties."""
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
        """Delete a node and all its edges."""
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
    
    def add_edge(
        self,
        subject: str,
        predicate: str,
        obj: str,
        metadata: Optional[Dict[str, Any]] = None,
        vector_index: Optional[str] = None
    ) -> Dict[str, Any]:
        """Add a single edge to the graph with intelligent evidence merging."""
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
                merged_evidence_list, message = merge_evidence(
                    existing_evidence,
                    new_source.get("source_nm", ""),
                    new_source.get("source_url", ""),
                    new_source.get("spans", [])
                )
                existing_evidence = merged_evidence_list
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
    
    def get_edge(
        self,
        subject: str,
        predicate: str,
        obj: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Retrieve edges matching the exact pattern."""
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
        """Retrieve all edges in the graph."""
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
        """Update edge properties."""
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
        """Delete all edges matching the pattern."""
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
    
    # ========== Query Operations ==========
    
    def query_by_pattern(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Query edges by pattern with wildcards."""
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
            
            return edges
        except Exception as e:
            return []
    
    def query_by_source(self, source_name: str) -> List[Dict[str, Any]]:
        """Query edges from a specific source."""
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
        """Query edges by extraction date range."""
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
        """Execute a raw Cypher query."""
        try:
            result = self.conn.execute(cypher_query)
            rows = result.get_as_df()
            
            # Convert DataFrame to list of dictionaries
            return rows.to_dict('records')
        except Exception as e:
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
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

