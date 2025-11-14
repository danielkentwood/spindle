"""
GraphStoreBackend: Abstract base class for graph database backends.

This module provides the abstract base class that all graph store backend
implementations must follow, enabling easy extension to new backends
(e.g., Neo4j, ArangoDB).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional


class GraphStoreBackend(ABC):
    """
    Abstract base class for graph store backend implementations.
    
    This class defines the interface that all graph store backends must follow,
    enabling easy extension to new backends (e.g., Neo4j, ArangoDB).
    
    All backends must implement the methods defined here to work with the
    GraphStore facade class.
    """
    
    @abstractmethod
    def initialize(self, db_path: str) -> None:
        """
        Initialize the database connection and create schema if needed.
        
        Args:
            db_path: Path to the database file or directory
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close database connection and cleanup resources."""
        pass
    
    # ========== Node Operations ==========
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def get_node(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a node by name.
        
        Args:
            name: Entity name (will be converted to uppercase for lookup)
        
        Returns:
            Dictionary with node properties or None if not found
        """
        pass
    
    @abstractmethod
    def nodes(self) -> List[Dict[str, Any]]:
        """
        Retrieve all nodes in the graph.
        
        Returns:
            List of node dictionaries
        """
        pass
    
    @abstractmethod
    def update_node(self, name: str, updates: Dict[str, Any]) -> bool:
        """
        Update node properties.
        
        Args:
            name: Entity name (will be converted to uppercase for lookup)
            updates: Dictionary of properties to update
        
        Returns:
            True if node was updated, False if not found
        """
        pass
    
    @abstractmethod
    def delete_node(self, name: str) -> bool:
        """
        Delete a node and all its edges.
        
        Args:
            name: Entity name (will be converted to uppercase for lookup)
        
        Returns:
            True if node was deleted, False if not found
        """
        pass
    
    # ========== Edge Operations ==========
    
    @abstractmethod
    def add_edge(
        self,
        subject: str,
        predicate: str,
        obj: str,
        metadata: Optional[Dict[str, Any]] = None,
        vector_index: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Add a single edge to the graph.
        
        Args:
            subject: Subject entity name (will be converted to uppercase)
            predicate: Relationship type (will be converted to uppercase)
            obj: Object entity name (will be converted to uppercase)
            metadata: Optional dictionary with 'supporting_evidence' (nested format)
            vector_index: Optional UID of the embedding in the vector store
        
        Returns:
            Dictionary with 'success' (bool) and 'message' (str) keys
        """
        pass
    
    @abstractmethod
    def get_edge(
        self,
        subject: str,
        predicate: str,
        obj: str
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieve edges matching the exact pattern.
        
        Args:
            subject: Subject entity name (will be converted to uppercase for lookup)
            predicate: Relationship type (will be converted to uppercase for lookup)
            obj: Object entity name (will be converted to uppercase for lookup)
        
        Returns:
            List with single edge dictionary (for consistency) or None if not found
        """
        pass
    
    @abstractmethod
    def edges(self) -> List[Dict[str, Any]]:
        """
        Retrieve all edges in the graph.
        
        Returns:
            List of edge dictionaries
        """
        pass
    
    @abstractmethod
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
            updates: Dictionary of properties to update
        
        Returns:
            True if edge was updated, False if none found
        """
        pass
    
    @abstractmethod
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
        pass
    
    # ========== Query Operations ==========
    
    @abstractmethod
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
            List of matching edge dictionaries
        """
        pass
    
    @abstractmethod
    def query_by_source(self, source_name: str) -> List[Dict[str, Any]]:
        """
        Query edges from a specific source.
        
        Args:
            source_name: Source name to filter by
        
        Returns:
            List of matching edge dictionaries
        """
        pass
    
    @abstractmethod
    def query_by_date_range(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Query edges by extraction date range.
        
        Args:
            start: Optional start datetime (inclusive)
            end: Optional end datetime (inclusive)
        
        Returns:
            List of matching edge dictionaries
        """
        pass
    
    @abstractmethod
    def query_cypher(self, cypher_query: str) -> List[Dict[str, Any]]:
        """
        Execute a raw query (Cypher for Kùzu, may differ for other backends).
        
        Args:
            cypher_query: Query string
        
        Returns:
            List of result dictionaries (keys depend on query)
        """
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get graph statistics.
        
        Returns:
            Dictionary with node count, edge count, sources, etc.
        """
        pass

