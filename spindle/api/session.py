"""Session management for stateful API operations."""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from spindle.api.models import SessionInfo


class Session:
    """Represents a single API session with state."""

    def __init__(
        self,
        session_id: str,
        name: Optional[str] = None,
        graph_store_path: Optional[str] = None,
        vector_store_uri: Optional[str] = None,
        catalog_url: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.session_id = session_id
        self.name = name or f"session-{session_id[:8]}"
        self.created_at = datetime.utcnow()
        self.graph_store_path = graph_store_path
        self.vector_store_uri = vector_store_uri
        self.catalog_url = catalog_url
        self.config = config or {}
        
        # Session state
        self.ontology: Optional[Dict[str, Any]] = None
        self.triples: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}

    def to_info(self) -> SessionInfo:
        """Convert to SessionInfo model."""
        return SessionInfo(
            session_id=self.session_id,
            name=self.name,
            created_at=self.created_at,
            graph_store_path=self.graph_store_path,
            vector_store_uri=self.vector_store_uri,
            catalog_url=self.catalog_url,
            ontology=self.ontology,
            triple_count=len(self.triples),
            config=self.config,
        )

    def update_ontology(self, ontology: Dict[str, Any]) -> None:
        """Update the session's ontology."""
        self.ontology = ontology

    def add_triples(self, triples: List[Dict[str, Any]]) -> None:
        """Add triples to the session."""
        self.triples.extend(triples)

    def update_config(self, config: Dict[str, Any]) -> None:
        """Update session configuration."""
        self.config.update(config)


class SessionManager:
    """Manages API sessions with in-memory storage."""

    def __init__(self):
        self._sessions: Dict[str, Session] = {}

    def create_session(
        self,
        name: Optional[str] = None,
        graph_store_path: Optional[str] = None,
        vector_store_uri: Optional[str] = None,
        catalog_url: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Session:
        """Create a new session.
        
        Args:
            name: Optional session name
            graph_store_path: Path to graph store database
            vector_store_uri: Vector store URI
            catalog_url: Document catalog URL
            config: Additional configuration
            
        Returns:
            Created Session object
        """
        session_id = str(uuid.uuid4())
        session = Session(
            session_id=session_id,
            name=name,
            graph_store_path=graph_store_path,
            vector_store_uri=vector_store_uri,
            catalog_url=catalog_url,
            config=config,
        )
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session object or None if not found
        """
        return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if deleted, False if not found
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def list_sessions(self) -> List[SessionInfo]:
        """List all active sessions.
        
        Returns:
            List of SessionInfo objects
        """
        return [session.to_info() for session in self._sessions.values()]

    def session_exists(self, session_id: str) -> bool:
        """Check if a session exists.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session exists
        """
        return session_id in self._sessions


# Global session manager instance
session_manager = SessionManager()

