"""
VectorStore: Abstract base class for vector storage implementations.

This module provides the abstract base class that all vector store implementations
must follow, enabling easy extension to new backends (e.g., Google Spanner).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class VectorStore(ABC):
    """
    Abstract base class for vector storage implementations.
    
    This class defines the interface that all vector store implementations
    must follow, enabling easy extension to new backends (e.g., Google Spanner).
    
    Example:
        >>> from spindle import ChromaVectorStore
        >>> 
        >>> # Create vector store
        >>> vector_store = ChromaVectorStore()
        >>> 
        >>> # Add embedding
        >>> uid = vector_store.add("Alice Johnson works at TechCorp", {"type": "edge"})
        >>> 
        >>> # Query similar
        >>> results = vector_store.query("person employment company", top_k=5)
        >>> 
        >>> # Cleanup
        >>> vector_store.close()
    """
    
    @abstractmethod
    def add(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store an embedding and return its UID.
        
        Args:
            text: Text to embed and store
            metadata: Optional dictionary of metadata to store with the embedding
        
        Returns:
            UID (string) of the stored embedding
        """
        pass
    
    @abstractmethod
    def get(self, uid: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve an embedding by UID.
        
        Args:
            uid: Unique identifier of the embedding
        
        Returns:
            Dictionary with 'text', 'embedding', and 'metadata' keys, or None if not found
        """
        pass
    
    @abstractmethod
    def query(self, text: str, top_k: int = 10, metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Perform semantic similarity search.
        
        Args:
            text: Query text to search for
            top_k: Number of results to return
            metadata_filter: Optional metadata filters
        
        Returns:
            List of dictionaries with 'uid', 'text', 'metadata', and 'distance' keys
        """
        pass
    
    @abstractmethod
    def delete(self, uid: str) -> bool:
        """
        Delete an embedding by UID.
        
        Args:
            uid: Unique identifier of the embedding to delete
        
        Returns:
            True if deleted, False if not found
        """
        pass

    def compute_embedding(self, text: str) -> List[float]:
        """
        Compute an embedding vector for the provided text without storing it.

        Implementations should override this if they support direct embedding
        generation. By default this method is not implemented.
        """
        raise NotImplementedError("VectorStore implementations must define compute_embedding()")
    
    @abstractmethod
    def close(self):
        """
        Close the vector store connection and cleanup resources.
        """
        pass
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False

