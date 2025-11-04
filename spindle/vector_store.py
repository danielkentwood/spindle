"""
VectorStore: Vector database abstraction for storing and querying embeddings.

This module provides a VectorStore abstract base class and implementations
for storing and retrieving embeddings. The design supports multiple backends
(Chroma now, Google Spanner later).

Key Features:
- Abstract base class for extensibility
- Chroma implementation for local vector storage
- Automatic embedding generation using sentence-transformers
- Semantic similarity search
- Metadata storage with embeddings
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable
import uuid

try:
    import chromadb
    from chromadb.config import Settings
    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False


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


class ChromaVectorStore(VectorStore):
    """
    Chroma implementation of VectorStore.
    
    Uses ChromaDB for local vector storage with automatic embedding generation
    using sentence-transformers.
    
    Example:
        >>> vector_store = ChromaVectorStore(collection_name="spindle_embeddings")
        >>> uid = vector_store.add("Alice works at TechCorp")
        >>> results = vector_store.query("person employment")
        >>> vector_store.close()
    """
    
    def __init__(
        self,
        collection_name: str = "spindle_embeddings",
        persist_directory: Optional[str] = None,
        embedding_model: Optional[str] = None,
        embedding_function: Optional[Callable[[str], List[float]]] = None
    ):
        """
        Initialize ChromaVectorStore.
        
        Args:
            collection_name: Name of the Chroma collection
            persist_directory: Optional directory to persist data (defaults to in-memory)
            embedding_model: Name of sentence-transformers model (default: "all-MiniLM-L6-v2")
            embedding_function: Optional custom embedding function
        
        Raises:
            ImportError: If chromadb or sentence-transformers are not installed
        """
        if not _CHROMA_AVAILABLE:
            raise ImportError(
                "ChromaDB is required for ChromaVectorStore. "
                "Install it with: pip install chromadb>=0.4.0"
            )
        
        if not _SENTENCE_TRANSFORMERS_AVAILABLE and embedding_function is None:
            raise ImportError(
                "sentence-transformers is required for ChromaVectorStore. "
                "Install it with: pip install sentence-transformers>=2.2.0"
            )
        
        # Initialize embedding model
        if embedding_function is not None:
            self._embedding_function = embedding_function
        else:
            model_name = embedding_model or "all-MiniLM-L6-v2"
            self._embedding_model = SentenceTransformer(model_name)
            self._embedding_function = lambda text: self._embedding_model.encode(text).tolist()
        
        # Initialize Chroma client
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text.
        
        Args:
            text: Text to embed
        
        Returns:
            List of floats representing the embedding vector
        """
        return self._embedding_function(text)
    
    def add(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store an embedding and return its UID.
        
        Args:
            text: Text to embed and store
            metadata: Optional dictionary of metadata to store with the embedding
        
        Returns:
            UID (string) of the stored embedding
        """
        # Generate unique ID
        uid = str(uuid.uuid4())
        
        # Generate embedding
        embedding = self._generate_embedding(text)
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
        
        # Add to Chroma collection
        self.collection.add(
            ids=[uid],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata]
        )
        
        return uid
    
    def get(self, uid: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve an embedding by UID.
        
        Args:
            uid: Unique identifier of the embedding
        
        Returns:
            Dictionary with 'text', 'embedding', and 'metadata' keys, or None if not found
        """
        try:
            results = self.collection.get(ids=[uid], include=["embeddings", "documents", "metadatas"])
            
            if not results["ids"]:
                return None
            
            return {
                "text": results["documents"][0],
                "embedding": results["embeddings"][0],
                "metadata": results["metadatas"][0] if results["metadatas"] else {}
            }
        except Exception:
            return None
    
    def query(self, text: str, top_k: int = 10, metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Perform semantic similarity search.
        
        Args:
            text: Query text to search for
            top_k: Number of results to return
            metadata_filter: Optional metadata filters (Chroma format)
        
        Returns:
            List of dictionaries with 'uid', 'text', 'metadata', and 'distance' keys
        """
        # Generate query embedding
        query_embedding = self._generate_embedding(text)
        
        # Query Chroma collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=metadata_filter,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        if results["ids"] and len(results["ids"][0]) > 0:
            for i in range(len(results["ids"][0])):
                formatted_results.append({
                    "uid": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] and results["metadatas"][0] else {},
                    "distance": results["distances"][0][i] if results["distances"] else None
                })
        
        return formatted_results
    
    def delete(self, uid: str) -> bool:
        """
        Delete an embedding by UID.
        
        Args:
            uid: Unique identifier of the embedding to delete
        
        Returns:
            True if deleted, False if not found
        """
        try:
            self.collection.delete(ids=[uid])
            return True
        except Exception:
            return False
    
    def close(self):
        """Close the vector store connection."""
        # Chroma client doesn't require explicit close, but we can clear references
        self.client = None
        self.collection = None

