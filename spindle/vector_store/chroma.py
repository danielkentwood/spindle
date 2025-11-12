"""
ChromaVectorStore: ChromaDB implementation of VectorStore.

This module provides the ChromaDB-based implementation of the VectorStore
abstract base class for local vector storage with automatic embedding generation.
"""

import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

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

from spindle.configuration import SpindleConfig
from spindle.vector_store.base import VectorStore
from spindle.vector_store.embeddings import (
    _record_vector_event,
    get_default_embedding_function,
)


class ChromaVectorStore(VectorStore):
    """
    Chroma implementation of VectorStore.
    
    Uses ChromaDB for local vector storage with automatic embedding generation
    using sentence-transformers (local) or API fallback (OpenAI/Hugging Face).
    
    Example:
        >>> # Using local embeddings (requires sentence-transformers)
        >>> vector_store = ChromaVectorStore(collection_name="spindle_embeddings")
        >>> 
        >>> # Using API embeddings (fallback when sentence-transformers unavailable)
        >>> from spindle.vector_store import create_openai_embedding_function
        >>> vector_store = ChromaVectorStore(
        ...     embedding_function=create_openai_embedding_function()
        ... )
        >>> 
        >>> # Auto-detect best available (tries local, then API)
        >>> vector_store = ChromaVectorStore()  # Uses best available option
        >>> 
        >>> uid = vector_store.add("Alice works at TechCorp")
        >>> results = vector_store.query("person employment")
        >>> vector_store.close()
    """
    
    def __init__(
        self,
        collection_name: str = "spindle_embeddings",
        persist_directory: Optional[str] = None,
        embedding_model: Optional[str] = None,
        embedding_function: Optional[Callable[[str], List[float]]] = None,
        use_api_fallback: bool = True,
        config: Optional[SpindleConfig] = None,
    ):
        """
        Initialize ChromaVectorStore.
        
        Args:
            collection_name: Name of the Chroma collection
            persist_directory: Optional directory to persist data (defaults to in-memory)
            embedding_model: Name of sentence-transformers model (default: "all-MiniLM-L6-v2")
            embedding_function: Optional custom embedding function
            use_api_fallback: If True, try API fallback when sentence-transformers unavailable
            config: Optional SpindleConfig providing default storage paths
        
        Raises:
            ImportError: If chromadb is not installed
            ValueError: If no embedding method is available
        """
        self._spindle_config = config
        self._persist_directory = (
            Path(persist_directory).expanduser()
            if persist_directory is not None
            else None
        )
        prefer_local_embeddings = True
        if self._spindle_config:
            self._spindle_config.storage.ensure_directories()
            vector_settings = self._spindle_config.vector_store
            prefer_local_embeddings = vector_settings.prefer_local_embeddings
            if self._persist_directory is None:
                self._persist_directory = self._spindle_config.storage.vector_store_dir
            if (
                collection_name == "spindle_embeddings"
                and vector_settings.collection_name
            ):
                collection_name = vector_settings.collection_name
            if embedding_model is None:
                embedding_model = vector_settings.embedding_model
            if not vector_settings.use_api_fallback:
                use_api_fallback = False
        else:
            vector_settings = None

        persist_directory_str = (
            str(self._persist_directory) if self._persist_directory is not None else None
        )
        _record_vector_event(
            "chroma.init.start",
            {
                "collection_name": collection_name,
                "persist_directory": persist_directory_str,
                "embedding_model": embedding_model,
                "embedding_function_provided": embedding_function is not None,
                "use_api_fallback": use_api_fallback,
            },
        )
        if not _CHROMA_AVAILABLE:
            err = ImportError(
                "ChromaDB is required for ChromaVectorStore. "
                "Install it with: pip install chromadb>=0.4.0"
            )
            _record_vector_event(
                "chroma.init.error",
                {
                    "collection_name": collection_name,
                    "error": str(err),
                },
            )
            raise err

        self._vector_settings = vector_settings

        try:
            # Initialize embedding function
            if embedding_function is not None:
                self._embedding_function = embedding_function
            elif _SENTENCE_TRANSFORMERS_AVAILABLE:
                # Use local sentence-transformers
                model_name = embedding_model or "all-MiniLM-L6-v2"
                self._embedding_model = SentenceTransformer(model_name)
                self._embedding_function = lambda text: self._embedding_model.encode(text).tolist()
            elif use_api_fallback:
                # Try to get API fallback
                api_func = get_default_embedding_function(
                    prefer_local=prefer_local_embeddings
                )
                if api_func:
                    self._embedding_function = api_func
                else:
                    raise ValueError(
                        "No embedding method available. Options:\n"
                        "1. Install sentence-transformers: pip install sentence-transformers\n"
                        "2. Set OPENAI_API_KEY environment variable for OpenAI embeddings\n"
                        "3. Set HF_API_KEY environment variable for Hugging Face embeddings\n"
                        "4. Provide a custom embedding_function"
                    )
            else:
                raise ImportError(
                    "sentence-transformers is required for ChromaVectorStore. "
                    "Install it with: pip install sentence-transformers>=2.2.0\n"
                    "Or set use_api_fallback=True to use API embeddings."
                )

            # Initialize Chroma client
            if persist_directory_str:
                self.client = chromadb.PersistentClient(path=persist_directory_str)
            else:
                self.client = chromadb.Client()

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as exc:
            _record_vector_event(
                "chroma.init.error",
                {
                    "collection_name": collection_name,
                    "error": str(exc),
                },
            )
            raise

        self._emit_event(
            "chroma.init.complete",
            {
                "collection_name": collection_name,
                "persist_directory": persist_directory_str,
            },
        )

    def _emit_event(self, name: str, payload: Optional[Dict[str, Any]] = None) -> None:
        base_payload: Dict[str, Any] = {
            "collection_name": getattr(getattr(self, "collection", None), "name", None),
            "persist_directory": self._persist_directory,
        }
        if payload:
            base_payload.update(payload)
        _record_vector_event(name, base_payload)
    
    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text.
        
        Args:
            text: Text to embed
        
        Returns:
            List of floats representing the embedding vector
        """
        return self._embedding_function(text)

    # Public embedding helper for resolution workflows
    def compute_embedding(self, text: str) -> List[float]:
        """
        Compute an embedding for the given text without persisting it.

        Returns:
            List[float]: embedding vector
        """
        return self._generate_embedding(text)
    
    def add(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store an embedding and return its UID.
        
        Args:
            text: Text to embed and store
            metadata: Optional dictionary of metadata to store with the embedding
        
        Returns:
            UID (string) of the stored embedding
        """
        self._emit_event(
            "add.start",
            {
                "has_metadata": bool(metadata),
            },
        )
        # Generate unique ID
        uid = str(uuid.uuid4())

        try:
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
        except Exception as exc:
            self._emit_event(
                "add.error",
                {
                    "error": str(exc),
                },
            )
            raise

        self._emit_event(
            "add.complete",
            {
                "has_metadata": bool(metadata),
            },
        )
        return uid
    
    def add_embedding(
        self,
        embedding: List[float],
        text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store a precomputed embedding vector.
        
        This method allows storing embeddings that were computed externally
        (e.g., using Node2Vec) rather than generated from text.
        
        Args:
            embedding: Precomputed embedding vector as list of floats
            text: Optional text representation (for compatibility/search)
            metadata: Optional dictionary of metadata to store with the embedding
        
        Returns:
            UID (string) of the stored embedding
        """
        self._emit_event(
            "add_embedding.start",
            {
                "has_metadata": bool(metadata),
                "text_provided": text is not None,
            },
        )
        # Generate unique ID
        uid = str(uuid.uuid4())

        try:
            # Prepare metadata
            if metadata is None:
                metadata = {}

            # Use text if provided, otherwise use placeholder
            text_repr = text or f"embedding_{uid[:8]}"

            # Add to Chroma collection
            self.collection.add(
                ids=[uid],
                embeddings=[embedding],
                documents=[text_repr],
                metadatas=[metadata]
            )
        except Exception as exc:
            self._emit_event(
                "add_embedding.error",
                {
                    "error": str(exc),
                },
            )
            raise

        self._emit_event(
            "add_embedding.complete",
            {
                "has_metadata": bool(metadata),
                "text_provided": text is not None,
            },
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
        self._emit_event(
            "get.start",
            {
                "uid": uid,
            },
        )
        try:
            results = self.collection.get(ids=[uid], include=["embeddings", "documents", "metadatas"])

            if not results["ids"]:
                self._emit_event(
                    "get.miss",
                    {
                        "uid": uid,
                    },
                )
                return None

            record = {
                "text": results["documents"][0],
                "embedding": results["embeddings"][0],
                "metadata": results["metadatas"][0] if results["metadatas"] else {}
            }
        except Exception as exc:
            self._emit_event(
                "get.error",
                {
                    "uid": uid,
                    "error": str(exc),
                },
            )
            return None

        self._emit_event(
            "get.complete",
            {
                "uid": uid,
            },
        )
        return record
    
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
        self._emit_event(
            "query.start",
            {
                "top_k": top_k,
                "has_filter": metadata_filter is not None,
            },
        )
        try:
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
        except Exception as exc:
            self._emit_event(
                "query.error",
                {
                    "top_k": top_k,
                    "has_filter": metadata_filter is not None,
                    "error": str(exc),
                },
            )
            raise

        self._emit_event(
            "query.complete",
            {
                "top_k": top_k,
                "result_count": len(formatted_results),
            },
        )
        return formatted_results
    
    def delete(self, uid: str) -> bool:
        """
        Delete an embedding by UID.
        
        Args:
            uid: Unique identifier of the embedding to delete
        
        Returns:
            True if deleted, False if not found
        """
        self._emit_event(
            "delete.start",
            {
                "uid": uid,
            },
        )
        try:
            self.collection.delete(ids=[uid])
        except Exception as exc:
            self._emit_event(
                "delete.error",
                {
                    "uid": uid,
                    "error": str(exc),
                },
            )
            return False

        self._emit_event(
            "delete.complete",
            {
                "uid": uid,
            },
        )
        return True
    
    def close(self):
        """Close the vector store connection."""
        self._emit_event("close.start", {})
        # Chroma client doesn't require explicit close, but we can clear references
        self.client = None
        self.collection = None
        self._emit_event("close.complete", {})

