"""
VectorStore: Vector database abstraction for storing and querying embeddings.

This module provides a VectorStore abstract base class and implementations
for storing and retrieving embeddings. The design supports multiple backends
(Chroma now, Google Spanner later).

Key Features:
- Abstract base class for extensibility
- Chroma implementation for local vector storage
- Automatic embedding generation using sentence-transformers (local) or API (fallback)
- Semantic similarity search
- Metadata storage with embeddings
"""

from abc import ABC, abstractmethod
import os
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import numpy as np

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

try:
    import networkx as nx
    _NETWORKX_AVAILABLE = True
except ImportError:
    _NETWORKX_AVAILABLE = False

try:
    from node2vec import Node2Vec
    _NODE2VEC_AVAILABLE = True
except ImportError:
    _NODE2VEC_AVAILABLE = False

# Try to import API clients for embeddings
try:
    import openai
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

try:
    from huggingface_hub import InferenceClient
    _HUGGINGFACE_HUB_AVAILABLE = True
except ImportError:
    _HUGGINGFACE_HUB_AVAILABLE = False

try:
    import google.generativeai as genai
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False


if TYPE_CHECKING:
    from spindle.graph_store import GraphStore

from spindle.configuration import SpindleConfig
from spindle.observability import get_event_recorder

VECTOR_STORE_RECORDER = get_event_recorder("vector_store")


def _record_vector_event(name: str, payload: Dict[str, Any]) -> None:
    VECTOR_STORE_RECORDER.record(name=name, payload=payload)


def create_openai_embedding_function(
    model: str = "text-embedding-3-small",
    api_key: Optional[str] = None
) -> Callable[[str], List[float]]:
    """
    Create an embedding function using OpenAI's API.
    
    Args:
        model: OpenAI embedding model name (default: "text-embedding-3-small")
        api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
    
    Returns:
        Callable function that takes text and returns embedding vector
    
    Raises:
        ImportError: If openai package is not installed
        ValueError: If API key is not provided
    """
    if not _OPENAI_AVAILABLE:
        raise ImportError(
            "OpenAI package is required for API embeddings. "
            "Install it with: pip install openai"
        )
    
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key required. Set OPENAI_API_KEY environment variable "
            "or pass api_key parameter."
        )
    
    client = openai.OpenAI(api_key=api_key)
    
    def embed(text: str) -> List[float]:
        response = client.embeddings.create(
            model=model,
            input=text
        )
        return response.data[0].embedding
    
    return embed


def create_huggingface_embedding_function(
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
    api_key: Optional[str] = None
) -> Callable[[str], List[float]]:
    """
    Create an embedding function using Hugging Face Inference API.
    
    Prefers InferenceClient from huggingface_hub if available, falls back to direct API.
    
    Args:
        model: Hugging Face model name
        api_key: Hugging Face API key (defaults to HF_API_KEY env var)
    
    Returns:
        Callable function that takes text and returns embedding vector
    
    Raises:
        ImportError: If neither huggingface_hub nor requests package is installed
        ValueError: If API key is not provided
    """
    api_key = api_key or os.getenv("HF_API_KEY") or os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        raise ValueError(
            "Hugging Face API key required. Set HF_API_KEY environment variable "
            "or pass api_key parameter."
        )
    
    # Try InferenceClient first (preferred method)
    if _HUGGINGFACE_HUB_AVAILABLE:
        client = InferenceClient(token=api_key)
        
        def embed(text: str) -> List[float]:
            result = client.feature_extraction(text, model=model)
            # Handle different response formats
            if isinstance(result, list):
                embedding = result[0] if len(result) > 0 else result
            elif hasattr(result, 'tolist'):
                embedding = result.tolist()
            else:
                embedding = result
            # Ensure we return a list of floats
            if isinstance(embedding, (list, tuple)):
                return list(embedding)
            return embedding
        
        return embed
    
    # Fallback to direct API if huggingface_hub not available
    if not _REQUESTS_AVAILABLE:
        raise ImportError(
            "For Hugging Face API embeddings, either huggingface_hub or requests package is required. "
            "Install with: pip install huggingface_hub (preferred) or pip install requests"
        )
    
    # Use the new Hugging Face Inference API endpoint
    # Old endpoint (deprecated): https://api-inference.huggingface.co/pipeline/feature-extraction/{model}
    # New endpoint: https://api-inference.huggingface.co/models/{model}
    api_url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    def embed(text: str) -> List[float]:
        response = requests.post(api_url, headers=headers, json={"inputs": text}, timeout=30)
        response.raise_for_status()
        result = response.json()
        # Handle different response formats
        if isinstance(result, list):
            return result[0] if len(result) > 0 else result
        elif isinstance(result, dict) and "embeddings" in result:
            return result["embeddings"][0] if result["embeddings"] else []
        else:
            return result
    
    return embed


def create_gemini_embedding_function(
    model: str = "models/embedding-001",
    api_key: Optional[str] = None,
    task_type: str = "retrieval_document"
) -> Callable[[str], List[float]]:
    """
    Create an embedding function using Google Gemini's API.
    
    Args:
        model: Gemini embedding model name (default: "models/embedding-001")
        api_key: Gemini API key (defaults to GEMINI_API_KEY env var)
        task_type: Task type for embedding (default: "retrieval_document")
                  Options: "retrieval_document", "retrieval_query", "semantic_similarity", etc.
    
    Returns:
        Callable function that takes text and returns embedding vector
    
    Raises:
        ImportError: If google-generativeai package is not installed
        ValueError: If API key is not provided
    """
    if not _GEMINI_AVAILABLE:
        raise ImportError(
            "google-generativeai package is required for Gemini API embeddings. "
            "Install it with: pip install google-generativeai"
        )
    
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "Gemini API key required. Set GEMINI_API_KEY environment variable "
            "or pass api_key parameter."
        )
    
    genai.configure(api_key=api_key)
    
    def embed(text: str) -> List[float]:
        embedding = genai.embed_content(
            model=model,
            content=text,
            task_type=task_type
        )
        return embedding['embedding']
    
    return embed


def get_default_embedding_function(
    prefer_local: bool = True,
    openai_api_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    hf_api_key: Optional[str] = None
) -> Optional[Callable[[str], List[float]]]:
    """
    Get the best available embedding function, with fallback priority:
    1. sentence-transformers (if available and prefer_local=True)
    2. OpenAI API (if API key available)
    3. Gemini API (if API key available)
    4. Hugging Face API (if API key available)
    5. None (if nothing available)
    
    Args:
        prefer_local: If True, prefer local sentence-transformers over API
        openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        gemini_api_key: Gemini API key (defaults to GEMINI_API_KEY env var)
        hf_api_key: Hugging Face API key (defaults to HF_API_KEY env var)
    
    Returns:
        Embedding function or None if no options available
    """
    # Try local first if preferred
    if prefer_local and _SENTENCE_TRANSFORMERS_AVAILABLE:
        return None  # Will use SentenceTransformer in ChromaVectorStore
    
    # Try OpenAI API
    openai_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    if openai_key and _OPENAI_AVAILABLE:
        try:
            return create_openai_embedding_function(api_key=openai_key)
        except Exception:
            pass
    
    # Try Gemini API
    gemini_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
    if gemini_key and _GEMINI_AVAILABLE:
        try:
            return create_gemini_embedding_function(api_key=gemini_key)
        except Exception:
            pass
    
    # Try Hugging Face API
    hf_key = hf_api_key or os.getenv("HF_API_KEY") or os.getenv("HUGGINGFACE_API_KEY")
    if hf_key and (_HUGGINGFACE_HUB_AVAILABLE or _REQUESTS_AVAILABLE):
        try:
            return create_huggingface_embedding_function(api_key=hf_key)
        except Exception:
            pass
    
    return None


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
        if self._spindle_config:
            self._spindle_config.storage.ensure_directories()
            if self._persist_directory is None:
                self._persist_directory = self._spindle_config.storage.vector_store_dir

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
                api_func = get_default_embedding_function(prefer_local=False)
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


class GraphEmbeddingGenerator:
    """
    Generator for Node2Vec-based graph structure-aware embeddings.

    This class extracts graph structure from a GraphStore and computes
    Node2Vec embeddings that capture the structural relationships between nodes.

    Example:
        >>> from spindle import GraphStore, ChromaVectorStore, GraphEmbeddingGenerator
        >>>
        >>> store = GraphStore(db_path="my_graph")
        >>> # ... add nodes and edges ...
        >>>
        >>> vector_store = ChromaVectorStore()
        >>> generator = GraphEmbeddingGenerator()
        >>>
        >>> embeddings = generator.compute_and_store_embeddings(
        ...     store, vector_store, dimensions=128
        ... )
    """

    @staticmethod
    def extract_graph_structure(store: 'GraphStore') -> 'nx.Graph':
        """
        Extract graph structure from Kùzu GraphStore as NetworkX format.

        Args:
            store: GraphStore instance to extract from

        Returns:
            NetworkX Graph object with nodes and edges from the store

        Raises:
            ImportError: If networkx is not installed
        """
        if not _NETWORKX_AVAILABLE:
            raise ImportError(
                "networkx is required for graph embedding computation. "
                "Install it with: pip install networkx>=3.0"
            )

        G = nx.Graph()

        # Get all nodes
        nodes_query = """
        MATCH (e:Entity)
        RETURN e.name, e.type, e.description
        """
        nodes = store.query_cypher(nodes_query)

        # Add nodes with attributes
        for node in nodes:
            node_name = node['e.name']
            node_type = node.get('e.type', 'Unknown')
            description = node.get('e.description', '') or ''

            G.add_node(
                node_name,
                type=node_type,
                description=description
            )

        # Get all edges
        edges_query = """
        MATCH (s:Entity)-[r:Relationship]->(o:Entity)
        RETURN s.name, r.predicate, o.name
        """
        edges = store.query_cypher(edges_query)

        # Add edges (with predicate information)
        edge_counts = {}
        for edge in edges:
            source = edge['s.name']
            target = edge['o.name']
            predicate = edge.get('r.predicate', 'RELATED')

            edge_key = (source, target)
            if edge_key in edge_counts:
                edge_counts[edge_key] += 1
                if 'predicates' in G[source][target]:
                    G[source][target]['predicates'].append(predicate)
            else:
                edge_counts[edge_key] = 1
                G.add_edge(
                    source,
                    target,
                    weight=1,
                    predicates=[predicate]
                )

        # Update edge weights based on frequency
        for (source, target), count in edge_counts.items():
            if G.has_edge(source, target):
                G[source][target]['weight'] = count

        return G

    @staticmethod
    def compute_node2vec_embeddings(
        graph: 'nx.Graph',
        dimensions: int = 128,
        walk_length: int = 80,
        num_walks: int = 10,
        p: float = 1.0,
        q: float = 1.0,
        workers: int = 1
    ) -> Dict[str, np.ndarray]:
        """
        Compute Node2Vec embeddings for nodes in the graph.

        Args:
            graph: NetworkX graph to compute embeddings for
            dimensions: Dimensionality of the embedding vectors
            walk_length: Length of each random walk
            num_walks: Number of random walks per node
            p: Return parameter (controls likelihood of immediately revisiting a node)
            q: In-out parameter (controls likelihood of exploring away from start node)
            workers: Number of worker threads to use

        Returns:
            Dictionary mapping node names to numpy array embeddings

        Raises:
            ImportError: If node2vec is not installed
            ValueError: If graph is empty
        """
        if not _NODE2VEC_AVAILABLE:
            raise ImportError(
                "node2vec is required for embedding computation. "
                "Install it with: pip install node2vec>=0.4.5"
            )

        if len(graph.nodes()) == 0:
            raise ValueError("Cannot compute embeddings for empty graph")

        # For single node graphs, return a zero vector
        if len(graph.nodes()) == 1:
            node_name = list(graph.nodes())[0]
            return {node_name: np.zeros(dimensions)}

        # For graphs with no edges, return zero vectors for all nodes
        if len(graph.edges()) == 0:
            embeddings = {}
            for node in graph.nodes():
                embeddings[node] = np.zeros(dimensions)
            return embeddings

        node2vec = Node2Vec(
            graph,
            dimensions=dimensions,
            walk_length=walk_length,
            num_walks=num_walks,
            p=p,
            q=q,
            workers=workers
        )

        model = node2vec.fit(window=10, min_count=1, batch_words=4)

        embeddings = {}
        for node in graph.nodes():
            embeddings[node] = model.wv[node]

        return embeddings

    @staticmethod
    def compute_and_store_embeddings(
        store: 'GraphStore',
        vector_store: 'VectorStore',
        dimensions: int = 128,
        walk_length: int = 80,
        num_walks: int = 10,
        p: float = 1.0,
        q: float = 1.0,
        workers: int = 1
    ) -> Dict[str, str]:
        """
        Compute Node2Vec embeddings and store them in VectorStore.

        This is the main method that:
        1. Extracts graph structure from GraphStore
        2. Computes Node2Vec embeddings
        3. Stores embeddings in VectorStore
        4. Returns mapping of node names to vector_index UIDs

        Args:
            store: GraphStore instance to extract graph from
            vector_store: VectorStore instance to store embeddings in
            dimensions: Dimensionality of embedding vectors
            walk_length: Length of each random walk
            num_walks: Number of random walks per node
            p: Return parameter for Node2Vec
            q: In-out parameter for Node2Vec
            workers: Number of worker threads

        Returns:
            Dictionary mapping node names to vector_index UIDs in VectorStore
        """
        graph = GraphEmbeddingGenerator.extract_graph_structure(store)

        embeddings = GraphEmbeddingGenerator.compute_node2vec_embeddings(
            graph,
            dimensions=dimensions,
            walk_length=walk_length,
            num_walks=num_walks,
            p=p,
            q=q,
            workers=workers
        )

        vector_index_map = {}

        for node_name, embedding_vector in embeddings.items():
            node = store.get_node(node_name)
            node_type = node.get('type', 'Unknown') if node else 'Unknown'

            embedding_list = embedding_vector.tolist()

            text_representation = node_name

            vector_index = vector_store.add_embedding(
                embedding=embedding_list,
                text=text_representation,
                metadata={
                    "type": "node",
                    "entity_type": node_type,
                    "name": node_name,
                    "embedding_method": "node2vec",
                    "dimensions": dimensions
                }
            )

            vector_index_map[node_name] = vector_index

        return vector_index_map
