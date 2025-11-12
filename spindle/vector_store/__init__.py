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

from spindle.vector_store.base import VectorStore
from spindle.vector_store.chroma import ChromaVectorStore
from spindle.vector_store.embeddings import (
    create_gemini_embedding_function,
    create_huggingface_embedding_function,
    create_openai_embedding_function,
    get_default_embedding_function,
)
from spindle.vector_store.graph_embeddings import GraphEmbeddingGenerator

__all__ = [
    "VectorStore",
    "ChromaVectorStore",
    "GraphEmbeddingGenerator",
    "create_openai_embedding_function",
    "create_huggingface_embedding_function",
    "create_gemini_embedding_function",
    "get_default_embedding_function",
]

