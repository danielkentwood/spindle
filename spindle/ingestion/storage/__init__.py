"""Storage adapters for documents, chunks, and ingestion metadata."""

from .catalog import DocumentCatalog
from .corpus import CorpusManager
from .factory import create_storage_backends
from .vector import ChromaVectorStoreAdapter

__all__ = [
    "DocumentCatalog",
    "CorpusManager",
    "ChromaVectorStoreAdapter",
    "create_storage_backends",
]

