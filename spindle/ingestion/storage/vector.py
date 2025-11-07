"""Vector store adapters for chunk persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import chromadb

from spindle.ingestion.types import ChunkArtifact


class ChromaVectorStoreAdapter:
    """Persist chunks into a Chroma collection."""

    def __init__(self, persist_directory: Path, collection_name: str = "spindle") -> None:
        self._client = chromadb.PersistentClient(path=str(persist_directory))
        self._collection = self._client.get_or_create_collection(collection_name)

    def upsert_chunks(self, chunks: Sequence[ChunkArtifact]) -> None:
        if not chunks:
            return
        ids = [chunk.chunk_id for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        embeddings = [list(chunk.embedding) if chunk.embedding else None for chunk in chunks]
        if any(embedding is None for embedding in embeddings):
            embeddings = None
        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

