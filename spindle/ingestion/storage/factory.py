"""Factory helpers for storage backends."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from spindle.ingestion.types import IngestionConfig

from .catalog import DocumentCatalog
from .vector import ChromaVectorStoreAdapter


def create_storage_backends(
    config: IngestionConfig,
) -> Tuple[DocumentCatalog | None, ChromaVectorStoreAdapter | None]:
    catalog: DocumentCatalog | None = None
    vector: ChromaVectorStoreAdapter | None = None

    if config.spindle_config:
        config.spindle_config.storage.ensure_directories()
    if config.cache_dir:
        config.cache_dir.mkdir(parents=True, exist_ok=True)

    if config.catalog_url:
        catalog = DocumentCatalog(config.catalog_url)

    if config.vector_store_uri:
        path = Path(config.vector_store_uri)
        path.mkdir(parents=True, exist_ok=True)
        vector = ChromaVectorStoreAdapter(path)

    return catalog, vector

