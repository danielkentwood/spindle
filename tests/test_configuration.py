from __future__ import annotations

from pathlib import Path

from spindle.configuration import (
    GraphStoreSettings,
    IngestionSettings,
    ObservabilitySettings,
    SpindleConfig,
    VectorStoreSettings,
    load_config_from_file,
)
from spindle.ingestion.service import build_config


def test_spindle_config_defaults(tmp_path: Path) -> None:
    config = SpindleConfig.with_root(tmp_path)

    assert config.observability.event_log_url is None
    assert config.ingestion.vector_store_uri is None
    assert config.vector_store.collection_name == "spindle_embeddings"
    assert config.graph_store.embedding_dimensions == 128
    assert config.llm is None


def test_spindle_config_with_custom_settings(tmp_path: Path) -> None:
    observability = ObservabilitySettings(event_log_url="sqlite:///events.db")
    ingestion = IngestionSettings(
        catalog_url="sqlite:///catalog.db",
        vector_store_uri="custom/vector",
        cache_dir=tmp_path / "cache",
        allow_network_requests=True,
        recursive=True,
    )
    vector_store = VectorStoreSettings(
        collection_name="custom_embeddings",
        embedding_model="model-name",
        use_api_fallback=False,
        prefer_local_embeddings=False,
    )
    graph_store = GraphStoreSettings(
        db_path_override=tmp_path / "graph" / "graph.db",
        auto_snapshot=True,
        snapshot_dir=tmp_path / "graph" / "snapshots",
        embedding_dimensions=256,
        auto_compute_embeddings=True,
    )

    config = SpindleConfig.with_root(
        tmp_path,
        observability=observability,
        ingestion=ingestion,
        vector_store=vector_store,
        graph_store=graph_store,
    )

    assert config.observability.event_log_url == "sqlite:///events.db"
    assert config.ingestion.recursive is True
    assert config.vector_store.use_api_fallback is False
    assert config.graph_store.embedding_dimensions == 256
    assert config.graph_store.auto_snapshot is True


def test_load_config_from_legacy_file(tmp_path: Path) -> None:
    config_py = tmp_path / "config.py"
    storage_root = tmp_path / "storage_root"
    config_py.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "from spindle.configuration import SpindleConfig",
                "",
                f"storage_root = Path({repr(str(storage_root))})",
                "",
                "SPINDLE_CONFIG = SpindleConfig.with_root(storage_root)",
                "",
            ]
        )
    )

    loaded = load_config_from_file(config_py)
    assert loaded.storage.root == storage_root
    # Legacy configs should auto-populate new dataclasses with defaults.
    assert loaded.observability.event_log_url is None
    assert loaded.ingestion.catalog_url is None


def test_build_config_prefers_ingestion_settings(tmp_path: Path) -> None:
    ingestion = IngestionSettings(
        catalog_url="sqlite:///ingestion.db",
        vector_store_uri="custom/vector",
        cache_dir=tmp_path / "cache",
        allow_network_requests=True,
    )
    config = SpindleConfig.with_root(tmp_path, ingestion=ingestion)

    ingestion_config = build_config(
        spindle_config=config,
        catalog_url=None,
        vector_store_uri=None,
    )

    assert ingestion_config.catalog_url == "sqlite:///ingestion.db"
    assert ingestion_config.vector_store_uri == "custom/vector"
    assert ingestion_config.cache_dir == ingestion.cache_dir
    assert ingestion_config.allow_network_requests is True
    assert ingestion_config.spindle_config is config

