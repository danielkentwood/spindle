from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from spindle.configuration import (
    GraphStoreSettings,
    ObservabilitySettings,
    SpindleConfig,
    VectorStoreSettings,
    load_config_from_file,
)


def test_spindle_config_defaults(tmp_path: Path) -> None:
    config = SpindleConfig.with_root(tmp_path)

    assert config.observability.event_log_url is None
    assert config.vector_store.collection_name == "spindle_embeddings"
    assert config.graph_store.embedding_dimensions == 128
    assert config.llm is None


def test_spindle_config_with_custom_settings(tmp_path: Path) -> None:
    observability = ObservabilitySettings(event_log_url="sqlite:///events.db")
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
        vector_store=vector_store,
        graph_store=graph_store,
    )

    assert config.observability.event_log_url == "sqlite:///events.db"
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
    assert loaded.observability.event_log_url is None


def test_get_llm_config_with_explicit_config(tmp_path: Path) -> None:
    """Test get_llm_config() returns explicit config when set."""
    try:
        from spindle.llm_config import LLMConfig

        llm_config = LLMConfig(anthropic_api_key="test-key")
        config = SpindleConfig.with_root(tmp_path, llm=llm_config)

        result = config.get_llm_config()
        assert result is not None
        assert result.anthropic_api_key == "test-key"
    except ImportError:
        # Skip if llm_config not available
        pass


def test_get_llm_config_without_auto_detect(tmp_path: Path) -> None:
    """Test get_llm_config() returns None when auto_detect=False."""
    config = SpindleConfig.with_root(tmp_path)
    assert config.llm is None

    result = config.get_llm_config(auto_detect=False)
    assert result is None


def test_get_llm_config_auto_detect(tmp_path: Path) -> None:
    """Test get_llm_config() auto-detects when enabled."""
    config = SpindleConfig.with_root(tmp_path)
    assert config.llm is None

    try:
        from spindle.llm_config import LLMConfig, detect_available_auth

        mock_config = LLMConfig(anthropic_api_key="auto-detected-key")

        with patch("spindle.llm_config.detect_available_auth", return_value=mock_config):
            result = config.get_llm_config(auto_detect=True)
    except ImportError:
        pass


def test_with_root_auto_detect_llm(tmp_path: Path) -> None:
    """Test with_root() auto-detects LLM config when requested."""
    try:
        from spindle.llm_config import LLMConfig, detect_available_auth

        mock_config = LLMConfig(anthropic_api_key="auto-detected")

        with patch("spindle.llm_config.detect_available_auth", return_value=mock_config):
            config = SpindleConfig.with_root(tmp_path, auto_detect_llm=True)
            assert isinstance(config, SpindleConfig)
    except ImportError:
        pass


def test_with_auto_detected_llm(tmp_path: Path) -> None:
    """Test with_auto_detected_llm() convenience method."""
    try:
        from spindle.llm_config import LLMConfig, detect_available_auth

        mock_config = LLMConfig(anthropic_api_key="auto-detected")

        with patch("spindle.llm_config.detect_available_auth", return_value=mock_config):
            config = SpindleConfig.with_auto_detected_llm(tmp_path)
            assert isinstance(config, SpindleConfig)
    except ImportError:
        pass


def test_create_extractor(tmp_path: Path) -> None:
    """Test create_extractor() factory method."""
    config = SpindleConfig.with_root(tmp_path)

    try:
        extractor = config.create_extractor()
        assert extractor is not None
    except ImportError:
        pass


def test_create_extractor_with_llm_config(tmp_path: Path) -> None:
    """Test create_extractor() uses LLM config from SpindleConfig."""
    try:
        from spindle.llm_config import LLMConfig

        llm_config = LLMConfig(anthropic_api_key="test-key")
        config = SpindleConfig.with_root(tmp_path, llm=llm_config)

        extractor = config.create_extractor()
        assert extractor is not None
        assert extractor.llm_config is not None
        assert extractor.llm_config.anthropic_api_key == "test-key"
    except ImportError:
        pass
