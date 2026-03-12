from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

from spindle.configuration import (
    DEFAULT_STORAGE_ROOT_NAME,
    GraphStoreSettings,
    ObservabilitySettings,
    SpindleConfig,
    VectorStoreSettings,
    default_config,
    find_stores_root,
    load_config_from_file,
)


# ---------------------------------------------------------------------------
# find_stores_root
# ---------------------------------------------------------------------------

def test_find_stores_root_inside_git_repo() -> None:
    """Inside the spindle git repo find_stores_root returns <repo_root>/stores."""
    result = find_stores_root()
    # Verify the path ends with the stores root name
    assert result.name == DEFAULT_STORAGE_ROOT_NAME
    # The parent must be the git root; confirm it contains a .git directory
    git_root = result.parent
    assert (git_root / ".git").exists(), (
        f"Expected {git_root} to be a git root (has .git), but it does not"
    )


def test_find_stores_root_outside_git(tmp_path: Path) -> None:
    """Outside a git repo find_stores_root falls back to CWD / stores."""
    import os

    original_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 128  # git not in a repo
            result = find_stores_root()
        assert result == tmp_path / DEFAULT_STORAGE_ROOT_NAME
    finally:
        os.chdir(original_cwd)


def test_find_stores_root_git_unavailable(tmp_path: Path) -> None:
    """When git binary is missing the fallback is CWD / stores."""
    import os

    original_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = find_stores_root()
        assert result == tmp_path / DEFAULT_STORAGE_ROOT_NAME
    finally:
        os.chdir(original_cwd)


# ---------------------------------------------------------------------------
# StoragePaths layout
# ---------------------------------------------------------------------------

def test_storage_paths_full_layout(tmp_path: Path) -> None:
    """with_root produces all expected sub-paths under the root."""
    config = SpindleConfig.with_root(tmp_path)
    s = config.storage

    assert s.root == tmp_path
    assert s.kos_dir == tmp_path / "kos"
    assert s.graphs_dir == tmp_path / "graphs"
    assert s.vector_store_dir == tmp_path / "vector_store"
    assert s.graph_store_path == tmp_path / "graphs" / "spindle_graph" / "graph.db"
    assert s.document_store_dir == tmp_path / "documents"
    assert s.log_dir == tmp_path / "logs"
    assert s.provenance_db == tmp_path / "sqlite" / "provenance.db"
    assert s.catalog_db == tmp_path / "sqlite" / "catalog.db"
    assert s.rejection_db == tmp_path / "sqlite" / "rejections.db"
    assert s.event_log_db == tmp_path / "sqlite" / "events.db"


def test_ensure_directories_creates_all_paths(tmp_path: Path) -> None:
    """ensure_directories() creates every configured directory."""
    config = SpindleConfig.with_root(tmp_path)
    config.storage.ensure_directories()

    expected_dirs = [
        config.storage.root,
        config.storage.kos_dir,
        config.storage.graphs_dir,
        config.storage.vector_store_dir,
        config.storage.document_store_dir,
        config.storage.log_dir,
        config.storage.provenance_db.parent,
        config.storage.catalog_db.parent,
        config.storage.rejection_db.parent,
        config.storage.event_log_db.parent,
    ]
    for d in expected_dirs:
        assert d.is_dir(), f"Expected directory to exist: {d}"


# ---------------------------------------------------------------------------
# default_config
# ---------------------------------------------------------------------------

def test_default_config_uses_find_stores_root() -> None:
    """default_config() without an explicit root resolves via find_stores_root."""
    mock_root = Path("/tmp/mock_spindle_stores")
    with patch("spindle.configuration.find_stores_root", return_value=mock_root):
        config = default_config()
    assert config.storage.root == mock_root


def test_default_config_explicit_root(tmp_path: Path) -> None:
    """default_config(root=...) uses the supplied root, bypassing auto-detection."""
    config = default_config(root=tmp_path)
    assert config.storage.root == tmp_path


# ---------------------------------------------------------------------------
# SpindleConfig.with_root (original tests, kept intact)
# ---------------------------------------------------------------------------

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
