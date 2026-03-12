"""
Unified configuration primitives for Spindle storage.

The `SpindleConfig` dataclass is the single entry point that downstream
components use to determine storage locations (vector store, graph DB,
document persistence, logging).

LLM configuration can be integrated via the optional `llm` field, which accepts
an :class:`~spindle.llm_config.LLMConfig` instance. See :mod:`spindle.llm_config`
for details on LLM authentication and credential management.

Example usage::

    from spindle.configuration import SpindleConfig, default_config

    # Use auto-detected stores root (git repo root / stores, or CWD / stores)
    config = default_config()
    print(config.storage.kos_dir)

    # Or specify an explicit root:
    config = SpindleConfig.with_root("/path/to/my/stores")
    print(config.storage.vector_store_dir)

The configuration loader can execute a user supplied `config.py` file::

    from spindle.configuration import load_config_from_file

    config = load_config_from_file("/path/to/config.py")

The file must define a variable named ``SPINDLE_CONFIG`` that is an instance
of :class:`SpindleConfig`.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, MutableMapping, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from spindle.llm_config import LLMConfig


DEFAULT_STORAGE_ROOT_NAME = "stores"
CONFIG_SYMBOL_NAME = "SPINDLE_CONFIG"


class ConfigurationError(RuntimeError):
    """Raised when loading a configuration file fails."""


def _ensure_path(path: Path | str) -> Path:
    result = Path(path).expanduser()
    if not result.is_absolute():
        result = result.resolve()
    return result


def find_stores_root() -> Path:
    """Locate the default stores root directory.

    Attempts to find the root of the current git repository via
    ``git rev-parse --show-toplevel``.  Returns ``<git_root>/stores`` when
    successful, otherwise falls back to ``<cwd>/stores``.

    The directory is **not** created by this function; callers that need the
    directory on disk should call :py:meth:`StoragePaths.ensure_directories`.

    Returns:
        Absolute path to the ``stores`` directory.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_root = Path(result.stdout.strip())
            return git_root / DEFAULT_STORAGE_ROOT_NAME
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return Path.cwd() / DEFAULT_STORAGE_ROOT_NAME


@dataclass(slots=True)
class StoragePaths:
    """Filesystem locations used by Spindle.

    All paths are absolute.  Use :py:meth:`SpindleConfig.with_root` to
    construct a coherent set of paths rooted at a single directory, or
    :py:func:`default_config` to use the auto-detected stores root.

    Attributes:
        root: Top-level stores directory that contains all sub-stores.
        kos_dir: Directory for KOS artifacts (kos.ttls, ontology.owl,
            shapes.ttl, staging/).
        graphs_dir: Parent directory for named Kùzu graph databases.
        vector_store_dir: Directory for the ChromaDB persistent client.
        graph_store_path: Full path to the *default* Kùzu database file
            (``<graphs_dir>/spindle_graph/graph.db``).  Named graphs created
            via ``GraphStore(db_path="my_graph")`` live alongside it under
            ``graphs_dir``.
        document_store_dir: Directory for Docling JSON output and related
            document artefacts.
        log_dir: Directory for structured log files.
        provenance_db: Path to the SQLite provenance database.
        catalog_db: Path to the SQLite document catalog database.
        rejection_db: Path to the SQLite KOS rejection-log database.
        event_log_db: Path to the SQLite observability event-log database.
    """

    root: Path
    kos_dir: Path
    graphs_dir: Path
    vector_store_dir: Path
    graph_store_path: Path
    document_store_dir: Path
    log_dir: Path
    provenance_db: Path
    catalog_db: Path
    rejection_db: Path
    event_log_db: Path

    def ensure_directories(self) -> None:
        """Create all directories represented by this configuration."""
        dirs = {
            self.root,
            self.kos_dir,
            self.graphs_dir,
            self.vector_store_dir,
            self.document_store_dir,
            self.log_dir,
        }
        # Parents of file-based paths
        for db_path in (
            self.graph_store_path,
            self.provenance_db,
            self.catalog_db,
            self.rejection_db,
            self.event_log_db,
        ):
            dirs.add(Path(db_path).parent)
        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class ObservabilitySettings:
    """Global observability and logging configuration."""

    event_log_url: str | None = None
    log_level: str = "INFO"
    enable_pipeline_events: bool = True


@dataclass(slots=True)
class VectorStoreSettings:
    """Settings that influence vector store creation."""

    collection_name: str = "spindle_embeddings"
    embedding_model: str | None = None
    use_api_fallback: bool = True
    prefer_local_embeddings: bool = True


@dataclass(slots=True)
class GraphStoreSettings:
    """Defaults for graph store persistence."""

    db_path_override: Path | None = None
    auto_snapshot: bool = False
    snapshot_dir: Path | None = None
    embedding_dimensions: int = 128
    auto_compute_embeddings: bool = False

    def __post_init__(self) -> None:
        if self.db_path_override is not None:
            self.db_path_override = _ensure_path(self.db_path_override)
        if self.snapshot_dir is not None:
            self.snapshot_dir = _ensure_path(self.snapshot_dir)


@dataclass(slots=True)
class SpindleConfig:
    """
    Root configuration structure for Spindle storage.

    Attributes:
        storage: Filesystem paths for storage backends.
        extras: User-defined metadata dictionary.
        observability: Logging and event configuration.
        vector_store: Vector store creation preferences.
        graph_store: Graph database persistence settings.
        llm: Optional LLM configuration for authentication. See
            :class:`~spindle.llm_config.LLMConfig` for details.
    """

    storage: StoragePaths
    extras: Mapping[str, Any] = field(default_factory=dict)
    observability: ObservabilitySettings = field(default_factory=ObservabilitySettings)
    vector_store: VectorStoreSettings = field(default_factory=VectorStoreSettings)
    graph_store: GraphStoreSettings = field(default_factory=GraphStoreSettings)
    llm: "LLMConfig | None" = None

    def get_llm_config(self, auto_detect: bool = True) -> "LLMConfig | None":
        """
        Get LLM configuration, auto-detecting if not set.

        Returns the configured LLM config if available, otherwise attempts
        to auto-detect credentials from the environment if `auto_detect=True`.

        Args:
            auto_detect: Whether to auto-detect LLM credentials if not configured.
                Defaults to True.

        Returns:
            LLMConfig instance if available, None otherwise.

        Example::

            config = SpindleConfig.with_root("/path/to/storage")
            llm_config = config.get_llm_config()
            if llm_config:
                print(f"Using auth method: {llm_config.preferred_auth_method}")
        """
        if self.llm is not None:
            return self.llm

        if not auto_detect:
            return None

        try:
            from spindle.llm_config import detect_available_auth

            if detect_available_auth is not None:
                return detect_available_auth()
        except ImportError:
            pass

        return None

    def create_extractor(
        self,
        ontology: Any = None,
        **kwargs: Any,
    ) -> Any:
        """
        Create a SpindleExtractor using this config's LLM settings.

        Args:
            ontology: Ontology to use for extraction. Must be provided before
                extraction occurs (here or at extract time).
            **kwargs: Additional arguments passed to SpindleExtractor constructor.

        Returns:
            SpindleExtractor instance configured with this config's LLM settings.

        Raises:
            ImportError: If SpindleExtractor cannot be imported.
        """
        try:
            from spindle.extraction.extractor import SpindleExtractor
        except ImportError as exc:
            raise ImportError(
                "SpindleExtractor is not available. "
                "Ensure spindle.extraction.extractor is importable."
            ) from exc

        llm_config = self.get_llm_config()
        return SpindleExtractor(
            ontology=ontology,
            llm_config=llm_config,
            auto_detect_auth=llm_config is None,
            **kwargs,
        )

    @classmethod
    def with_root(
        cls,
        root: Path | str,
        *,
        extras: Mapping[str, Any] | None = None,
        observability: ObservabilitySettings | None = None,
        vector_store: VectorStoreSettings | None = None,
        graph_store: GraphStoreSettings | None = None,
        llm: "LLMConfig | None" = None,
        auto_detect_llm: bool = False,
    ) -> "SpindleConfig":
        """
        Create a SpindleConfig with storage paths derived from a root directory.

        All sub-stores are placed under ``root`` using the standard layout::

            <root>/
              kos/                  # KOS artifacts
              graphs/               # Kùzu graph databases
                spindle_graph/
                  graph.db
              vector_store/         # ChromaDB
              documents/            # Docling JSON output
              logs/
              sqlite/
                provenance.db
                catalog.db
                rejections.db
                events.db

        Args:
            root: Root directory for all storage backends.
            extras: Optional user-defined metadata.
            observability: Optional observability settings.
            vector_store: Optional vector store settings.
            graph_store: Optional graph store settings.
            llm: Optional LLM configuration. See :class:`~spindle.llm_config.LLMConfig`.
            auto_detect_llm: If True and llm is None, automatically detect LLM
                credentials from the environment. Defaults to False.

        Returns:
            Configured SpindleConfig instance.
        """
        root_path = _ensure_path(root)
        sqlite_dir = root_path / "sqlite"
        storage = StoragePaths(
            root=root_path,
            kos_dir=root_path / "kos",
            graphs_dir=root_path / "graphs",
            vector_store_dir=root_path / "vector_store",
            graph_store_path=root_path / "graphs" / "spindle_graph" / "graph.db",
            document_store_dir=root_path / "documents",
            log_dir=root_path / "logs",
            provenance_db=sqlite_dir / "provenance.db",
            catalog_db=sqlite_dir / "catalog.db",
            rejection_db=sqlite_dir / "rejections.db",
            event_log_db=sqlite_dir / "events.db",
        )
        resolved_extras: Mapping[str, Any]
        if extras:
            resolved_extras = MappingProxyType(dict(extras))
        else:
            resolved_extras = MappingProxyType({})

        # Auto-detect LLM config if requested
        resolved_llm = llm
        if resolved_llm is None and auto_detect_llm:
            try:
                from spindle.llm_config import detect_available_auth

                if detect_available_auth is not None:
                    resolved_llm = detect_available_auth()
            except ImportError:
                pass

        return cls(
            storage=storage,
            extras=resolved_extras,
            observability=observability or ObservabilitySettings(),
            vector_store=vector_store or VectorStoreSettings(),
            graph_store=graph_store or GraphStoreSettings(),
            llm=resolved_llm,
        )

    @classmethod
    def with_auto_detected_llm(
        cls,
        root: Path | str,
        *,
        extras: Mapping[str, Any] | None = None,
        observability: ObservabilitySettings | None = None,
        vector_store: VectorStoreSettings | None = None,
        graph_store: GraphStoreSettings | None = None,
    ) -> "SpindleConfig":
        """
        Create a SpindleConfig with auto-detected LLM configuration.

        This is a convenience method that calls :meth:`with_root` with
        `auto_detect_llm=True`. LLM credentials will be automatically detected
        from the environment if available.

        Args:
            root: Root directory for all storage backends.
            extras: Optional user-defined metadata.
            observability: Optional observability settings.
            vector_store: Optional vector store settings.
            graph_store: Optional graph store settings.

        Returns:
            Configured SpindleConfig instance with auto-detected LLM config.
        """
        return cls.with_root(
            root,
            extras=extras,
            observability=observability,
            vector_store=vector_store,
            graph_store=graph_store,
            auto_detect_llm=True,
        )


def default_config(root: Path | None = None) -> SpindleConfig:
    """Return a default configuration using the auto-detected stores root.

    When ``root`` is not provided the stores root is resolved by
    :py:func:`find_stores_root` — ``<git_root>/stores`` when running inside a
    git repository, ``<cwd>/stores`` otherwise.

    Args:
        root: Optional explicit root directory.  Overrides auto-detection.

    Returns:
        :class:`SpindleConfig` with all paths under ``root``.
    """
    if root is None:
        root = find_stores_root()
    return SpindleConfig.with_root(root)


def load_config_from_file(path: Path | str) -> SpindleConfig:
    """
    Execute a user provided config module and return ``SpindleConfig``.

    The target file must define a global named ``SPINDLE_CONFIG`` that is an
    instance of :class:`SpindleConfig`.
    """
    path = Path(path).expanduser()
    if not path.exists():
        raise ConfigurationError(f"Configuration file not found: {path}")

    namespace: MutableMapping[str, Any] = {}
    code = path.read_text()
    compiled = compile(code, str(path), "exec")
    exec(compiled, namespace, namespace)  # noqa: S102 (exec used for config loading)

    if CONFIG_SYMBOL_NAME not in namespace:
        raise ConfigurationError(
            f"Configuration file {path} must define `{CONFIG_SYMBOL_NAME}`"
        )

    config_obj = namespace[CONFIG_SYMBOL_NAME]
    if not isinstance(config_obj, SpindleConfig):
        raise ConfigurationError(
            f"{CONFIG_SYMBOL_NAME} in {path} must be a SpindleConfig, "
            f"got {type(config_obj)!r}"
        )

    return config_obj


__all__ = [
    "CONFIG_SYMBOL_NAME",
    "ConfigurationError",
    "DEFAULT_STORAGE_ROOT_NAME",
    "GraphStoreSettings",
    "ObservabilitySettings",
    "SpindleConfig",
    "StoragePaths",
    "VectorStoreSettings",
    "default_config",
    "find_stores_root",
    "load_config_from_file",
]
