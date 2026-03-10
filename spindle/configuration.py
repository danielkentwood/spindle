"""
Unified configuration primitives for Spindle storage.

The `SpindleConfig` dataclass is the single entry point that downstream
components use to determine storage locations (vector store, graph DB,
document persistence, logging).

LLM configuration can be integrated via the optional `llm` field, which accepts
an :class:`~spindle.llm_config.LLMConfig` instance. See :mod:`spindle.llm_config`
for details on LLM authentication and credential management.

Example usage::

    from pathlib import Path
    from spindle.configuration import SpindleConfig

    config = SpindleConfig.with_root(Path.cwd() / "spindle_storage")
    print(config.storage.vector_store_dir)

The configuration loader can execute a user supplied `config.py` file::

    from spindle.configuration import load_config_from_file

    config = load_config_from_file("/path/to/config.py")

The file must define a variable named ``SPINDLE_CONFIG`` that is an instance
of :class:`SpindleConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, MutableMapping, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from spindle.llm_config import LLMConfig


DEFAULT_STORAGE_ROOT_NAME = "spindle_storage"
CONFIG_SYMBOL_NAME = "SPINDLE_CONFIG"


class ConfigurationError(RuntimeError):
    """Raised when loading a configuration file fails."""


def _ensure_path(path: Path | str) -> Path:
    result = Path(path).expanduser()
    if not result.is_absolute():
        result = result.resolve()
    return result


@dataclass(slots=True)
class StoragePaths:
    """Filesystem locations used by Spindle."""

    root: Path
    vector_store_dir: Path
    graph_store_path: Path
    document_store_dir: Path
    log_dir: Path

    def ensure_directories(self) -> None:
        """Create directories represented by this configuration."""
        dirs = {
            self.root,
            self.vector_store_dir,
            self.document_store_dir,
            self.log_dir,
        }
        dirs.add(Path(self.graph_store_path).parent)
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
        storage = StoragePaths(
            root=root_path,
            vector_store_dir=root_path / "vector_store",
            graph_store_path=root_path / "graph" / "graph.db",
            document_store_dir=root_path / "documents",
            log_dir=root_path / "logs",
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
    """Return a default configuration rooted at the provided directory."""
    if root is None:
        root = Path.cwd() / DEFAULT_STORAGE_ROOT_NAME
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
    "load_config_from_file",
]

