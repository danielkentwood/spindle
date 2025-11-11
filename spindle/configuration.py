"""
Unified configuration primitives for Spindle storage and templates.

The `SpindleConfig` dataclass is the single entry point that downstream
components use to determine storage locations (vector store, graph DB,
document persistence, logging) and template search paths.

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
import textwrap
from types import MappingProxyType
from typing import Any, Iterable, Mapping, MutableMapping, Sequence


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
    catalog_path: Path
    template_root: Path | None = None

    def ensure_directories(self) -> None:
        """Create directories represented by this configuration."""
        dirs = {
            self.root,
            self.vector_store_dir,
            self.document_store_dir,
            self.log_dir,
        }
        if self.template_root:
            dirs.add(self.template_root)
        dirs.add(Path(self.graph_store_path).parent)
        dirs.add(self.catalog_path.parent)
        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def catalog_url(self) -> str:
        """Return the SQLAlchemy URL for the ingestion catalog."""
        return f"sqlite:///{self.catalog_path}"


@dataclass(slots=True)
class TemplateSettings:
    """Template registry related configuration."""

    search_paths: tuple[Path, ...] = field(default_factory=tuple)

    @classmethod
    def from_iterable(cls, values: Iterable[Path | str] | None) -> "TemplateSettings":
        if not values:
            return cls()
        return cls(
            search_paths=tuple(
                _ensure_path(value) for value in values
            )
        )


@dataclass(slots=True)
class SpindleConfig:
    """Root configuration structure for Spindle storage and templates."""

    storage: StoragePaths
    templates: TemplateSettings = field(default_factory=TemplateSettings)
    extras: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def with_root(
        cls,
        root: Path | str,
        *,
        template_paths: Sequence[Path | str] | None = None,
        extras: Mapping[str, Any] | None = None,
    ) -> "SpindleConfig":
        root_path = _ensure_path(root)
        storage = StoragePaths(
            root=root_path,
            vector_store_dir=root_path / "vector_store",
            graph_store_path=root_path / "graph" / "graph.db",
            document_store_dir=root_path / "documents",
            log_dir=root_path / "logs",
            catalog_path=root_path / "catalog" / "ingestion.db",
            template_root=(root_path / "templates"),
        )
        templates = TemplateSettings.from_iterable(template_paths)
        if not templates.search_paths and storage.template_root:
            templates = TemplateSettings.from_iterable([storage.template_root])
        resolved_extras: Mapping[str, Any]
        if extras:
            resolved_extras = MappingProxyType(dict(extras))
        else:
            resolved_extras = MappingProxyType({})
        return cls(storage=storage, templates=templates, extras=resolved_extras)


def default_config(root: Path | None = None) -> SpindleConfig:
    """Return a default configuration rooted at the provided directory."""
    if root is None:
        root = Path.cwd() / DEFAULT_STORAGE_ROOT_NAME
    return SpindleConfig.with_root(root)


def render_default_config(root: Path | None = None) -> str:
    """
    Render the canonical ``config.py`` contents for a user workspace.

    Parameters
    ----------
    root:
        Optional storage root. Defaults to ``<cwd>/spindle_storage`` when not
        supplied.

    Returns
    -------
    str
        The string content for a `config.py` file.
    """
    config = default_config(root)
    template_entries: list[str] = [
        f"    Path({path!r})," for path in config.templates.search_paths
    ]
    if not template_entries:
        template_entries = [
            "    # Path(\"/path/to/templates\"),",
        ]
    template_block = "\n".join(template_entries)
    extras_repr = ",\n        ".join(
        f"{key!r}: {value!r}" for key, value in config.extras.items()
    )

    extras_block = (
        textwrap.dedent(
            """
            extras = {{
                {extras}
            }}
            """
        ).format(extras=extras_repr)
        if extras_repr
        else "extras: dict[str, Any] = {}"
    )

    return textwrap.dedent(
        f"""\
        from pathlib import Path
        from typing import Any

        from spindle.configuration import SpindleConfig


        storage_root = Path({config.storage.root!r})

        {extras_block}

        # Template search paths allow you to reference custom ingestion templates.
        template_paths: tuple[Path, ...] = (
        {template_block}
        )

        SPINDLE_CONFIG = SpindleConfig.with_root(
            storage_root,
            template_paths=template_paths,
            extras=extras,
        )
        """
    )


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
    "SpindleConfig",
    "StoragePaths",
    "TemplateSettings",
    "default_config",
    "load_config_from_file",
    "render_default_config",
]

