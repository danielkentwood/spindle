# Spindle Configuration & Storage Layout

Spindle centralizes its runtime configuration in a Python module that exports a
`SpindleConfig` object. The config controls where storage backends live,
which template directories to load, and any user-defined metadata you want to
pass through the system.

This document covers the configuration entry points and how they influence the
storage layout across the project.

## Generating a Config File

Use the CLI to scaffold a starter `config.py`:

```bash
uv run spindle-ingest config init
```

The command writes `config.py` in the current working directory and initializes
the default storage tree under `./spindle_storage/`. Provide a custom root or
output file if needed:

```bash
uv run spindle-ingest config init my-spindle-config.py --root ~/data/spindle --force
```

The generator always creates the full directory structure (vector store,
catalog, graph DB, documents, logs, templates) so backends are ready to use.

## Anatomy of `SpindleConfig`

Generated `config.py` files export a module‐level constant named
`SPINDLE_CONFIG`:

```python
from pathlib import Path
from spindle.configuration import SpindleConfig

storage_root = Path("/absolute/path/to/spindle_storage")

template_paths: tuple[Path, ...] = (
    storage_root / "templates",
)

extras: dict[str, object] = {}

SPINDLE_CONFIG = SpindleConfig.with_root(
    storage_root,
    template_paths=template_paths,
    extras=extras,
)
```

`SpindleConfig` contains:

- `storage`: a `StoragePaths` dataclass with resolved directories for the vector
  store, graph database, document cache, log directory, catalog database, and
  optional template root. Call `storage.ensure_directories()` if you create a
  config manually.
- `templates`: a `TemplateSettings` object listing template search paths.
  Ingestion merges these with the built-in defaults.
- `extras`: an immutable mapping for your own metadata (kept available for user
  hooks or downstream tools).

The helper `SpindleConfig.with_root` derives all child paths relative to the
root, ensuring a self-contained storage bundle.

## Loading Configuration

The ingestion pipeline accepts an explicit config file via CLI:

```bash
uv run spindle-ingest --config /path/to/config.py docs/*
```

Code paths can load the same file using `spindle.configuration`:

```python
from spindle.configuration import load_config_from_file

config = load_config_from_file("/path/to/config.py")
```

The loader executes the module and validates that `SPINDLE_CONFIG` is a
`SpindleConfig` instance.

## Storage Backends and Defaults

Each storage client respects the values in `SpindleConfig`:

- **Vector store** (`ChromaVectorStore`) uses `storage.vector_store_dir` when no
  explicit `persist_directory` is provided.
- **Graph store** (`GraphStore`) stores the default database at
  `storage.graph_store_path`.
- **Document catalog** (SQLite) persists to `storage.catalog_path`.
- **Logs** and **documents** utilities rely on `storage.log_dir` and
  `storage.document_store_dir`.

Passing `config=...` to these classes ensures every backend points to the same
root. When you omit the parameter they continue to create sensible defaults,
but the generated config keeps everything together.

## Template Discovery

Ingestion merges templates from:

1. Built-in defaults
2. `SpindleConfig.templates.search_paths`
3. Any explicit `--templates` CLI overrides

This allows you to keep reusable templates under the storage root (or anywhere
else) and still layer additional directories per run.

## Versioning Configurations

Treat generated `config.py` files as environment-specific artifacts. They can
live next to your project or in a separate infrastructure repository—just point
CLI commands or APIs at the correct path. Because the file is plain Python, you
can add helper functions, environment-specific logic, or comments as needed.


