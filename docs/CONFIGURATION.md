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
from spindle.configuration import (
    GraphStoreSettings,
    IngestionSettings,
    ObservabilitySettings,
    SpindleConfig,
    VectorStoreSettings,
)
# Optional: from spindle.llm_config import LLMConfig

storage_root = Path("/absolute/path/to/spindle_storage")

template_paths: tuple[Path, ...] = (
    storage_root / "templates",
)

extras: dict[str, object] = {}

observability = ObservabilitySettings(
    event_log_url="sqlite:///spindle_events.db",
    log_level="INFO",
    enable_pipeline_events=True,
)

ingestion = IngestionSettings(
    catalog_url="sqlite:///custom_catalog.db",
    vector_store_uri=str(storage_root / "vector_store"),
    cache_dir=storage_root / "cache",
    allow_network_requests=True,
    recursive=False,
)

vector_store = VectorStoreSettings(
    collection_name="team_embeddings",
    embedding_model="sentence-transformers/all-mpnet-base-v2",
    use_api_fallback=False,
    prefer_local_embeddings=True,
)

graph_store = GraphStoreSettings(
    db_path_override=storage_root / "graph" / "company_graph.db",
    auto_snapshot=True,
    snapshot_dir=storage_root / "graph" / "snapshots",
    embedding_dimensions=128,
    auto_compute_embeddings=False,
)

llm = None  # Replace with LLMConfig() if you want to bake in provider credentials

SPINDLE_CONFIG = SpindleConfig.with_root(
    storage_root,
    template_paths=template_paths,
    extras=extras,
    observability=observability,
    ingestion=ingestion,
    vector_store=vector_store,
    graph_store=graph_store,
    llm=llm,
)
```

- `storage`: a `StoragePaths` dataclass with resolved directories for the vector
  store, graph database, document cache, log directory, catalog database, and the
  optional template root. Call `storage.ensure_directories()` if you create a
  config manually.
- `templates`: a `TemplateSettings` object listing template search paths. The
  ingestion runtime merges these with the built-in defaults.
- `observability`: settings for event logging and log levels. Includes
  `event_log_url`, `log_level`, and `enable_pipeline_events`. See
  [`docs/OBSERVABILITY.md`](OBSERVABILITY.md) for the downstream consumers.
- `ingestion`: defaults that influence CLI runs and programmatic ingestion
  (catalog URL, vector store URI, cache directory, recursion, network access).
- `vector_store`: preferences for `ChromaVectorStore` creation such as collection
  name, embedding model, whether to fall back to API embeddings, and preference
  for local vs API embeddings.
- `graph_store`: defaults used by `GraphStore`, including optional overrides for
  the database path, snapshot locations, embedding dimensions, and auto-compute
  settings.
- `llm`: optional `LLMConfig` instance with authentication and provider
  priorities for `SpindleExtractor` and the ontology recommender. Can also use
  `config.get_llm_config()` or `config.create_extractor()` / `config.create_recommender()`
  to automatically use the configured LLM settings.
- `extras`: an immutable mapping for your own metadata (kept available for user
  hooks or downstream tools).

The helper `SpindleConfig.with_root` derives all child paths relative to the
root, ensuring a self-contained storage bundle.

`spindle-ingest` honours these defaults automatically: if you omit `--catalog`
or `--vector-store`, the CLI falls back to `ingestion.catalog_url` and
`ingestion.vector_store_uri`. When no event log is supplied, the observability
settings take over, so a single config change reroutes telemetry for both CLI
and programmatic runs.

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

## Using `SPINDLE_CONFIG` in Application Code

Once `config.py` exists you can import the constant and reuse the resolved
settings anywhere in your project:

```python
from pathlib import Path

from my_spindle_config import SPINDLE_CONFIG
from spindle.ingestion.service import build_config, run_ingestion

# Ensure the storage root exists before you start writing data.
SPINDLE_CONFIG.storage.ensure_directories()

ingestion_config = build_config(
    spindle_config=SPINDLE_CONFIG,
    template_paths=[Path("custom_templates")],
)

run_ingestion([Path("docs")], ingestion_config)
```

- `build_config(spindle_config=...)` merges the ingestion defaults from the
  config with any ad-hoc overrides, so CLI behavior and programmatic runs stay in
  sync.
- `SPINDLE_CONFIG.storage` exposes fully resolved paths (`catalog_url`,
  `vector_store_dir`, `log_dir`, etc.) that you can pass directly to custom
  utilities or integrations.
- Higher-level clients like `GraphStore` accept the config via `GraphStore(config=SPINDLE_CONFIG)`
  to reuse snapshot and database settings.
- Use `SPINDLE_CONFIG.create_extractor()` or `SPINDLE_CONFIG.create_recommender()` to
  automatically create extractors/recommenders with the configured LLM settings.
- Extras such as `analytics_database` remain available through
  `dict(SPINDLE_CONFIG.extras)`, letting you surface user-defined metadata in
  downstream pipelines.

## Storage Backends and Defaults

Each storage client respects the values in `SpindleConfig`:

- **Vector store** (`ChromaVectorStore`) uses `ingestion.vector_store_uri` when
  set, otherwise falls back to `storage.vector_store_dir`.
- **Graph store** (`GraphStore`) prefers `graph_store.db_path_override` or, when
  unset, `storage.graph_store_path`.
- **Document catalog** (SQLite) persists to `storage.catalog_path`.
- **Logs** and **documents** utilities rely on `storage.log_dir` and
  `storage.document_store_dir`.

Passing `config=...` to these classes ensures every backend points to the same
root. When you omit the parameter they continue to create sensible defaults,
but the generated config keeps everything together.

## Analytics Storage

The ingestion analytics layer persists structured observations to SQLite. By
default Spindle writes analytics to `storage.log_dir / "analytics.db"` using the
same root as other persistence backends. You can override the destination by
setting an `analytics_database` entry in `SpindleConfig.extras`:

```python
observability = ObservabilitySettings(
    event_log_url="sqlite:///spindle_events.db",
)

extras = {"analytics_database": "sqlite:////absolute/path/to/analytics.db"}

SPINDLE_CONFIG = SpindleConfig.with_root(
    storage_root,
    template_paths=template_paths,
    extras=extras,
    observability=observability,
)
```

When present, the ingestion service writes each document observation into the
specified database and mirrors associated `ServiceEvent` records. Review
`docs/INGESTION_ANALYTICS.md` for the taxonomy of captured fields.

The generated SQLite file powers the CLI dashboard:

```bash
uv run spindle-dashboard --database /absolute/path/to/analytics.db --open
```

Because the dashboard is a standalone HTML export, you can point other metadata
pipelines at the same database without affecting ingestion throughput.

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
