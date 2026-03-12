## Configuration

Spindle supports two config styles:

- programmatic config with `SpindleConfig`
- Hydra config groups under `spindle/conf`

## Default Storage Layout

All persisted artifacts default to a single `stores/` directory.  The
location of that directory is auto-detected:

- **Inside a git repository** (the normal case) → `<git_root>/stores/`
- **Outside a git repository** → `<cwd>/stores/`

```
stores/
  kos/                   # KOS artifacts (kos.ttls, ontology.owl, shapes.ttl, staging/)
  graphs/                # Kùzu graph databases
    spindle_graph/
      graph.db           # default graph
  vector_store/          # ChromaDB persistent client
  documents/             # Docling JSON output
  logs/
  sqlite/
    provenance.db
    catalog.db
    rejections.db
    events.db
```

`find_stores_root()` returns the stores root without creating it.
`StoragePaths.ensure_directories()` (called by `GraphStore` and
`ChromaVectorStore` when they receive a `SpindleConfig`) creates everything
on first use.

## Programmatic config (`SpindleConfig`)

Use this when embedding Spindle in Python applications.

```python
from spindle.configuration import default_config, SpindleConfig

# Auto-detected stores root (git root / stores, or CWD / stores)
cfg = default_config()
cfg.storage.ensure_directories()

# Or specify an explicit root
cfg = SpindleConfig.with_root("/path/to/my/stores")
cfg.storage.ensure_directories()
```

### `StoragePaths` fields

| Field | Default (relative to root) |
|---|---|
| `root` | the root itself |
| `kos_dir` | `kos/` |
| `graphs_dir` | `graphs/` |
| `vector_store_dir` | `vector_store/` |
| `graph_store_path` | `graphs/spindle_graph/graph.db` |
| `document_store_dir` | `documents/` |
| `log_dir` | `logs/` |
| `provenance_db` | `sqlite/provenance.db` |
| `catalog_db` | `sqlite/catalog.db` |
| `rejection_db` | `sqlite/rejections.db` |
| `event_log_db` | `sqlite/events.db` |

### Other config fields

- `vector_store`: embedding model and fallback behavior.
- `graph_store`: graph path override and snapshot behavior.
- `observability`: event logging controls.
- `llm`: optional explicit LLM config.

## Loading config from file

```python
from spindle.configuration import load_config_from_file

cfg = load_config_from_file("config.py")
```

`config.py` must define `SPINDLE_CONFIG` as a `SpindleConfig` instance.

## Hydra groups

Package config groups are under:

- `spindle/conf/preprocessing`
- `spindle/conf/kos_extraction`
- `spindle/conf/ontology_synthesis`
- `spindle/conf/retrieval`
- `spindle/conf/generation`

Hydra plugin registration is wired through the package entry point in `pyproject.toml`.

## Pipeline factory contract

`get_pipeline_definition(...)` returns a list of stage definitions.  When
`kos_dir` is omitted it defaults to the auto-detected `<stores_root>/kos`:

```python
from spindle import get_pipeline_definition

# Uses <stores_root>/kos automatically
stages = get_pipeline_definition(cfg=my_cfg, ontology=my_ontology)

# Or pass an explicit kos_dir
stages = get_pipeline_definition(cfg=my_cfg, kos_dir="/custom/kos", ontology=my_ontology)
```

Returned type is `list[StageDef]`, ready for spindle-eval style executors.
