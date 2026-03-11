## Configuration

Spindle supports two config styles:

- programmatic config with `SpindleConfig`
- Hydra config groups under `spindle/conf`

## Programmatic config (`SpindleConfig`)

Use this when embedding Spindle in Python applications.

```python
from pathlib import Path
from spindle.configuration import SpindleConfig

cfg = SpindleConfig.with_root(Path.cwd() / "spindle_storage")
cfg.storage.ensure_directories()
```

Important fields:

- `storage`: filesystem locations for graph/vector/doc/log stores.
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

`get_pipeline_definition(...)` returns a list of stage definitions:

```python
stages = get_pipeline_definition(cfg=my_cfg, kos_dir="kos", ontology=my_ontology)
```

Returned type is `list[StageDef]`, ready for spindle-eval style executors.
