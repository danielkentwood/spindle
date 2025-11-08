# Observability Logging

Spindle emits structured `ServiceEvent` records for all major services. These
events provide a uniform way to capture lifecycle transitions, metrics, and
errors across ingestion, extraction, graph persistence, and vector storage.

## Event fundamentals

- Events use the `ServiceEvent` dataclass (`timestamp`, `service`, `name`,
  `payload`).
- `EventRecorder` instances manage observers and fire events. Recorders can be
  nested (`recorder.scoped("ingestion.pipeline")`) to form hierarchical service
  namespaces.
- The global recorder is accessible with `get_event_recorder()`. Module-level
  helpers use scoped recorders (for example, `"ingestion.service"` or
  `"vector_store"`).

```python
from spindle.observability import get_event_recorder

recorder = get_event_recorder("my.custom.service")
recorder.record("stage.start", {"step": "load"})
```

## Persistence and replay

The `EventLogStore` persists events to the existing SQLite persistence layer:

```python
from spindle.observability import attach_persistent_observer, get_event_recorder
from spindle.observability.storage import EventLogStore

store = EventLogStore("sqlite:///spindle_events.db")
detach = attach_persistent_observer(get_event_recorder(), store)

# events are written automatically
...

detach()  # remove the observer when finished
```

Stored events can be queried or replayed into a recorder:

```python
events = store.fetch_events(service="ingestion.pipeline", limit=100)
store.replay_to(get_event_recorder("analysis"))
```

## CLI integration

The ingestion CLI records start/completion events and can persist them with a
single flag:

```bash
uv run python -m spindle.ingestion.cli docs/ --event-log sqlite:///spindle_events.db
```

When `--event-log` is supplied, the CLI attaches a persistent observer to the
global recorder for the duration of the run.

## Service coverage

The following components emit structured service events:

- Ingestion service orchestration and pipeline observers
- Extractor and ontology recommender flows
- Graph store lifecycle, mutations, and queries
- Vector store initialization and CRUD operations
- Ingestion CLI entry point

Each component uses a scoped recorder so event consumers can filter by service
prefix (e.g., `ingestion.pipeline`, `graph_store`, `vector_store`, or
`ontology.recommender`).


