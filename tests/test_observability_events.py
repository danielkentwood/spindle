from __future__ import annotations

from datetime import datetime, timedelta

from spindle.observability import (
    EventRecorder,
    ServiceEvent,
    attach_persistent_observer,
    get_event_recorder,
    reset_event_recorder,
)
from spindle.observability.storage import EventLogStore


def test_event_recorder_scoped_paths_propagate_events() -> None:
    events: list[ServiceEvent] = []
    recorder = EventRecorder()
    recorder.register(events.append)
    scoped = recorder.scoped("ingestion.pipeline")

    event = scoped.record("stage", {"stage": "load"})

    assert event.service == "ingestion.pipeline"
    assert events and events[0].name == "stage"
    assert events[0].payload["stage"] == "load"


def test_event_recorder_namespace_stack() -> None:
    root = EventRecorder()
    scoped = root.scoped("ingestion")
    nested = scoped.scoped("pipeline")

    event = nested.record("complete", {})

    assert event.service == "ingestion.pipeline"


def test_get_event_recorder_returns_global_singleton() -> None:
    reset_event_recorder()
    recorder = get_event_recorder()
    other_reference = get_event_recorder()

    assert recorder is other_reference


def test_event_log_store_persist_and_fetch(tmp_path) -> None:
    db_path = tmp_path / "events.db"
    store = EventLogStore(f"sqlite:///{db_path}")
    now = datetime.utcnow()
    event = ServiceEvent(
        timestamp=now,
        service="ingestion.pipeline",
        name="stage_complete",
        payload={"duration_ms": 12.3},
    )

    store.persist_events([event])
    fetched = store.fetch_events()

    assert len(fetched) == 1
    fetched_event = fetched[0]
    assert fetched_event.name == event.name
    assert fetched_event.payload["duration_ms"] == 12.3


def test_event_log_store_filters(tmp_path) -> None:
    db_path = tmp_path / "events.db"
    store = EventLogStore(f"sqlite:///{db_path}")
    now = datetime.utcnow()
    earlier = now - timedelta(hours=1)
    later = now + timedelta(hours=1)
    service = "ingestion.pipeline"

    store.persist_events(
        [
            ServiceEvent(timestamp=earlier, service=service, name="stage", payload={}),
            ServiceEvent(timestamp=now, service=service, name="stage", payload={}),
            ServiceEvent(timestamp=later, service="other", name="stage", payload={}),
        ]
    )

    filtered = store.fetch_events(service=service, since=now)
    assert len(filtered) == 1
    assert filtered[0].timestamp >= now


def test_attach_persistent_observer_records_events(tmp_path) -> None:
    db_path = tmp_path / "events.db"
    store = EventLogStore(f"sqlite:///{db_path}")
    recorder = EventRecorder()

    remove = attach_persistent_observer(recorder, store)
    recorder.record("test", {"value": 1})
    remove()
    recorder.record("after_remove", {})

    events = store.fetch_events()
    assert len(events) == 1
    assert events[0].name == "test"

