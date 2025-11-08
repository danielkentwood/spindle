"""Observer that forwards ingestion events to the observability recorder."""

from __future__ import annotations

from spindle.ingestion.types import IngestionEvent
from spindle.observability import get_event_recorder

RECORDER = get_event_recorder("ingestion.pipeline")


def observability_observer(event: IngestionEvent) -> None:
    """Bridge ingestion pipeline events into the observability recorder."""

    RECORDER.record(
        name=event.name,
        payload=dict(event.payload),
        timestamp=event.timestamp,
    )


