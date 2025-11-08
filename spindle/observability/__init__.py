"""Observability primitives for Spindle services."""

from spindle.observability.events import (
    EventObserver,
    EventRecorder,
    ServiceEvent,
    get_event_recorder,
    reset_event_recorder,
    set_event_recorder,
)
from spindle.observability.storage import (
    EventLogStore,
    attach_persistent_observer,
)

__all__ = [
    "EventObserver",
    "EventRecorder",
    "ServiceEvent",
    "EventLogStore",
    "attach_persistent_observer",
    "get_event_recorder",
    "reset_event_recorder",
    "set_event_recorder",
]


