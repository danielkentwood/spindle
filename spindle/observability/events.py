"""Event primitives and dispatcher for Spindle observability."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from threading import RLock
from typing import Any, Dict, Iterator, Optional, Tuple

Metadata = Dict[str, Any]
EventObserver = Callable[["ServiceEvent"], None]


@dataclass(slots=True, frozen=True)
class ServiceEvent:
    """Event emitted from Spindle services for observability."""

    timestamp: datetime
    service: str
    name: str
    payload: Metadata = field(default_factory=dict)


class EventRecorder:
    """Central dispatcher for service events."""

    __slots__ = ("_service_path", "_root", "_observers", "_lock")

    def __init__(
        self,
        service: Sequence[str] | None = None,
        *,
        parent: "EventRecorder" | None = None,
    ) -> None:
        if parent is None:
            self._root = self
            self._observers: list[EventObserver] = []
            self._service_path: Tuple[str, ...] = tuple(service or ())
            self._lock = RLock()
        else:
            self._root = parent._root
            self._observers = parent._root._observers
            parent_path = parent._service_path
            additional = tuple(self._normalize_service_path(service))
            self._service_path = parent_path + additional
            self._lock = parent._root._lock

    @staticmethod
    def _normalize_service_path(
        service: Sequence[str] | str | None,
    ) -> Tuple[str, ...]:
        if service is None:
            return ()
        if isinstance(service, str):
            return tuple(part for part in service.split(".") if part)
        return tuple(part for part in service if part)

    @property
    def service(self) -> str:
        """Return the service namespace for this recorder."""

        return ".".join(self._service_path)

    def scoped(
        self,
        service: Sequence[str] | str,
    ) -> "EventRecorder":
        """Return a child recorder scoped to the given service."""

        return EventRecorder(service=self._normalize_service_path(service), parent=self)

    def register(self, observer: EventObserver) -> None:
        """Register an observer to receive events."""

        with self._lock:
            if observer not in self._root._observers:
                self._root._observers.append(observer)

    def unregister(self, observer: EventObserver) -> None:
        """Remove a previously registered observer."""

        with self._lock:
            try:
                self._root._observers.remove(observer)
            except ValueError:
                pass

    def clear_observers(self) -> None:
        """Remove all observers from the recorder."""

        with self._lock:
            self._root._observers.clear()

    @contextmanager
    def temporary_observer(self, observer: EventObserver) -> Iterator[None]:
        """Register an observer for the duration of the context manager."""

        self.register(observer)
        try:
            yield
        finally:
            self.unregister(observer)

    def record(
        self,
        name: str,
        payload: Metadata | None = None,
        *,
        service: Sequence[str] | str | None = None,
        timestamp: Optional[datetime] = None,
    ) -> ServiceEvent:
        """Create an event and notify observers."""

        full_service = self._compose_service(service)
        event = ServiceEvent(
            timestamp=timestamp or datetime.utcnow(),
            service=full_service,
            name=name,
            payload=dict(payload or {}),
        )
        self._dispatch(event)
        return event

    def emit(self, event: ServiceEvent) -> None:
        """Forward an existing event to observers."""

        if not event.service:
            scoped_event = ServiceEvent(
                timestamp=event.timestamp,
                service=self.service,
                name=event.name,
                payload=dict(event.payload),
            )
            self._dispatch(scoped_event)
        else:
            self._dispatch(event)

    def _dispatch(self, event: ServiceEvent) -> None:
        for observer in self._snapshot_observers():
            try:
                observer(event)
            except Exception:
                # Observers are best-effort; errors are swallowed to avoid cascading failures.
                # Logging will be attached via instrumentation when observers are configured.
                continue

    def _snapshot_observers(self) -> Tuple[EventObserver, ...]:
        with self._lock:
            return tuple(self._root._observers)

    def _compose_service(
        self,
        service: Sequence[str] | str | None,
    ) -> str:
        parts = list(self._service_path)
        parts.extend(self._normalize_service_path(service))
        return ".".join(parts)


_GLOBAL_RECORDER = EventRecorder()


def get_event_recorder(service: Sequence[str] | str | None = None) -> EventRecorder:
    """Return the global event recorder or a scoped variant."""

    if service is None:
        return _GLOBAL_RECORDER
    return _GLOBAL_RECORDER.scoped(service)


def set_event_recorder(recorder: EventRecorder) -> None:
    """Replace the global event recorder."""

    global _GLOBAL_RECORDER
    _GLOBAL_RECORDER = recorder


def reset_event_recorder() -> None:
    """Reset the global recorder to a clean instance."""

    set_event_recorder(EventRecorder())


