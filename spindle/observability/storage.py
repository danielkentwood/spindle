"""Persistence helpers for service events."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import JSON, DateTime, Integer, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from spindle.observability.events import EventObserver, EventRecorder, ServiceEvent


class Base(DeclarativeBase):
    """Declarative base for observability tables."""


class ServiceEventRow(Base):
    """SQLAlchemy row mapped to stored service events."""

    __tablename__ = "service_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=False), index=True)
    service: Mapped[str] = mapped_column(String, index=True)
    name: Mapped[str] = mapped_column(String, index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class EventLogStore:
    """SQLite-backed persistence for service events."""

    def __init__(self, database_url: str) -> None:
        self._engine = create_engine(database_url, future=True)
        self._session_factory = sessionmaker(self._engine, expire_on_commit=False)
        Base.metadata.create_all(self._engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def persist_events(self, events: Sequence[ServiceEvent]) -> None:
        """Persist a batch of events."""

        if not events:
            return
        with self.session() as session:
            session.add_all(self._to_rows(events))

    def replay_to(self, recorder: EventRecorder, *, service: str | None = None) -> None:
        """Replay stored events into a recorder."""

        for event in self.fetch_events(service=service):
            recorder.emit(event)

    def fetch_events(
        self,
        *,
        service: str | None = None,
        name: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int | None = None,
    ) -> list[ServiceEvent]:
        """Retrieve events using simple filtering criteria."""

        stmt = select(ServiceEventRow).order_by(ServiceEventRow.timestamp.asc())
        if service:
            stmt = stmt.where(ServiceEventRow.service == service)
        if name:
            stmt = stmt.where(ServiceEventRow.name == name)
        if since:
            stmt = stmt.where(ServiceEventRow.timestamp >= since)
        if until:
            stmt = stmt.where(ServiceEventRow.timestamp <= until)
        if limit:
            stmt = stmt.limit(limit)

        with self.session() as session:
            rows = session.execute(stmt).scalars().all()
        return [self._from_row(row) for row in rows]

    def create_persistent_observer(self) -> EventObserver:
        """Return an observer that writes each event to the store."""

        def _observer(event: ServiceEvent) -> None:
            try:
                self.persist_events([event])
            except Exception:
                # Persistence should not break application flow.
                pass

        return _observer

    @staticmethod
    def _to_rows(events: Iterable[ServiceEvent]) -> list[ServiceEventRow]:
        return [
            ServiceEventRow(
                timestamp=event.timestamp,
                service=event.service,
                name=event.name,
                payload=dict(event.payload),
            )
            for event in events
        ]

    @staticmethod
    def _from_row(row: ServiceEventRow) -> ServiceEvent:
        return ServiceEvent(
            timestamp=row.timestamp,
            service=row.service,
            name=row.name,
            payload=dict(row.payload or {}),
        )


def attach_persistent_observer(
    recorder: EventRecorder,
    store: EventLogStore,
) -> Callable[[], None]:
    """Attach a persistent observer to a recorder and return a remover callback."""

    observer = store.create_persistent_observer()
    recorder.register(observer)

    def _remove() -> None:
        recorder.unregister(observer)

    return _remove


