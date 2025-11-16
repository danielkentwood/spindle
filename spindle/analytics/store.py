"""SQLite-backed persistence for ingestion analytics observations."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Integer,
    String,
    create_engine,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from spindle.analytics.schema import DocumentObservation, ServiceEventRecord
from spindle.observability.storage import EventLogStore


class AnalyticsBase(DeclarativeBase):
    """Declarative base for analytics persistence."""


class IngestionObservationRow(AnalyticsBase):
    """Row storing serialized document observations."""

    __tablename__ = "ingestion_observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(String, index=True)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), index=True)
    schema_version: Mapped[str] = mapped_column(String, default="1.0.0")
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), default=datetime.utcnow)

    events: Mapped[list["ObservationServiceEventRow"]] = relationship(
        back_populates="observation",
        cascade="all, delete-orphan",
    )


class ObservationServiceEventRow(AnalyticsBase):
    """Service events associated with a specific observation."""

    __tablename__ = "ingestion_observation_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    observation_id: Mapped[int] = mapped_column(
        ForeignKey("ingestion_observations.id", ondelete="CASCADE"),
        index=True,
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=False), index=True)
    service: Mapped[str] = mapped_column(String, index=True)
    name: Mapped[str] = mapped_column(String, index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    observation: Mapped[IngestionObservationRow] = relationship(back_populates="events")


class AnalyticsStore:
    """Persist analytics observations and mirror service events."""

    def __init__(
        self,
        database_url: str,
        *,
        event_store: EventLogStore | None = None,
    ) -> None:
        self._engine = create_engine(database_url, future=True)
        self._session_factory = sessionmaker(self._engine, expire_on_commit=False)
        AnalyticsBase.metadata.create_all(self._engine)
        self._event_store = event_store or EventLogStore(database_url)

    def __enter__(self) -> "AnalyticsStore":
        """Support context manager protocol."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close connections when exiting context."""
        self.close()

    def close(self) -> None:
        """Close all database connections and dispose of the engine."""
        if hasattr(self, '_engine'):
            self._engine.dispose()
        if hasattr(self, '_event_store'):
            self._event_store.close()

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

    def persist_observations(
        self,
        observations: Sequence[DocumentObservation] | Iterable[DocumentObservation],
    ) -> None:
        """Persist a sequence of document observations."""

        observation_list = list(observations)
        if not observation_list:
            return

        service_events = [
            record.to_service_event()
            for observation in observation_list
            for record in observation.observability.service_events
        ]
        if service_events:
            self._event_store.persist_events(service_events)

        with self.session() as session:
            for observation in observation_list:
                row = IngestionObservationRow(
                    document_id=observation.metadata.document_id,
                    ingested_at=observation.metadata.ingested_at,
                    schema_version=observation.schema_version,
                    payload=observation.model_dump(mode="json"),
                )
                session.add(row)
                session.flush()
                session.add_all(
                    ObservationServiceEventRow(
                        observation_id=row.id,
                        timestamp=record.timestamp,
                        service=record.service,
                        name=record.name,
                        payload=dict(record.payload),
                    )
                    for record in observation.observability.service_events
                )

    def fetch_observations(
        self,
        *,
        document_id: str | None = None,
        limit: int | None = None,
    ) -> list[DocumentObservation]:
        """Fetch observations, optionally filtered by document identifier."""

        stmt = select(IngestionObservationRow).order_by(
            IngestionObservationRow.ingested_at.desc()
        )
        if document_id:
            stmt = stmt.where(IngestionObservationRow.document_id == document_id)
        if limit:
            stmt = stmt.limit(limit)
        with self.session() as session:
            rows = session.execute(stmt).scalars().all()
        return [self._row_to_observation(row) for row in rows]

    def fetch_service_events(
        self,
        *,
        document_id: str | None = None,
        service: str | None = None,
        limit: int | None = None,
    ) -> list[ServiceEventRecord]:
        """Fetch persisted service events.
        
        If document_id is provided, queries observation-linked events.
        Otherwise, queries from EventLogStore (general service events).
        """

        # If document_id is specified, query observation-linked events
        if document_id:
            stmt = select(ObservationServiceEventRow).order_by(
                ObservationServiceEventRow.timestamp.desc()
            )
            stmt = stmt.join(ObservationServiceEventRow.observation).where(
                IngestionObservationRow.document_id == document_id
            )
            if service:
                stmt = stmt.where(ObservationServiceEventRow.service == service)
            if limit:
                stmt = stmt.limit(limit)
            with self.session() as session:
                rows = session.execute(stmt).scalars().all()
            return [
                ServiceEventRecord(
                    timestamp=row.timestamp,
                    service=row.service,
                    name=row.name,
                    payload=dict(row.payload or {}),
                )
                for row in rows
            ]
        
        # Otherwise, query from EventLogStore (general service events)
        events = self._event_store.fetch_events(service=service, limit=limit)
        return [
            ServiceEventRecord(
                timestamp=event.timestamp,
                service=event.service,
                name=event.name,
                payload=dict(event.payload or {}),
            )
            for event in events
        ]

    @staticmethod
    def _row_to_observation(row: IngestionObservationRow) -> DocumentObservation:
        return DocumentObservation.model_validate(row.payload)


__all__ = ["AnalyticsStore"]

