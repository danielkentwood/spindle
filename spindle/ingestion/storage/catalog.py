"""SQLite-backed catalog for ingestion artifacts."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from typing import Iterator, Sequence

from sqlalchemy import JSON, DateTime, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from spindle.ingestion.types import (
    ChunkArtifact,
    Corpus,
    CorpusDocument,
    DocumentArtifact,
    DocumentGraph,
    DocumentGraphEdge,
    DocumentGraphNode,
    IngestionResult,
)


class Base(DeclarativeBase):
    pass


class DocumentRow(Base):
    __tablename__ = "documents"

    document_id: Mapped[str] = mapped_column(String, primary_key=True)
    source_path: Mapped[str] = mapped_column(String, nullable=False)
    checksum: Mapped[str] = mapped_column(String, nullable=False)
    loader_name: Mapped[str] = mapped_column(String, nullable=False)
    template_name: Mapped[str] = mapped_column(String, nullable=False)
    metadata_: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    bytes_read: Mapped[int] = mapped_column(Integer, default=0)


class ChunkRow(Base):
    __tablename__ = "chunks"

    chunk_id: Mapped[str] = mapped_column(String, primary_key=True)
    document_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    text: Mapped[str] = mapped_column(String, nullable=False)
    metadata_: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    embedding: Mapped[list[float] | None] = mapped_column(JSON, nullable=True)


class GraphNodeRow(Base):
    __tablename__ = "graph_nodes"

    node_id: Mapped[str] = mapped_column(String, primary_key=True)
    document_id: Mapped[str] = mapped_column(String, nullable=False)
    label: Mapped[str] = mapped_column(String, nullable=False)
    attributes: Mapped[dict] = mapped_column(JSON, default=dict)


class GraphEdgeRow(Base):
    __tablename__ = "graph_edges"

    edge_id: Mapped[str] = mapped_column(String, primary_key=True)
    source_id: Mapped[str] = mapped_column(String, nullable=False)
    target_id: Mapped[str] = mapped_column(String, nullable=False)
    relation: Mapped[str] = mapped_column(String, nullable=False)
    attributes: Mapped[dict] = mapped_column(JSON, default=dict)


class IngestionRunRow(Base):
    __tablename__ = "ingestion_runs"

    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    started_at: Mapped[datetime] = mapped_column(DateTime)
    finished_at: Mapped[datetime] = mapped_column(DateTime)
    processed_documents: Mapped[int] = mapped_column(Integer)
    processed_chunks: Mapped[int] = mapped_column(Integer)
    bytes_read: Mapped[int] = mapped_column(Integer)
    errors: Mapped[list[str]] = mapped_column(JSON, default=list)
    extras: Mapped[dict] = mapped_column(JSON, default=dict)


class CorpusRow(Base):
    """SQLAlchemy model for corpus storage."""

    __tablename__ = "corpora"

    corpus_id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(String, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    pipeline_state: Mapped[dict] = mapped_column(JSON, default=dict)


class CorpusDocumentRow(Base):
    """SQLAlchemy model linking documents to corpora."""

    __tablename__ = "corpus_documents"

    corpus_id: Mapped[str] = mapped_column(String, primary_key=True)
    document_id: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    added_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class DocumentCatalog:
    """Persist ingestion runs to a SQLite database."""

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

    def persist_result(self, result: IngestionResult, run_id: str) -> None:
        with self.session() as session:
            self._store_documents(session, result.documents)
            self._store_chunks(session, result.chunks)
            self._store_graph(session, result.document_graph)
            self._store_run(session, result, run_id)

    def _store_documents(
        self, session: Session, documents: Sequence[DocumentArtifact]
    ) -> None:
        for document in documents:
            session.merge(
                DocumentRow(
                    document_id=document.document_id,
                    source_path=str(document.source_path),
                    checksum=document.checksum,
                    loader_name=document.loader_name,
                    template_name=document.template_name,
                    metadata_=document.metadata,
                    created_at=document.created_at,
                    bytes_read=len(document.raw_bytes or b""),
                )
            )

    def _store_chunks(self, session: Session, chunks: Sequence[ChunkArtifact]) -> None:
        for chunk in chunks:
            session.merge(
                ChunkRow(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    text=chunk.text,
                    metadata_=chunk.metadata,
                    embedding=list(chunk.embedding) if chunk.embedding else None,
                )
            )

    def _store_graph(self, session: Session, graph: DocumentGraph) -> None:
        self._store_graph_nodes(session, graph.nodes)
        self._store_graph_edges(session, graph.edges)

    @staticmethod
    def _store_graph_nodes(
        session: Session, nodes: Sequence[DocumentGraphNode]
    ) -> None:
        for node in nodes:
            session.merge(
                GraphNodeRow(
                    node_id=node.node_id,
                    document_id=node.document_id,
                    label=node.label,
                    attributes=node.attributes,
                )
            )

    @staticmethod
    def _store_graph_edges(
        session: Session, edges: Sequence[DocumentGraphEdge]
    ) -> None:
        for edge in edges:
            session.merge(
                GraphEdgeRow(
                    edge_id=edge.edge_id,
                    source_id=edge.source_id,
                    target_id=edge.target_id,
                    relation=edge.relation,
                    attributes=edge.attributes,
                )
            )

    def _store_run(self, session: Session, result: IngestionResult, run_id: str) -> None:
        metrics = result.metrics
        session.merge(
            IngestionRunRow(
                run_id=run_id,
                started_at=metrics.started_at,
                finished_at=metrics.finished_at or datetime.utcnow(),
                processed_documents=metrics.processed_documents,
                processed_chunks=metrics.processed_chunks,
                bytes_read=metrics.bytes_read,
                errors=metrics.errors,
                extras=metrics.extra,
            )
        )

