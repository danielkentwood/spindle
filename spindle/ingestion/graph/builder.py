"""Helper for building document graphs during ingestion."""

from __future__ import annotations

from typing import Iterable

from spindle.ingestion.types import (
    ChunkArtifact,
    DocumentArtifact,
    DocumentGraph,
    DocumentGraphEdge,
    DocumentGraphNode,
)


class DocumentGraphBuilder:
    """Utility to construct `DocumentGraph` incrementally."""

    def __init__(self) -> None:
        self._graph = DocumentGraph()

    @property
    def graph(self) -> DocumentGraph:
        return self._graph

    def add_document(self, artifact: DocumentArtifact) -> DocumentGraphNode:
        node = DocumentGraphNode(
            node_id=f"doc::{artifact.document_id}",
            document_id=artifact.document_id,
            label=artifact.metadata.get("title")
            or artifact.source_path.name,
            attributes=artifact.metadata,
        )
        self._graph.add_node(node)
        return node

    def add_chunks(
        self,
        document_node: DocumentGraphNode,
        chunks: Iterable[ChunkArtifact],
    ) -> list[DocumentGraphNode]:
        chunk_nodes: list[DocumentGraphNode] = []
        for index, chunk in enumerate(chunks):
            node = DocumentGraphNode(
                node_id=f"chunk::{chunk.chunk_id}",
                document_id=chunk.document_id,
                label=f"Chunk {index}",
                attributes=chunk.metadata,
            )
            self._graph.add_node(node)
            self._graph.add_edge(
                DocumentGraphEdge(
                    edge_id=f"doc-chunk::{document_node.node_id}->{node.node_id}",
                    source_id=document_node.node_id,
                    target_id=node.node_id,
                    relation="has_chunk",
                    attributes={},
                )
            )
            chunk_nodes.append(node)
        return chunk_nodes

    def add_relationship(
        self,
        source: DocumentGraphNode,
        target: DocumentGraphNode,
        relation: str,
        **attributes: str,
    ) -> None:
        self._graph.add_edge(
            DocumentGraphEdge(
                edge_id=f"{relation}::{source.node_id}->{target.node_id}",
                source_id=source.node_id,
                target_id=target.node_id,
                relation=relation,
                attributes=attributes,
            )
        )

