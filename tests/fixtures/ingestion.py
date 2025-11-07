"""Fixtures for ingestion unit tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document


class DummyLoader:
    """Simple loader returning the file contents as a single document."""

    def __init__(self, file_path: str) -> None:
        self._path = Path(file_path)

    def load(self) -> list[Document]:
        text = self._path.read_text(encoding="utf-8") if self._path.exists() else ""
        return [Document(page_content=text, metadata={"source": str(self._path)})]


@dataclass
class DummySplitter:
    """Splitter that returns each line as a chunk."""

    def split_documents(self, documents: Iterable[Document]) -> list[Document]:
        chunks: list[Document] = []
        for document in documents:
            for idx, line in enumerate(document.page_content.splitlines() or [document.page_content]):
                chunks.append(
                    Document(
                        page_content=line,
                        metadata={
                            **document.metadata,
                            "line_number": idx,
                        },
                    )
                )
        return chunks


def dummy_metadata_extractor(documents: list[Document]) -> dict[str, str]:
    return {"num_documents": str(len(documents))}


def dummy_graph_hook(
    *,
    document_artifact,
    chunk_artifacts,
    document_node,
    chunk_nodes,
    builder,
) -> None:
    for chunk_node in chunk_nodes:
        builder.add_relationship(document_node, chunk_node, relation="contains")

