"""Built-in helper functions that templates can reference."""

from __future__ import annotations

from langchain_core.documents import Document


def strip_empty_text(documents: list[Document]) -> list[Document]:
    """Remove documents whose text is empty or whitespace only."""

    return [doc for doc in documents if doc.page_content.strip()]

