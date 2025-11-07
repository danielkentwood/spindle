"""Default templates shipped with spindle."""

from __future__ import annotations

from spindle.ingestion.types import TemplateSelector, TemplateSpec


DEFAULT_TEMPLATE_SPECS = (
    TemplateSpec(
        name="default-text",
        description="Plain text and Markdown documents",
        selector=TemplateSelector(
            file_extensions=(".txt", ".md", ".mdx", ".rst"),
        ),
        loader="langchain_community.document_loaders.TextLoader",
        splitter={
            "name": "langchain_text_splitters.RecursiveCharacterTextSplitter",
            "params": {"chunk_size": 800, "chunk_overlap": 100},
        },
    ),
    TemplateSpec(
        name="default-pdf",
        description="PDF documents with layout-aware parsing",
        selector=TemplateSelector(
            file_extensions=(".pdf",),
        ),
        loader="langchain_community.document_loaders.PDFMinerLoader",
        preprocessors=(
            "spindle.ingestion.templates.hooks.strip_empty_text",
        ),
        splitter={
            "name": "langchain_text_splitters.RecursiveCharacterTextSplitter",
            "params": {"chunk_size": 1000, "chunk_overlap": 150},
        },
    ),
)

