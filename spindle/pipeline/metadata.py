"""Stage 2: Metadata Standards extraction.

Extracts metadata schema elements from corpus documents to establish
consistent data description standards.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import JSON, DateTime, String, Boolean, select
from sqlalchemy.orm import Mapped, mapped_column
from langfuse import observe, get_client as get_langfuse_client
import baml_py

from spindle.baml_client import b
from spindle.extraction.helpers import _extract_model_from_collector
from spindle.baml_client.types import MetadataElement as BAMLMetadataElement
from spindle.ingestion.storage.catalog import Base
from spindle.ingestion.storage.corpus import CorpusManager
from spindle.pipeline.base import BasePipelineStage
from spindle.pipeline.types import (
    MetadataElement,
    MetadataElementType,
    PipelineStage,
    PipelineState,
)


class MetadataElementRow(Base):
    """SQLAlchemy model for metadata element storage."""

    __tablename__ = "metadata_elements"

    element_id: Mapped[str] = mapped_column(String, primary_key=True)
    corpus_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    element_type: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(String, nullable=False)
    data_type: Mapped[str] = mapped_column(String, nullable=False)
    required: Mapped[bool] = mapped_column(Boolean, default=False)
    allowed_values: Mapped[list | None] = mapped_column(JSON, nullable=True)
    default_value: Mapped[str | None] = mapped_column(String, nullable=True)
    examples: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class MetadataStage(BasePipelineStage[MetadataElement]):
    """Stage 2: Metadata Standards extraction.

    Analyzes corpus documents to identify metadata patterns and
    establish a consistent metadata schema with:
    - Structural elements (format, encoding)
    - Descriptive elements (title, author, keywords)
    - Administrative elements (dates, provenance)
    """

    stage = PipelineStage.METADATA

    def __init__(
        self,
        corpus_manager: CorpusManager,
        graph_store: Optional[Any] = None,
    ) -> None:
        """Initialize metadata stage."""
        super().__init__(corpus_manager, graph_store)
        Base.metadata.create_all(self.corpus_manager._catalog._engine)

    @observe(as_type="generation", capture_input=False, capture_output=False)
    def extract_from_text(
        self,
        text: str,
        document_id: str,
        existing_artifacts: List[MetadataElement],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[MetadataElement]:
        """Extract metadata schema elements from document.

        Args:
            text: The document text (sample).
            document_id: Source document ID.
            existing_artifacts: Previously extracted elements.
            context: Optional context (e.g., document metadata).

        Returns:
            List of extracted MetadataElement objects.
        """
        # Get document metadata if available
        doc_metadata = ""
        if context and "document_metadata" in context:
            doc_metadata = json.dumps(context["document_metadata"], indent=2)

        # Convert existing elements to BAML format
        existing_baml = [
            BAMLMetadataElement(
                element_id=e.element_id,
                name=e.name,
                element_type=e.element_type.value,
                description=e.description,
                data_type=e.data_type,
                required=e.required,
                examples=list(e.examples),
            )
            for e in existing_artifacts
        ]

        # Call BAML extraction with collector
        collector = baml_py.baml_py.Collector("metadata-extraction-collector")
        result = b.with_options(collector=collector).ExtractMetadataSchema(
            text=text[:2000],  # Sample text
            document_metadata=doc_metadata,
            existing_elements=existing_baml,
        )

        # Extract model from collector
        model = _extract_model_from_collector(collector) or "CustomFast"

        # Update Langfuse generation
        langfuse = get_langfuse_client()
        langfuse.update_current_generation(
            name="ExtractMetadataSchema",
            model=model,
            input={
                "text": text[:2000],
                "document_id": document_id,
                "document_metadata": doc_metadata,
                "existing_elements": [e.name for e in existing_baml],
            },
            output={
                "elements": [
                    {
                        "name": e.name,
                        "element_type": e.element_type,
                        "description": e.description,
                        "data_type": e.data_type,
                        "required": e.required,
                        "examples": list(e.examples) if e.examples else [],
                    }
                    for e in result.elements
                ],
            },
        )

        # Convert to MetadataElement
        elements = []
        for baml_elem in result.elements:
            try:
                elem_type = MetadataElementType(baml_elem.element_type.lower())
            except ValueError:
                elem_type = MetadataElementType.DESCRIPTIVE

            element = MetadataElement(
                element_id=baml_elem.element_id or f"meta_{uuid.uuid4().hex[:8]}",
                name=baml_elem.name,
                element_type=elem_type,
                description=baml_elem.description,
                data_type=baml_elem.data_type,
                required=baml_elem.required,
                allowed_values=None,
                default_value=None,
                examples=list(baml_elem.examples) if baml_elem.examples else [],
                created_at=datetime.utcnow(),
            )
            elements.append(element)

        return elements

    def merge_artifacts(
        self,
        artifact_sets: List[List[MetadataElement]],
    ) -> List[MetadataElement]:
        """Merge metadata elements from multiple extractions.

        Args:
            artifact_sets: List of element lists to merge.

        Returns:
            Merged and deduplicated list of elements.
        """
        all_elements = []
        for elem_set in artifact_sets:
            all_elements.extend(elem_set)

        if not all_elements:
            return []

        # Merge by name
        merged: Dict[str, MetadataElement] = {}
        for elem in all_elements:
            key = elem.name.lower().strip()

            if key in merged:
                existing = merged[key]
                # Merge examples
                all_examples = set(existing.examples) | set(elem.examples)
                existing.examples = list(all_examples)
                # Use more specific description if available
                if len(elem.description) > len(existing.description):
                    existing.description = elem.description
            else:
                merged[key] = elem

        return list(merged.values())

    def persist_artifacts(
        self,
        corpus_id: str,
        artifacts: List[MetadataElement],
    ) -> int:
        """Persist metadata elements to SQLite.

        Args:
            corpus_id: The corpus identifier.
            artifacts: Elements to persist.

        Returns:
            Number of elements persisted.
        """
        with self.corpus_manager._catalog.session() as session:
            count = 0
            for elem in artifacts:
                if not elem.element_id:
                    elem.element_id = f"meta_{corpus_id}_{uuid.uuid4().hex[:8]}"

                session.merge(
                    MetadataElementRow(
                        element_id=elem.element_id,
                        corpus_id=corpus_id,
                        name=elem.name,
                        element_type=elem.element_type.value,
                        description=elem.description,
                        data_type=elem.data_type,
                        required=elem.required,
                        allowed_values=elem.allowed_values,
                        default_value=elem.default_value,
                        examples=elem.examples,
                        created_at=elem.created_at,
                    )
                )
                count += 1

            return count

    def load_artifacts(self, corpus_id: str) -> List[MetadataElement]:
        """Load metadata elements for a corpus.

        Args:
            corpus_id: The corpus identifier.

        Returns:
            List of MetadataElement objects.
        """
        with self.corpus_manager._catalog.session() as session:
            rows = (
                session.execute(
                    select(MetadataElementRow).where(
                        MetadataElementRow.corpus_id == corpus_id
                    )
                )
                .scalars()
                .all()
            )

            return [
                MetadataElement(
                    element_id=row.element_id,
                    name=row.name,
                    element_type=MetadataElementType(row.element_type),
                    description=row.description,
                    data_type=row.data_type,
                    required=row.required,
                    allowed_values=row.allowed_values,
                    default_value=row.default_value,
                    examples=row.examples or [],
                    created_at=row.created_at,
                )
                for row in rows
            ]

    def get_previous_stage_context(
        self,
        corpus_id: str,
        pipeline_state: PipelineState,
    ) -> Dict[str, Any]:
        """Get context from vocabulary stage."""
        context = {}

        # Get document metadata from corpus
        documents = self.corpus_manager.get_document_artifacts(corpus_id)
        if documents:
            context["document_metadata"] = documents[0].metadata

        return context

