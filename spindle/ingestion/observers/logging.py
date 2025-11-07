"""Logging observer for ingestion events."""

from __future__ import annotations

import logging

from spindle.ingestion.types import IngestionEvent


LOGGER = logging.getLogger(__name__)


def logging_observer(event: IngestionEvent) -> None:
    if event.name == "stage_complete":
        LOGGER.debug(
            "Stage %s completed in %.2f ms",
            event.payload.get("stage"),
            event.payload.get("duration_ms", 0.0),
        )
    elif event.name == "graph_built":
        LOGGER.info(
            "Graph built for document %s with %s chunks",
            event.payload.get("document_id"),
            event.payload.get("chunk_count"),
        )

