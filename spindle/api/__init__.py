"""Spindle REST API module.

Provides FastAPI-based REST endpoints for Spindle's core services:
- Document ingestion
- Triple extraction
- Ontology recommendation and extension
- Entity resolution
- Process extraction
"""

from spindle.api.main import app

__all__ = ["app"]

