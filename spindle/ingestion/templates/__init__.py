"""Template registry and configuration loading for ingestion."""

from .defaults import DEFAULT_TEMPLATE_SPECS
from .registry import (
    TemplateRegistry,
    load_templates_from_paths,
    merge_template_sequences,
)

__all__ = [
    "DEFAULT_TEMPLATE_SPECS",
    "TemplateRegistry",
    "load_templates_from_paths",
    "merge_template_sequences",
]

