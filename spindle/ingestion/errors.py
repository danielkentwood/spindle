"""Exception hierarchy for the ingestion subsystem."""


class IngestionError(Exception):
    """Base error for ingestion related failures."""


class TemplateResolutionError(IngestionError):
    """Raised when no template can be resolved for a document."""


class StorageError(IngestionError):
    """Raised when persistence layers encounter an issue."""


class PipelineExecutionError(IngestionError):
    """Raised when a pipeline step fails irrecoverably."""

