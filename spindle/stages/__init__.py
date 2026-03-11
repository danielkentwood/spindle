"""spindle-eval Stage implementations.

Each Stage wraps a Spindle v2 subsystem and implements the spindle-eval
Stage protocol (run / input_schema / output_schema).

Stages are designed to work without spindle-eval installed; in that case,
they can still be used as plain callable objects.
"""

from spindle.stages.preprocessing import PreprocessingStage
from spindle.stages.kos_extraction import KOSExtractionStage
from spindle.stages.ontology_synthesis import OntologySynthesisStage
from spindle.stages.retrieval import RetrievalStage
from spindle.stages.generation import GenerationStage
from spindle.stages.entity_resolution import EntityResolutionStage

__all__ = [
    "PreprocessingStage",
    "KOSExtractionStage",
    "OntologySynthesisStage",
    "RetrievalStage",
    "GenerationStage",
    "EntityResolutionStage",
]
