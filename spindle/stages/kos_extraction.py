"""Stage wrapper for KOSExtractionPipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from spindle.kos.service import KOSService
    from spindle.preprocessing.models import Chunk


class KOSExtractionStage:
    """Wraps KOSExtractionPipeline for spindle-eval Stage protocol.

    Args:
        kos_service: Live KOSService instance.
        stage_dir: Path to the ``kos/staging/`` directory.
        provenance_store: Optional ProvenanceStore for evidence recording.
        tracker: Optional tracker.
    """

    name: str = "kos_extraction"

    def __init__(
        self,
        kos_service: "KOSService",
        stage_dir: Optional[Path] = None,
        provenance_store: Optional[Any] = None,
        tracker: Optional[Any] = None,
    ) -> None:
        self._kos = kos_service
        self._stage_dir = stage_dir
        self._prov = provenance_store
        self._tracker = tracker

    def run(
        self,
        chunks: List["Chunk"],
        mode: str = "auto",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run KOS extraction on preprocessed chunks.

        Args:
            chunks: Chunks from PreprocessingStage.
            mode: ``"cold_start"`` | ``"incremental"`` | ``"auto"``.

        Returns:
            Summary dict from KOSExtractionPipeline.run().
        """
        from spindle.kos.extraction import KOSExtractionPipeline

        pipeline = KOSExtractionPipeline(
            kos_service=self._kos,
            stage_dir=self._stage_dir,
            provenance_store=self._prov,
            tracker=self._tracker,
        )
        return pipeline.run(chunks, mode=mode)

    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "array",
            "items": {"type": "object", "description": "Chunk"},
        }

    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "mode": {"type": "string"},
                "chunks_processed": {"type": "integer"},
            },
        }

    def __call__(self, chunks: List["Chunk"], **kwargs: Any) -> Dict[str, Any]:
        return self.run(chunks, **kwargs)
