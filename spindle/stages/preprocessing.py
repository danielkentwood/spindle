"""Stage wrapper for SpindlePreprocessor."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from spindle.preprocessing.models import Chunk


class PreprocessingStage:
    """Wraps SpindlePreprocessor for spindle-eval Stage protocol.

    Args:
        cfg: Configuration dict or Hydra DictConfig with keys:
             chunk_size, overlap, strategy, coref_model, output_dir.
        tracker: Optional tracker for metrics emission.
    """

    name: str = "preprocessing"

    def __init__(
        self,
        cfg: Optional[Dict[str, Any]] = None,
        tracker: Optional[Any] = None,
    ) -> None:
        self._cfg = cfg or {}
        self._tracker = tracker

    def run(self, paths: List[str], **kwargs: Any) -> List["Chunk"]:
        """Run preprocessing on a list of file paths.

        Args:
            paths: List of paths to source documents.

        Returns:
            List of Chunk objects.
        """
        from spindle.preprocessing.preprocessor import SpindlePreprocessor

        cfg = dict(self._cfg)
        cfg["input_paths"] = paths
        preprocessor = SpindlePreprocessor(tracker=self._tracker)
        return preprocessor(cfg)

    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "paths": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["paths"],
        }

    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "source_id": {"type": "string"},
                    "metadata": {"type": "object"},
                },
            },
        }

    def __call__(self, paths: List[str], **kwargs: Any) -> List["Chunk"]:
        return self.run(paths, **kwargs)
