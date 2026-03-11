"""Stage wrapper for LLM generation via SpindleExtractor."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from spindle.extraction.extractor import SpindleExtractor


class GenerationStage:
    """Wraps SpindleExtractor for spindle-eval Stage protocol.

    This stage receives retrieved context chunks and runs knowledge-graph
    triple extraction, returning structured triples.

    Args:
        extractor: SpindleExtractor instance (created lazily if not provided).
        tracker: Optional tracker for metrics emission.
    """

    name: str = "generation"

    def __init__(
        self,
        extractor: Optional["SpindleExtractor"] = None,
        ontology: Optional[Any] = None,
        tracker: Optional[Any] = None,
    ) -> None:
        self._extractor = extractor
        self._ontology = ontology
        self._tracker = tracker

    def run(
        self,
        text: str,
        source_name: str = "unknown",
        source_url: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run triple extraction on input text.

        Args:
            text: Document or chunk text to extract from.
            source_name: Source document identifier.
            source_url: Optional URL of the source.

        Returns:
            Dict with ``triples``, ``reasoning``, and ``triple_count``.
        """
        extractor = self._extractor or self._create_extractor()
        result = extractor.extract(
            text=text,
            source_name=source_name,
            source_url=source_url,
        )
        from spindle.extraction import triples_to_dict
        triples_data = triples_to_dict(result.triples)
        return {
            "triples": triples_data,
            "reasoning": result.reasoning if hasattr(result, "reasoning") else None,
            "triple_count": len(triples_data),
        }

    def run_batch(
        self,
        texts: List[str],
        source_names: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Run extraction on multiple texts.

        Args:
            texts: List of text strings.
            source_names: Optional parallel list of source names.

        Returns:
            List of extraction result dicts.
        """
        results = []
        for i, text in enumerate(texts):
            name = (source_names or [])[i] if source_names and i < len(source_names) else f"doc-{i}"
            results.append(self.run(text, source_name=name))
        return results

    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "source_name": {"type": "string"},
            },
            "required": ["text"],
        }

    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "triples": {"type": "array"},
                "triple_count": {"type": "integer"},
            },
        }

    def __call__(self, text: str, **kwargs: Any) -> Dict[str, Any]:
        return self.run(text, **kwargs)

    def _create_extractor(self) -> "SpindleExtractor":
        from spindle.extraction.extractor import SpindleExtractor
        return SpindleExtractor(ontology=self._ontology, tracker=self._tracker)
