"""LLM-based candidate extraction for KOS cold start.

Delegates to the existing spindle.pipeline vocabulary stage for BAML-driven
extraction.  Kept separate so the main extraction.py does not need BAML/LLM
imports when running in incremental (non-LLM) mode.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from spindle.preprocessing.models import Chunk


def extract_candidates_via_llm(chunks: List["Chunk"]) -> List[Dict[str, Any]]:
    """Extract concept candidates from chunks using the BAML vocabulary stage.

    Falls back to an empty list if BAML/pipeline dependencies are absent.
    """
    # Re-use the BAML b client directly to avoid needing a full
    # VocabularyStage (which requires CorpusManager / SQLAlchemy).
    try:
        from spindle.baml_client import b
    except ImportError:
        return []

    candidates: List[Dict[str, Any]] = []
    for chunk in chunks:
        try:
            result = b.ExtractControlledVocabulary(
                text=chunk.text,
                existing_terms=[],
                document_id=chunk.source_id,
            )
            for term in result.terms:
                candidates.append(
                    {
                        "text": term.preferred_label,
                        "pref_label": term.preferred_label,
                        "definition": term.definition or "",
                        "confidence": 0.8,
                        "method": "llm_cold_start",
                    }
                )
        except Exception:
            continue

    return candidates
