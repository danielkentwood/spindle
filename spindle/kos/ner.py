"""NER pass implementations for the KOS extraction pipeline.

Three passes with increasing cost and decreasing recall speed:
1. fast_pass   — Aho-Corasick (O(n) text scan)
2. medium_pass — multistep resolution (exact-label + optional ANN)
3. discovery_pass — GLiNER2 open NER (discovers novel terms)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Tuple

if TYPE_CHECKING:
    from spindle.kos.service import KOSService
    from spindle.preprocessing.models import Chunk

from spindle.kos.models import EntityMention
from spindle.preprocessing.offsets import reconstruct_coref_resolved_text


def fast_pass(
    chunks: List["Chunk"],
    kos_service: "KOSService",
    longest_match_only: bool = True,
) -> Tuple[List[Dict[str, Any]], List["Chunk"]]:
    """Pass 1: Aho-Corasick scan over coref-resolved chunk text.

    Args:
        chunks: Preprocessed chunks.
        kos_service: KOSService instance with loaded indices.
        longest_match_only: Filter overlapping matches.

    Returns:
        Tuple of (matched mentions as dicts, unmatched_chunks).
    """
    matched: List[Dict[str, Any]] = []
    unmatched_chunks: List["Chunk"] = []

    for chunk in chunks:
        resolved_text = reconstruct_coref_resolved_text(chunk)
        mentions = kos_service.search_ahocorasick(
            resolved_text, longest_match_only=longest_match_only
        )
        if mentions:
            for m in mentions:
                matched.append(
                    {
                        "text": m.text,
                        "start": m.start,
                        "end": m.end,
                        "concept_uri": m.concept_uri,
                        "matched_label": m.matched_label,
                        "pref_label": m.pref_label,
                        "confidence": 1.0,
                        "method": "aho_corasick",
                        "chunk": chunk,
                    }
                )
        else:
            unmatched_chunks.append(chunk)

    return matched, unmatched_chunks


def medium_pass(
    unmatched_chunks: List["Chunk"],
    kos_service: "KOSService",
    threshold: float = 0.7,
) -> Tuple[List[Dict[str, Any]], List["Chunk"]]:
    """Pass 2: Multi-step resolution over unmatched chunk text segments.

    Splits unmatched chunks into noun-phrase-like candidates, then resolves
    via exact-label → ANN.

    Args:
        unmatched_chunks: Chunks not covered by fast_pass.
        kos_service: KOSService with multistep resolution.
        threshold: Minimum ANN score to accept a resolution.

    Returns:
        Tuple of (resolved mentions as dicts, still_unmatched_chunks).
    """
    resolved: List[Dict[str, Any]] = []
    still_unmatched: List["Chunk"] = []

    for chunk in unmatched_chunks:
        candidates = _extract_candidate_spans(chunk)
        if not candidates:
            still_unmatched.append(chunk)
            continue

        results = kos_service.resolve_multistep(candidates, threshold=threshold)
        chunk_resolved = False
        for result in results:
            if result.resolved:
                resolved.append(
                    {
                        "text": result.mention,
                        "start": None,
                        "end": None,
                        "concept_uri": result.concept_uri,
                        "pref_label": result.pref_label,
                        "score": result.score,
                        "confidence": result.score,
                        "method": result.method,
                        "chunk": chunk,
                    }
                )
                chunk_resolved = True

        if not chunk_resolved:
            still_unmatched.append(chunk)

    return resolved, still_unmatched


def discovery_pass(
    unmatched_chunks: List["Chunk"],
    kos_service: "KOSService",
    labels: List[str] | None = None,
    threshold: float = 0.5,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Pass 3: GLiNER2 open NER for novel concept discovery.

    Args:
        unmatched_chunks: Chunks still unresolved after medium pass.
        kos_service: KOSService (used for label seeding and provenance).
        labels: Entity type labels for GLiNER (defaults to KOS label set).
        threshold: GLiNER confidence threshold.

    Returns:
        Tuple of (novel_candidates, matched_existing).
    """
    if labels is None:
        labels = kos_service.get_label_set(include_alt=False)[:50]  # cap for performance

    try:
        from gliner2 import GLiNER2  # noqa: F811
    except ImportError:
        return [], []

    novel_candidates: List[Dict[str, Any]] = []
    matched_existing: List[Dict[str, Any]] = []

    try:
        model = _get_gliner_model()
    except Exception:
        return [], []

    for chunk in unmatched_chunks:
        resolved_text = reconstruct_coref_resolved_text(chunk)
        try:
            result = model.extract_entities(
                resolved_text, labels,
                include_confidence=True, include_spans=True,
            )
        except Exception:
            continue

        entities_by_label = result.get("entities", {})
        for label, entities in entities_by_label.items():
            for entity in entities:
                score = entity.get("confidence", 0.0)
                if score < threshold:
                    continue
                span_text = entity.get("text", "")

                # Check if the discovered entity matches an existing concept
                resolution = kos_service.resolve_multistep([span_text], threshold=0.85)
                if resolution and resolution[0].resolved:
                    matched_existing.append(
                        {
                            "text": span_text,
                            "concept_uri": resolution[0].concept_uri,
                            "pref_label": resolution[0].pref_label,
                            "confidence": score,
                            "method": "gliner_matched",
                            "chunk": chunk,
                        }
                    )
                else:
                    novel_candidates.append(
                        {
                            "text": span_text,
                            "label": label,
                            "confidence": score,
                            "method": "gliner_discovery",
                            "chunk": chunk,
                        }
                    )

    return novel_candidates, matched_existing


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_gliner_model_cache: Any = None


def _get_gliner_model(model_name: str = "fastino/gliner2-base-v1") -> Any:
    """Load (and cache) a GLiNER2 model."""
    global _gliner_model_cache
    if _gliner_model_cache is None:
        from gliner2 import GLiNER2
        _gliner_model_cache = GLiNER2.from_pretrained(model_name)
    return _gliner_model_cache


def _extract_candidate_spans(chunk: "Chunk") -> List[str]:
    """Extract simple noun-phrase-like candidate strings from a chunk.

    Uses a heuristic whitespace tokeniser as a lightweight alternative to
    a full NLP pipeline.  Returns up to 20 unique tokens.
    """
    import re
    text = reconstruct_coref_resolved_text(chunk)
    # Extract capitalised phrases and all words ≥ 3 chars
    phrases = re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", text)
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text)
    candidates = list(dict.fromkeys(phrases + words))
    return candidates[:20]
