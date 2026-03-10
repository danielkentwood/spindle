"""Offset mapping utilities for chunk-to-document position translation.

These helpers let callers convert a (chunk_index, chunk_relative_offset) pair
into an absolute offset within the full document text, and vice-versa.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from spindle.preprocessing.models import Chunk


def to_document_offset(chunk: Chunk, chunk_relative_offset: int) -> Optional[int]:
    """Convert a chunk-relative character offset to a document-absolute offset.

    Args:
        chunk: The Chunk containing ``start_index`` in its metadata.
        chunk_relative_offset: Offset relative to the start of ``chunk.text``.

    Returns:
        Absolute offset within the source document, or None if ``start_index``
        is not available.
    """
    start = chunk.start_index
    if start is None:
        return None
    return start + chunk_relative_offset


def build_offset_map(chunks: List[Chunk]) -> Dict[int, Tuple[int, int]]:
    """Build a mapping from document offset to (chunk_index, chunk_relative_offset).

    Args:
        chunks: List of Chunk objects sharing the same ``source_id``.

    Returns:
        Dict mapping each absolute character offset that falls within any chunk
        to (chunk_list_index, offset_within_chunk).  Useful for projecting
        document-level annotations (e.g. coref chains) onto chunks.
    """
    offset_map: Dict[int, Tuple[int, int]] = {}
    for list_idx, chunk in enumerate(chunks):
        start = chunk.start_index
        if start is None:
            continue
        for i in range(len(chunk.text)):
            offset_map[start + i] = (list_idx, i)
    return offset_map


def reconstruct_coref_resolved_text(chunk: Chunk) -> str:
    """Reconstruct the chunk text with coreference mentions replaced by their resolutions.

    The original ``chunk.text`` is **never modified**.  This function produces a
    new string where each non-representative mention (stored in
    ``chunk.metadata["coref_annotations"]``) is replaced by its resolved form.

    Replacements are applied in reverse order of ``span_start`` so that earlier
    offsets are not invalidated by later substitutions.

    Args:
        chunk: Chunk with optional ``coref_annotations`` in metadata.

    Returns:
        A new string with coref mentions replaced, or the original text if there
        are no annotations.
    """
    annotations: List[Dict[str, Any]] = chunk.coref_annotations
    if not annotations:
        return chunk.text

    # Sort by start descending so later replacements don't shift earlier offsets
    sorted_annotations = sorted(annotations, key=lambda a: a["span_start"], reverse=True)

    text = chunk.text
    for ann in sorted_annotations:
        start = ann.get("span_start")
        end = ann.get("span_end")
        resolved = ann.get("resolved_to")
        if start is None or end is None or resolved is None:
            continue
        text = text[:start] + resolved + text[end:]

    return text
