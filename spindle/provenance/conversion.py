"""Utility for converting extracted triples to normalized provenance objects.

Maps a ``Triple`` (BAML extraction output) to a ``ProvenanceObject`` suitable
for persisting to ``ProvenanceStore``.  The mapping is:

- ``object_id``   → deterministic SHA-256 from SPO + source (via ``triple_provenance_id``)
- ``object_type`` → ``"kg_edge"``
- One ``ProvenanceDoc`` per triple (keyed by ``source.source_name``)
- ``EvidenceSpan`` list from ``triple.supporting_spans``

Sentinel offsets (``-1``) produced by ``_compute_all_span_indices`` when a
span cannot be located are converted to ``None`` so the store does not
persist meaningless offsets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from spindle.provenance.keys import triple_provenance_id
from spindle.provenance.models import EvidenceSpan, ProvenanceDoc, ProvenanceObject

if TYPE_CHECKING:
    from spindle.baml_client.types import CharacterSpan, Triple


def _char_span_to_evidence(span: "CharacterSpan") -> EvidenceSpan:
    """Convert a BAML ``CharacterSpan`` to an ``EvidenceSpan``.

    Sentinel ``-1`` values (produced when span text is not found in source)
    are stored as ``None`` so the provenance store does not record false
    offset information.
    """
    start: Optional[int] = span.start if (span.start is not None and span.start >= 0) else None
    end: Optional[int] = span.end if (span.end is not None and span.end >= 0) else None
    return EvidenceSpan(text=span.text, start_offset=start, end_offset=end)


def triple_to_provenance_object(triple: "Triple") -> ProvenanceObject:
    """Convert an extracted ``Triple`` to a normalized ``ProvenanceObject``.

    The returned object is ready to be passed directly to
    ``ProvenanceStore.create_provenance()``.

    Args:
        triple: Fully post-processed ``Triple`` (with span indices computed).

    Returns:
        ``ProvenanceObject`` with ``object_type="kg_edge"`` and one doc per
        ``triple.source.source_name`` containing evidence spans.
    """
    object_id = triple_provenance_id(triple)

    evidence_spans = [_char_span_to_evidence(span) for span in triple.supporting_spans]

    doc = ProvenanceDoc(
        doc_id=triple.source.source_name,
        spans=evidence_spans,
    )

    return ProvenanceObject(
        object_id=object_id,
        object_type="kg_edge",
        docs=[doc],
    )
