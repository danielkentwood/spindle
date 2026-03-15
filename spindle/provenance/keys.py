"""Deterministic provenance key generation for extracted triples.

The provenance object_id for a KG edge is derived solely from its SPO tuple
and source, so the same logical fact extracted from the same document always
maps to the same provenance record — enabling idempotent upserts and reverse
lookups without needing to pass IDs across layers.
"""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spindle.baml_client.types import Triple


def _canonical_token(text: str) -> str:
    """Lowercase and collapse whitespace in a string for stable comparison."""
    return re.sub(r'\s+', ' ', text.strip().lower())


def triple_provenance_id(triple: "Triple") -> str:
    """Return a stable, deterministic SHA-256 provenance object_id for a triple.

    The key is derived from the normalized canonical form of::

        (subject.name, predicate, object.name, source.source_name, source.source_url)

    Properties:
    - Case-insensitive: "Alice Johnson" == "ALICE JOHNSON"
    - Whitespace-normalized: "Alice  Johnson" == "Alice Johnson"
    - Span-independent: span text/offsets and extraction_datetime do not affect the key
    - Stable across repeated re-extractions of the same fact from the same source
    - source_url=None is distinct from source_url=""

    Returns:
        64-character lowercase hex SHA-256 digest.
    """
    subject = _canonical_token(triple.subject.name)
    predicate = _canonical_token(triple.predicate)
    obj = _canonical_token(triple.object.name)
    source_name = _canonical_token(triple.source.source_name)
    source_url = _canonical_token(triple.source.source_url) if triple.source.source_url is not None else "__none__"

    canonical = f"{subject}|{predicate}|{obj}|{source_name}|{source_url}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
