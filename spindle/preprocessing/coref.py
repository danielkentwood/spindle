"""Stage 3: Coreference resolution via fastcoref.

Resolves pronouns and nominal references within each document, then projects
the resolution chains onto individual chunks.  The chunk text is never
modified — annotations are stored in chunk.metadata["coref_annotations"].
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from spindle.preprocessing.models import Chunk


def resolve_coreferences_for_document(
    doc_text: str,
    chunks: List[Chunk],
    blacklist: Optional[Set[str]] = None,
    model_arch: str = "FCoref",
) -> List[Chunk]:
    """Run per-document coreference resolution and project onto chunks.

    Args:
        doc_text: Full document text (concatenation of all chunk texts is
                  insufficient — use the original Docling output so coref
                  chains span chunk boundaries correctly).
        chunks: All chunks for this document, sharing the same ``source_id``.
                Must carry ``start_index`` and ``end_index`` in metadata.
        blacklist: Optional set of lower-cased terms to exclude from
                   coref annotations.
        model_arch: fastcoref model architecture.  ``"FCoref"`` (default,
                    fast) or ``"LingMessCoref"`` (more accurate, slower).

    Returns:
        The same list of chunks, each with ``coref_annotations`` added to
        metadata (empty list when no chains overlap the chunk).

    Notes:
        Falls back to returning unchanged chunks when fastcoref is not installed.
    """
    try:
        chains = _run_fastcoref(doc_text, model_arch)
    except ImportError:
        # fastcoref not installed — return chunks unchanged
        for chunk in chunks:
            chunk.metadata.setdefault("coref_annotations", [])
        return chunks

    if blacklist is None:
        blacklist = set()

    # Initialize coref_annotations on all chunks
    for chunk in chunks:
        chunk.metadata["coref_annotations"] = []

    # Project chains onto chunks
    for chain_id, chain in enumerate(chains):
        # Select representative: longest mention
        representative = max(chain, key=lambda m: len(m["text"]))
        representative_text = representative["text"]

        if representative_text.lower() in blacklist:
            continue

        for mention in chain:
            if mention is representative:
                continue
            if mention["text"].lower() in blacklist:
                continue

            mention_start = mention["start"]
            mention_end = mention["end"]

            # Find which chunk contains this mention
            for chunk in chunks:
                chunk_start = chunk.metadata.get("start_index")
                chunk_end = chunk.metadata.get("end_index")
                if chunk_start is None or chunk_end is None:
                    continue
                if chunk_start <= mention_start < chunk_end:
                    chunk.metadata["coref_annotations"].append(
                        {
                            "mention": mention["text"],
                            "span_start": mention_start - chunk_start,
                            "span_end": mention_end - chunk_start,
                            "resolved_to": representative_text,
                            "chain_id": f"chain_{chain_id}",
                            "confidence": mention.get("confidence", 1.0),
                        }
                    )
                    break

    return chunks


def _run_fastcoref(doc_text: str, model_arch: str) -> List[List[Dict[str, Any]]]:
    """Run fastcoref on ``doc_text`` and return a list of coreference chains.

    Each chain is a list of mention dicts with keys:
        ``text``, ``start``, ``end``, ``confidence`` (optional).

    Raises:
        ImportError: If fastcoref is not installed.
    """
    try:
        from fastcoref import FCoref, LingMessCoref
    except ImportError as exc:
        raise ImportError(
            "fastcoref is required for coreference resolution. "
            "Install it with: uv pip install fastcoref"
        ) from exc

    if model_arch == "LingMessCoref":
        model = LingMessCoref()
    else:
        model = FCoref()

    result = model.predict(texts=[doc_text])
    preds = result[0] if result else None
    if preds is None:
        return []

    # fastcoref returns clusters as lists of (start, end) tuples
    clusters = getattr(preds, "clusters", None) or []
    chains: List[List[Dict[str, Any]]] = []
    for cluster in clusters:
        chain = []
        for start, end in cluster:
            chain.append({"text": doc_text[start:end], "start": start, "end": end})
        chains.append(chain)
    return chains
