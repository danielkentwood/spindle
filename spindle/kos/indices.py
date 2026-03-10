"""Derived indices for low-latency KOS operations.

Four indices are built from the Oxigraph store contents and rebuilt on reload:
1. Aho-Corasick NER automaton
2. Vector ANN search index (hnswlib)
3. Label-to-URI map
4. URI-to-Concept cache
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple

from spindle.kos.models import ConceptRecord, EntityMention


_PREFIXES = """
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX dct:  <http://purl.org/dc/terms/>
PREFIX spndl: <http://spindle.dev/ns/>
"""


def normalize_label(label: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation."""
    label = unicodedata.normalize("NFC", label).lower()
    label = re.sub(r"[^\w\s-]", "", label)
    label = re.sub(r"\s+", " ", label).strip()
    return label


def build_uri_concept_cache(store: object) -> Dict[str, ConceptRecord]:
    """Build a URI → ConceptRecord mapping from the Oxigraph store."""
    records: Dict[str, ConceptRecord] = {}

    query = _PREFIXES + """
    SELECT ?uri ?id ?pref ?def ?scope WHERE {
        GRAPH ?g {
            ?uri a skos:Concept ;
                 skos:prefLabel ?pref .
            OPTIONAL { ?uri dct:identifier ?id }
            OPTIONAL { ?uri skos:definition ?def }
            OPTIONAL { ?uri skos:scopeNote ?scope }
        }
    }
    """
    try:
        for row in store.query(query):
            uri = str(row[0])
            concept_id = _rdf_str(row[1]) if row[1] else uri.split("/")[-1]
            pref_label = _rdf_str(row[2])
            definition = _rdf_str(row[3]) if row[3] else None
            scope_note = _rdf_str(row[4]) if row[4] else None
            records[uri] = ConceptRecord(
                uri=uri,
                concept_id=concept_id,
                pref_label=pref_label,
                definition=definition,
                scope_note=scope_note,
                provenance_object_id=concept_id,
            )
    except Exception:
        return records

    # Second pass: broader/narrower/related/altLabels
    rel_query = _PREFIXES + """
    SELECT ?uri ?rel ?target WHERE {
        GRAPH ?g {
            ?uri a skos:Concept .
            { ?uri skos:broader ?target . BIND("broader" AS ?rel) }
            UNION { ?uri skos:narrower ?target . BIND("narrower" AS ?rel) }
            UNION { ?uri skos:related ?target . BIND("related" AS ?rel) }
        }
    }
    """
    try:
        for row in store.query(rel_query):
            uri = str(row[0])
            rel = str(row[1])
            target = str(row[2])
            if uri in records:
                getattr(records[uri], rel).append(target)
    except Exception:
        pass

    alt_query = _PREFIXES + """
    SELECT ?uri ?label ?kind WHERE {
        GRAPH ?g {
            ?uri a skos:Concept .
            { ?uri skos:altLabel ?label . BIND("alt" AS ?kind) }
            UNION { ?uri skos:hiddenLabel ?label . BIND("hidden" AS ?kind) }
        }
    }
    """
    try:
        for row in store.query(alt_query):
            uri = str(row[0])
            label = _rdf_str(row[1])
            kind = _rdf_str(row[2])
            if uri in records:
                if kind == "alt":
                    records[uri].alt_labels.append(label)
                else:
                    records[uri].hidden_labels.append(label)
    except Exception:
        pass

    return records


def _rdf_str(node: Any) -> str:
    """Extract plain string value from an RDF term.

    For pyoxigraph Literal nodes, returns `.value` (strips language tag / datatype).
    For NamedNode / BlankNode, returns `str()`.
    """
    if hasattr(node, "value"):
        return node.value
    return str(node)


def build_label_uri_map(
    concept_cache: Dict[str, ConceptRecord],
    blacklist: Set[str],
) -> Dict[str, List[str]]:
    """Build normalized-label → [concept_uri] map for O(1) exact lookup."""
    label_map: Dict[str, List[str]] = {}
    for uri, record in concept_cache.items():
        all_labels = [record.pref_label] + record.alt_labels + record.hidden_labels
        for label in all_labels:
            norm = normalize_label(label)
            if norm and norm not in blacklist:
                label_map.setdefault(norm, []).append(uri)
    return label_map


def build_aho_corasick(
    label_uri_map: Dict[str, List[str]],
) -> Any:
    """Build a pyahocorasick Automaton from the label→URI map.

    Returns:
        Finalized ``pyahocorasick.Automaton`` or None if pyahocorasick is not installed.
    """
    try:
        import ahocorasick
    except ImportError:
        return None

    A = ahocorasick.Automaton()
    for label, uris in label_uri_map.items():
        if label:
            A.add_word(label, (label, uris))
    if len(A) > 0:
        A.make_automaton()
    return A


def search_aho_corasick(
    automaton: Any,
    text: str,
    concept_cache: Dict[str, ConceptRecord],
    longest_match_only: bool = False,
) -> List[EntityMention]:
    """Scan text using the Aho-Corasick automaton.

    Args:
        automaton: Finalized ``pyahocorasick.Automaton``.
        text: Input text to scan (should be coref-resolved).
        concept_cache: URI → ConceptRecord cache for pref_label lookup.
        longest_match_only: If True, filter overlapping matches keeping longest.

    Returns:
        List of EntityMention objects sorted by start position.
    """
    if automaton is None:
        return []

    lower_text = text.lower()
    mentions: List[EntityMention] = []

    try:
        for end_idx, (matched_label, uris) in automaton.iter(lower_text):
            start_idx = end_idx - len(matched_label) + 1
            for uri in uris:
                record = concept_cache.get(uri)
                pref_label = record.pref_label if record else matched_label
                mentions.append(
                    EntityMention(
                        text=text[start_idx : end_idx + 1],
                        start=start_idx,
                        end=end_idx + 1,
                        concept_uri=uri,
                        matched_label=matched_label,
                        pref_label=pref_label,
                    )
                )
    except Exception:
        return mentions

    if longest_match_only and mentions:
        mentions = _filter_longest_match(mentions)

    return sorted(mentions, key=lambda m: m.start)


def _filter_longest_match(mentions: List[EntityMention]) -> List[EntityMention]:
    """Keep only the longest non-overlapping match at each position."""
    mentions = sorted(mentions, key=lambda m: (m.start, -(m.end - m.start)))
    result: List[EntityMention] = []
    last_end = -1
    for m in mentions:
        if m.start >= last_end:
            result.append(m)
            last_end = m.end
    return result


def build_ann_index(
    concept_cache: Dict[str, ConceptRecord],
    embedding_fn: Any,
    dim: int = 384,
) -> Tuple[Any, List[str]]:
    """Build an hnswlib ANN index over concept embeddings.

    Args:
        concept_cache: URI → ConceptRecord dict.
        embedding_fn: Callable that takes a list of strings and returns
                      a list of float vectors (e.g. SentenceTransformer.encode).
        dim: Embedding dimensionality.

    Returns:
        Tuple of (hnswlib.Index or None, ordered_uri_list).

    Raises:
        ImportError: If hnswlib is not installed (returns (None, [])).
    """
    try:
        import hnswlib
        import numpy as np
    except ImportError:
        return None, []

    uris = list(concept_cache.keys())
    if not uris:
        return None, []

    texts = [
        f"{r.pref_label}: {r.definition or ''}"
        for r in (concept_cache[u] for u in uris)
    ]

    try:
        embeddings = embedding_fn(texts)
        embeddings = np.array(embeddings, dtype="float32")
    except Exception:
        return None, []

    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=len(uris), ef_construction=200, M=16)
    index.add_items(embeddings, list(range(len(uris))))
    index.set_ef(50)
    return index, uris


def search_ann(
    index: Any,
    uri_list: List[str],
    query: str,
    embedding_fn: Any,
    concept_cache: Dict[str, ConceptRecord],
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Search the ANN index for concepts similar to query.

    Returns:
        List of dicts with concept_uri, pref_label, definition, score.
    """
    if index is None:
        return []

    try:
        import numpy as np
        query_emb = np.array(embedding_fn([query]), dtype="float32")
        labels, distances = index.knn_query(query_emb, k=min(top_k, len(uri_list)))
        results = []
        for label_idx, dist in zip(labels[0], distances[0]):
            uri = uri_list[label_idx]
            record = concept_cache.get(uri)
            results.append(
                {
                    "concept_uri": uri,
                    "pref_label": record.pref_label if record else uri,
                    "definition": record.definition if record else None,
                    "score": float(1.0 - dist),  # cosine distance → similarity
                }
            )
        return results
    except Exception:
        return []
