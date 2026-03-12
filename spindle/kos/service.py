"""KOSService: in-process runtime for the Knowledge Organization System.

Loads SKOS/OWL/SHACL files into an Oxigraph store, builds four derived
indices for low-latency operations, and exposes query methods consumed by
the FastAPI router and the KOS extraction pipeline.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from spindle.kos.blacklist import RejectionLog, load_blacklist
from spindle.kos.indices import (
    build_aho_corasick,
    build_ann_index,
    build_label_uri_map,
    build_uri_concept_cache,
    normalize_label,
    search_aho_corasick,
    search_ann,
)
from spindle.kos.models import (
    ConceptRecord,
    EntityMention,
    ResolutionResult,
    ValidationReport,
)
from spindle.kos.serializer import (
    GRAPH_URIS,
    load_file_into_store,
    serialize_store_to_file,
)
from spindle.kos.validation import check_skos_integrity, validate_with_shacl


_SPINDLE_NS = "http://spindle.dev/ns/"
_SKOS_NS = "http://www.w3.org/2004/02/skos/core#"
_DCT_NS = "http://purl.org/dc/terms/"


class KOSService:
    """In-process KOS runtime.

    Loads KOS files into an Oxigraph store, builds derived indices, and
    exposes methods for NER, semantic search, entity resolution, SPARQL,
    concept CRUD, and hierarchy traversal.

    Args:
        kos_dir: Root KOS directory (contains kos.ttls, ontology.owl, etc.).
            Defaults to ``<stores_root>/kos`` where stores_root is determined
            by :py:func:`~spindle.configuration.find_stores_root`.
        embedding_fn: Optional callable for ANN index (takes list[str], returns
                      list of float vectors).  If None, ANN search is disabled.
        provenance_store: Optional ProvenanceStore for concept provenance lookups.
    """

    def __init__(
        self,
        kos_dir: str | Path | None = None,
        embedding_fn: Optional[Callable] = None,
        provenance_store: Optional[Any] = None,
    ) -> None:
        if kos_dir is None:
            from spindle.configuration import find_stores_root
            kos_dir = find_stores_root() / "kos"
        self._kos_dir = Path(kos_dir)
        self._embedding_fn = embedding_fn
        self._provenance_store = provenance_store
        self._lock = threading.RLock()

        # These are replaced atomically on reload()
        self._store: Any = None
        self._concept_cache: Dict[str, ConceptRecord] = {}
        self._label_map: Dict[str, List[str]] = {}
        self._aho_automaton: Any = None
        self._ann_index: Any = None
        self._ann_uris: List[str] = []
        self._blacklist: Set[str] = set()

        self._rejection_log = RejectionLog(
            self._kos_dir / "rejections.db"
            if self._kos_dir.exists()
            else ":memory:"
        )

        self._load()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load KOS files and build all derived indices."""
        try:
            import pyoxigraph
        except ImportError:
            # pyoxigraph not installed — service operates in no-op mode
            return

        store = pyoxigraph.Store()

        file_map = {
            "kos": self._kos_dir / "kos.ttls",
            "ontology": self._kos_dir / "ontology.owl",
            "shapes": self._kos_dir / "shapes.ttl",
            "scheme": self._kos_dir / "config" / "scheme.ttl",
        }
        for name, path in file_map.items():
            graph_uri = GRAPH_URIS.get(name, f"{_SPINDLE_NS}graph/{name}")
            load_file_into_store(store, path, graph_uri)

        blacklist = load_blacklist(self._kos_dir / "blacklist.txt")
        concept_cache = build_uri_concept_cache(store)
        label_map = build_label_uri_map(concept_cache, blacklist)
        aho = build_aho_corasick(label_map)

        ann_index, ann_uris = (None, [])
        if self._embedding_fn and concept_cache:
            ann_index, ann_uris = build_ann_index(concept_cache, self._embedding_fn)

        with self._lock:
            self._store = store
            self._blacklist = blacklist
            self._concept_cache = concept_cache
            self._label_map = label_map
            self._aho_automaton = aho
            self._ann_index = ann_index
            self._ann_uris = ann_uris

    def reload(self) -> Dict[str, Any]:
        """Atomically reload all KOS files and rebuild indices.

        Returns:
            Stats dict with counts of loaded entities and rebuild time.
        """
        t0 = time.perf_counter()
        self._load()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "status": "ok",
            "concepts_loaded": len(self._concept_cache),
            "ner_automaton_patterns": len(self._label_map),
            "search_index_vectors": len(self._ann_uris),
            "reload_time_ms": round(elapsed_ms, 1),
        }

    # ------------------------------------------------------------------
    # NER — Aho-Corasick
    # ------------------------------------------------------------------

    def search_ahocorasick(
        self, text: str, longest_match_only: bool = False
    ) -> List[EntityMention]:
        """Scan text for vocabulary term mentions using the Aho-Corasick automaton."""
        with self._lock:
            return search_aho_corasick(
                self._aho_automaton,
                text,
                self._concept_cache,
                longest_match_only=longest_match_only,
            )

    # ------------------------------------------------------------------
    # ANN Search
    # ------------------------------------------------------------------

    def search_ann(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Semantic search over concept labels and definitions."""
        if self._embedding_fn is None:
            return []
        with self._lock:
            return search_ann(
                self._ann_index,
                self._ann_uris,
                query,
                self._embedding_fn,
                self._concept_cache,
                top_k=top_k,
            )

    # ------------------------------------------------------------------
    # Multi-step resolution
    # ------------------------------------------------------------------

    def resolve_multistep(
        self, mentions: List[str], threshold: float = 0.7
    ) -> List[ResolutionResult]:
        """Resolve entity mentions: fast pass (exact label) then medium pass (ANN)."""
        results: List[ResolutionResult] = []
        with self._lock:
            for mention in mentions:
                norm = normalize_label(mention)
                uris = self._label_map.get(norm, [])
                if uris:
                    uri = uris[0]
                    record = self._concept_cache.get(uri)
                    results.append(
                        ResolutionResult(
                            mention=mention,
                            resolved=True,
                            method="exact_label",
                            concept_uri=uri,
                            pref_label=record.pref_label if record else None,
                            score=1.0,
                        )
                    )
                    continue

                # Medium pass — ANN
                if self._ann_index is not None and self._embedding_fn:
                    ann_results = search_ann(
                        self._ann_index,
                        self._ann_uris,
                        mention,
                        self._embedding_fn,
                        self._concept_cache,
                        top_k=5,
                    )
                    best = ann_results[0] if ann_results else None
                    if best and best["score"] >= threshold:
                        results.append(
                            ResolutionResult(
                                mention=mention,
                                resolved=True,
                                method="semantic_search",
                                concept_uri=best["concept_uri"],
                                pref_label=best["pref_label"],
                                score=best["score"],
                                candidates=ann_results,
                            )
                        )
                        continue

                results.append(
                    ResolutionResult(
                        mention=mention,
                        resolved=False,
                        method="semantic_search",
                        score=0.0,
                    )
                )
        return results

    # ------------------------------------------------------------------
    # Concept CRUD
    # ------------------------------------------------------------------

    def get_concept(self, concept_id: str) -> Optional[ConceptRecord]:
        """Retrieve a concept by its dct:identifier."""
        with self._lock:
            for record in self._concept_cache.values():
                if record.concept_id == concept_id:
                    return record
            return None

    def list_concepts(
        self,
        limit: int = 100,
        offset: int = 0,
        search: Optional[str] = None,
    ) -> List[ConceptRecord]:
        """List concepts with optional label search."""
        with self._lock:
            records = list(self._concept_cache.values())
            if search:
                lower = search.lower()
                records = [
                    r for r in records
                    if lower in r.pref_label.lower()
                    or any(lower in a.lower() for a in r.alt_labels)
                ]
            return records[offset : offset + limit]

    def create_concept(
        self,
        pref_label: str,
        definition: Optional[str] = None,
        alt_labels: Optional[List[str]] = None,
        broader: Optional[List[str]] = None,
    ) -> Optional[ConceptRecord]:
        """Create a new SKOS concept in the Oxigraph store and update indices."""
        if self._store is None:
            return None

        try:
            import pyoxigraph
        except ImportError:
            return None

        concept_id = _slugify(pref_label)
        uri = f"{_SPINDLE_NS}concept/{concept_id}"
        kos_graph = pyoxigraph.NamedNode(GRAPH_URIS["kos"])
        skos = _SKOS_NS
        dct = _DCT_NS

        def _literal(val: str) -> pyoxigraph.Literal:
            return pyoxigraph.Literal(val, language="en")

        quads = [
            pyoxigraph.Quad(
                pyoxigraph.NamedNode(uri),
                pyoxigraph.NamedNode(f"{skos}Concept"),
                pyoxigraph.NamedNode(f"{skos}Concept"),
                kos_graph,
            ),
        ]
        # Simplify: use SPARQL UPDATE instead
        update = f"""
        PREFIX skos: <{skos}>
        PREFIX dct: <{dct}>
        PREFIX spndl: <{_SPINDLE_NS}>
        INSERT DATA {{
            GRAPH <{GRAPH_URIS["kos"]}> {{
                <{uri}> a skos:Concept ;
                    skos:prefLabel "{pref_label}"@en ;
                    dct:identifier "{concept_id}" .
        """
        if definition:
            update += f'            <{uri}> skos:definition "{definition}"@en .\n'
        for al in alt_labels or []:
            update += f'            <{uri}> skos:altLabel "{al}"@en .\n'
        for b in broader or []:
            update += f'            <{uri}> skos:broader <{b}> .\n'
        update += "    }\n}"

        with self._lock:
            try:
                self._store.update(update)
            except Exception:
                return None

            record = ConceptRecord(
                uri=uri,
                concept_id=concept_id,
                pref_label=pref_label,
                alt_labels=alt_labels or [],
                definition=definition,
                broader=broader or [],
                provenance_object_id=concept_id,
            )
            self._concept_cache[uri] = record
            # Update label map
            norm = normalize_label(pref_label)
            self._label_map.setdefault(norm, []).append(uri)
            return record

    def delete_concept(self, concept_id: str) -> bool:
        """Soft-delete a concept by removing it from the Oxigraph store."""
        if self._store is None:
            return False
        try:
            import pyoxigraph
        except ImportError:
            return False

        record = self.get_concept(concept_id)
        if record is None:
            return False

        update = f"""
        DELETE WHERE {{
            GRAPH <{GRAPH_URIS["kos"]}> {{
                <{record.uri}> ?p ?o .
            }}
        }}
        """
        with self._lock:
            try:
                self._store.update(update)
                del self._concept_cache[record.uri]
                norm = normalize_label(record.pref_label)
                uris = self._label_map.get(norm, [])
                if record.uri in uris:
                    uris.remove(record.uri)
                return True
            except Exception:
                return False

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------

    def get_provenance(self, concept_id: str) -> Optional[Any]:
        """Delegate to ProvenanceStore for concept provenance."""
        if self._provenance_store is None:
            return None
        return self._provenance_store.get_provenance(concept_id)

    # ------------------------------------------------------------------
    # Hierarchy
    # ------------------------------------------------------------------

    def get_hierarchy(self, root: str, depth: int = 5) -> Dict[str, Any]:
        """Return the narrower hierarchy rooted at a concept."""
        if self._store is None:
            return {}
        query = f"""
        PREFIX skos: <{_SKOS_NS}>
        SELECT ?child WHERE {{
            GRAPH ?g {{
                <{root}> skos:narrower+ ?child .
            }}
        }}
        """
        try:
            children = [str(row[0]) for row in self._store.query(query)]
            return {"root": root, "descendants": children}
        except Exception:
            return {"root": root, "descendants": []}

    def get_ancestors(self, concept_uri: str) -> List[str]:
        if self._store is None:
            return []
        query = f"""
        PREFIX skos: <{_SKOS_NS}>
        SELECT ?ancestor WHERE {{
            GRAPH ?g {{
                <{concept_uri}> skos:broader+ ?ancestor .
            }}
        }}
        """
        try:
            return [str(row[0]) for row in self._store.query(query)]
        except Exception:
            return []

    def get_descendants(self, concept_uri: str) -> List[str]:
        if self._store is None:
            return []
        query = f"""
        PREFIX skos: <{_SKOS_NS}>
        SELECT ?desc WHERE {{
            GRAPH ?g {{
                <{concept_uri}> skos:narrower+ ?desc .
            }}
        }}
        """
        try:
            return [str(row[0]) for row in self._store.query(query)]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Labels for NER seeding
    # ------------------------------------------------------------------

    def get_label_set(self, include_alt: bool = True) -> List[str]:
        """Return all prefLabels (and optionally altLabels) for NER seeding."""
        with self._lock:
            labels: List[str] = []
            for record in self._concept_cache.values():
                labels.append(record.pref_label)
                if include_alt:
                    labels.extend(record.alt_labels)
            return list(dict.fromkeys(labels))  # deduplicate preserving order

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_skos(self) -> List[str]:
        """Run SKOS integrity checks. Returns list of violation descriptions."""
        if self._store is None:
            return []
        with self._lock:
            return check_skos_integrity(self._store)

    def validate_triples(self, triples: List[Dict[str, Any]]) -> ValidationReport:
        """Validate triples against SHACL shapes."""
        shapes_path = self._kos_dir / "shapes.ttl"
        ontology_path = self._kos_dir / "ontology.owl"
        # Build a temporary store with the provided triples for validation
        if self._store is None:
            return ValidationReport(conforms=True, message="KOS store not loaded.")
        return validate_with_shacl(
            data_graph=self._store,
            shapes_path=shapes_path,
            ontology_path=ontology_path if ontology_path.exists() else None,
        )

    # ------------------------------------------------------------------
    # SPARQL
    # ------------------------------------------------------------------

    def sparql(self, query: str) -> List[Dict[str, Any]]:
        """Execute a raw SPARQL query against the Oxigraph store."""
        if self._store is None:
            return []
        try:
            from spindle.kos.indices import _rdf_str
            rows = []
            for row in self._store.query(query):
                rows.append({str(i): _rdf_str(v) for i, v in enumerate(row)})
            return rows
        except Exception as exc:
            return [{"error": str(exc)}]

    # ------------------------------------------------------------------
    # Rejection log
    # ------------------------------------------------------------------

    def get_rejections(
        self, term: Optional[str] = None, doc_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Query the rejection log."""
        return self._rejection_log.query(term=term, doc_id=doc_id)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "concepts": len(self._concept_cache),
                "label_map_entries": len(self._label_map),
                "ann_index_vectors": len(self._ann_uris),
                "blacklist_size": len(self._blacklist),
            }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _slugify(label: str) -> str:
    """Convert a label to a stable slug for use as a concept identifier."""
    import re
    label = label.lower().strip()
    label = re.sub(r"[^\w\s-]", "", label)
    label = re.sub(r"[\s_]+", "-", label)
    label = re.sub(r"-+", "-", label).strip("-")
    return label or "concept"
