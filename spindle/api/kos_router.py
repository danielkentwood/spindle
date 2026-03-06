"""KOS FastAPI router — /kos endpoint group.

All endpoints delegate to a module-level KOSService singleton that is
initialised once (lazy) and swapped atomically on ``POST /kos/reload``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# KOSService singleton
# ---------------------------------------------------------------------------

_kos_service: Any = None  # KOSService | None
_KOS_DIR: Path = Path("kos")


def _get_kos() -> Any:
    """Return the module-level KOSService, initialising it on first call."""
    global _kos_service
    if _kos_service is None:
        try:
            from spindle.kos.service import KOSService
            _kos_service = KOSService(kos_dir=_KOS_DIR)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"KOS service could not be initialised: {exc}",
            )
    return _kos_service


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class AhocorasickRequest(BaseModel):
    text: str
    longest_match_only: bool = False


class AnnSearchRequest(BaseModel):
    query: str
    top_k: int = 10


class MultistepRequest(BaseModel):
    mentions: List[str]
    threshold: float = 0.7


class ConceptCreate(BaseModel):
    pref_label: str
    definition: Optional[str] = None
    alt_labels: Optional[List[str]] = None
    broader: Optional[List[str]] = None


class ValidateRequest(BaseModel):
    triples: List[Dict[str, Any]]


class SparqlRequest(BaseModel):
    query: str


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter()


# ── Search ──────────────────────────────────────────────────────────────────

@router.post("/search/ahocorasick", tags=["KOS Search"])
def ner_scan(req: AhocorasickRequest) -> List[Dict[str, Any]]:
    """NER scan using the Aho-Corasick automaton."""
    svc = _get_kos()
    mentions = svc.search_ahocorasick(req.text, longest_match_only=req.longest_match_only)
    return [
        {
            "text": m.text,
            "start": m.start,
            "end": m.end,
            "concept_uri": m.concept_uri,
            "matched_label": m.matched_label,
            "pref_label": m.pref_label,
        }
        for m in mentions
    ]


@router.get("/search/ann", tags=["KOS Search"])
def ann_search(
    query: str = Query(..., description="Search query string"),
    top_k: int = Query(10, ge=1, le=100),
) -> List[Dict[str, Any]]:
    """Semantic ANN search over concept embeddings."""
    svc = _get_kos()
    return svc.search_ann(query, top_k=top_k)


@router.post("/search/multistep", tags=["KOS Search"])
def multistep_resolve(req: MultistepRequest) -> List[Dict[str, Any]]:
    """Multi-step resolution: exact label → ANN fallback."""
    svc = _get_kos()
    results = svc.resolve_multistep(req.mentions, threshold=req.threshold)
    return [
        {
            "mention": r.mention,
            "resolved": r.resolved,
            "method": r.method,
            "concept_uri": r.concept_uri,
            "pref_label": r.pref_label,
            "score": r.score,
            "candidates": r.candidates or [],
        }
        for r in results
    ]


# ── Concept CRUD ─────────────────────────────────────────────────────────────

@router.get("/concepts", tags=["KOS Concepts"])
def list_concepts(
    search: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> List[Dict[str, Any]]:
    """List concepts with optional label search and pagination."""
    svc = _get_kos()
    records = svc.list_concepts(limit=limit, offset=offset, search=search)
    return [
        {
            "uri": r.uri,
            "concept_id": r.concept_id,
            "pref_label": r.pref_label,
            "alt_labels": r.alt_labels,
            "definition": r.definition,
            "broader": r.broader,
            "narrower": r.narrower,
            "related": r.related,
        }
        for r in records
    ]


@router.get("/concepts/{concept_id}", tags=["KOS Concepts"])
def get_concept(concept_id: str) -> Dict[str, Any]:
    """Retrieve a concept by its dct:identifier."""
    svc = _get_kos()
    record = svc.get_concept(concept_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Concept '{concept_id}' not found",
        )
    return {
        "uri": record.uri,
        "concept_id": record.concept_id,
        "pref_label": record.pref_label,
        "alt_labels": record.alt_labels,
        "definition": record.definition,
        "scope_note": record.scope_note,
        "broader": record.broader,
        "narrower": record.narrower,
        "related": record.related,
    }


@router.post("/concepts", status_code=status.HTTP_201_CREATED, tags=["KOS Concepts"])
def create_concept(req: ConceptCreate) -> Dict[str, Any]:
    """Create a new SKOS concept."""
    svc = _get_kos()
    record = svc.create_concept(
        pref_label=req.pref_label,
        definition=req.definition,
        alt_labels=req.alt_labels,
        broader=req.broader,
    )
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create concept",
        )
    return {
        "uri": record.uri,
        "concept_id": record.concept_id,
        "pref_label": record.pref_label,
    }


@router.delete("/concepts/{concept_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["KOS Concepts"])
def delete_concept(concept_id: str) -> None:
    """Remove a concept from the KOS."""
    svc = _get_kos()
    deleted = svc.delete_concept(concept_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Concept '{concept_id}' not found or could not be deleted",
        )


# ── Provenance ──────────────────────────────────────────────────────────────

@router.get("/concepts/{concept_id}/provenance", tags=["KOS Provenance"])
def get_concept_provenance(concept_id: str) -> Dict[str, Any]:
    """Retrieve provenance data for a concept."""
    svc = _get_kos()
    prov = svc.get_provenance(concept_id)
    if prov is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No provenance found for concept '{concept_id}'",
        )
    return prov if isinstance(prov, dict) else {"provenance": str(prov)}


# ── Hierarchy ────────────────────────────────────────────────────────────────

@router.get("/hierarchy/{concept_uri:path}", tags=["KOS Hierarchy"])
def get_hierarchy(
    concept_uri: str,
    depth: int = Query(5, ge=1, le=10),
) -> Dict[str, Any]:
    """Return the narrower hierarchy rooted at a concept URI."""
    svc = _get_kos()
    return svc.get_hierarchy(concept_uri, depth=depth)


@router.get("/ancestors/{concept_uri:path}", tags=["KOS Hierarchy"])
def get_ancestors(concept_uri: str) -> List[str]:
    """Return all broader ancestors of a concept."""
    svc = _get_kos()
    return svc.get_ancestors(concept_uri)


@router.get("/descendants/{concept_uri:path}", tags=["KOS Hierarchy"])
def get_descendants(concept_uri: str) -> List[str]:
    """Return all narrower descendants of a concept."""
    svc = _get_kos()
    return svc.get_descendants(concept_uri)


# ── Ontology ─────────────────────────────────────────────────────────────────

@router.get("/ontology/labels", tags=["KOS Ontology"])
def get_label_set(include_alt: bool = True) -> List[str]:
    """Return all prefLabels (and optionally altLabels) for NER seeding."""
    svc = _get_kos()
    return svc.get_label_set(include_alt=include_alt)


@router.get("/ontology/stats", tags=["KOS Ontology"])
def get_stats() -> Dict[str, Any]:
    """Return KOS service stats."""
    svc = _get_kos()
    return svc.stats()


# ── Validation ───────────────────────────────────────────────────────────────

@router.get("/validate/skos", tags=["KOS Validation"])
def validate_skos() -> Dict[str, Any]:
    """Run SKOS integrity checks."""
    svc = _get_kos()
    violations = svc.validate_skos()
    return {"conforms": len(violations) == 0, "violations": violations}


@router.post("/validate", tags=["KOS Validation"])
def validate_triples(req: ValidateRequest) -> Dict[str, Any]:
    """Validate triples against SHACL shapes."""
    svc = _get_kos()
    report = svc.validate_triples(req.triples)
    return {
        "conforms": report.conforms,
        "violations": report.violations,
        "message": report.message,
    }


# ── SPARQL ───────────────────────────────────────────────────────────────────

@router.post("/sparql", tags=["KOS SPARQL"])
def sparql_query(req: SparqlRequest) -> List[Dict[str, Any]]:
    """Execute a raw SPARQL SELECT query against the Oxigraph store."""
    svc = _get_kos()
    return svc.sparql(req.query)


# ── Reload ───────────────────────────────────────────────────────────────────

@router.post("/reload", tags=["KOS Admin"])
def reload_kos() -> Dict[str, Any]:
    """Atomically reload all KOS files and rebuild indices."""
    svc = _get_kos()
    return svc.reload()


# ── Rejection log ────────────────────────────────────────────────────────────

@router.get("/rejections", tags=["KOS Admin"])
def get_rejections(
    term: Optional[str] = None,
    doc_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Query the staging rejection log."""
    svc = _get_kos()
    return svc.get_rejections(term=term, doc_id=doc_id)
