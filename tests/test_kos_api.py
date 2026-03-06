"""Tests for KOS FastAPI router (spindle/api/kos_router.py)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from spindle.kos.models import (
    ConceptRecord,
    EntityMention,
    ResolutionResult,
    ValidationReport,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_concept(
    uri: str = "http://spindle.dev/ns/concept/pump",
    concept_id: str = "pump-001",
    pref_label: str = "Pump",
) -> ConceptRecord:
    return ConceptRecord(
        uri=uri,
        concept_id=concept_id,
        pref_label=pref_label,
        alt_labels=["Fluid pump"],
        definition="A device to move fluids.",
        provenance_object_id=concept_id,
    )


def _make_mention() -> EntityMention:
    return EntityMention(
        text="pump",
        start=4,
        end=8,
        concept_uri="http://spindle.dev/ns/concept/pump",
        matched_label="pump",
        pref_label="Pump",
    )


# ---------------------------------------------------------------------------
# Fixture: mock KOSService injected at module level
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_svc():
    """Return a MagicMock that stands in for KOSService."""
    svc = MagicMock()
    svc.search_ahocorasick.return_value = [_make_mention()]
    svc.search_ann.return_value = [
        {"concept_uri": "http://spindle.dev/ns/concept/pump", "pref_label": "Pump",
         "definition": "A device.", "score": 0.95}
    ]
    svc.resolve_multistep.return_value = [
        ResolutionResult(
            mention="pump", resolved=True, method="exact_label",
            concept_uri="http://spindle.dev/ns/concept/pump",
            pref_label="Pump", score=1.0,
        )
    ]
    svc.list_concepts.return_value = [_make_concept()]
    svc.get_concept.return_value = _make_concept()
    svc.create_concept.return_value = _make_concept()
    svc.delete_concept.return_value = True
    svc.get_provenance.return_value = {"doc_ids": ["doc-1"], "spans": []}
    svc.get_hierarchy.return_value = {"root": "http://spindle.dev/ns/concept/pump", "descendants": []}
    svc.get_ancestors.return_value = []
    svc.get_descendants.return_value = []
    svc.get_label_set.return_value = ["Pump", "Valve"]
    svc.stats.return_value = {"concepts": 2, "label_map_entries": 3, "ann_index_vectors": 0, "blacklist_size": 3}
    svc.validate_skos.return_value = []
    svc.validate_triples.return_value = ValidationReport(conforms=True)
    svc.sparql.return_value = [{"0": "http://spindle.dev/ns/concept/pump"}]
    svc.reload.return_value = {"status": "ok", "concepts_loaded": 2, "reload_time_ms": 1.0, "ner_automaton_patterns": 3, "search_index_vectors": 0}
    svc.get_rejections.return_value = []
    return svc


@pytest.fixture
def client(mock_svc):
    """TestClient with KOS service mocked out."""
    import spindle.api.kos_router as kr
    original = kr._kos_service
    kr._kos_service = mock_svc
    try:
        from spindle.api.main import app
        with TestClient(app) as c:
            yield c
    finally:
        kr._kos_service = original


# ---------------------------------------------------------------------------
# Search endpoints
# ---------------------------------------------------------------------------

class TestAhocorasickSearch:
    def test_returns_mentions(self, client):
        resp = client.post("/kos/search/ahocorasick", json={"text": "The pump failed."})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["text"] == "pump"
        assert data[0]["start"] == 4

    def test_longest_match_flag(self, client, mock_svc):
        client.post("/kos/search/ahocorasick", json={"text": "pump", "longest_match_only": True})
        mock_svc.search_ahocorasick.assert_called_once_with("pump", longest_match_only=True)


class TestAnnSearch:
    def test_returns_results(self, client):
        resp = client.get("/kos/search/ann", params={"query": "fluid mover", "top_k": 5})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["score"] == pytest.approx(0.95)


class TestMultistepResolve:
    def test_resolved(self, client):
        resp = client.post("/kos/search/multistep", json={"mentions": ["pump"]})
        assert resp.status_code == 200
        data = resp.json()
        assert data[0]["resolved"] is True
        assert data[0]["method"] == "exact_label"


# ---------------------------------------------------------------------------
# Concept CRUD
# ---------------------------------------------------------------------------

class TestConceptList:
    def test_list_returns_concepts(self, client):
        resp = client.get("/kos/concepts")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["pref_label"] == "Pump"

    def test_search_param_passed(self, client, mock_svc):
        client.get("/kos/concepts", params={"search": "pump"})
        mock_svc.list_concepts.assert_called_once_with(limit=100, offset=0, search="pump")


class TestConceptGet:
    def test_found(self, client):
        resp = client.get("/kos/concepts/pump-001")
        assert resp.status_code == 200
        assert resp.json()["concept_id"] == "pump-001"

    def test_not_found(self, client, mock_svc):
        mock_svc.get_concept.return_value = None
        resp = client.get("/kos/concepts/missing")
        assert resp.status_code == 404


class TestConceptCreate:
    def test_creates(self, client):
        resp = client.post("/kos/concepts", json={"pref_label": "Pump"})
        assert resp.status_code == 201
        assert "concept_id" in resp.json()

    def test_service_failure_returns_500(self, client, mock_svc):
        mock_svc.create_concept.return_value = None
        resp = client.post("/kos/concepts", json={"pref_label": "Bad"})
        assert resp.status_code == 500


class TestConceptDelete:
    def test_deleted(self, client):
        resp = client.delete("/kos/concepts/pump-001")
        assert resp.status_code == 204

    def test_not_found(self, client, mock_svc):
        mock_svc.delete_concept.return_value = False
        resp = client.delete("/kos/concepts/missing")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

class TestProvenance:
    def test_found(self, client):
        resp = client.get("/kos/concepts/pump-001/provenance")
        assert resp.status_code == 200
        assert "doc_ids" in resp.json()

    def test_not_found(self, client, mock_svc):
        mock_svc.get_provenance.return_value = None
        resp = client.get("/kos/concepts/missing/provenance")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Hierarchy
# ---------------------------------------------------------------------------

class TestHierarchy:
    def test_get_hierarchy(self, client):
        resp = client.get("/kos/hierarchy/http://spindle.dev/ns/concept/pump")
        assert resp.status_code == 200
        assert "descendants" in resp.json()

    def test_get_ancestors(self, client):
        resp = client.get("/kos/ancestors/http://spindle.dev/ns/concept/pump")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_get_descendants(self, client):
        resp = client.get("/kos/descendants/http://spindle.dev/ns/concept/pump")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Ontology
# ---------------------------------------------------------------------------

class TestOntology:
    def test_get_labels(self, client):
        resp = client.get("/kos/ontology/labels")
        assert resp.status_code == 200
        assert "Pump" in resp.json()

    def test_get_stats(self, client):
        resp = client.get("/kos/ontology/stats")
        assert resp.status_code == 200
        assert resp.json()["concepts"] == 2


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_skos_valid(self, client):
        resp = client.get("/kos/validate/skos")
        assert resp.status_code == 200
        assert resp.json()["conforms"] is True

    def test_validate_triples(self, client):
        resp = client.post("/kos/validate", json={"triples": [{"s": "a", "p": "b", "o": "c"}]})
        assert resp.status_code == 200
        assert resp.json()["conforms"] is True

    def test_validate_violations(self, client, mock_svc):
        mock_svc.validate_triples.return_value = ValidationReport(
            conforms=False, violations=["Missing prefLabel"], message="1 violation"
        )
        resp = client.post("/kos/validate", json={"triples": []})
        assert resp.json()["conforms"] is False
        assert len(resp.json()["violations"]) == 1


# ---------------------------------------------------------------------------
# SPARQL
# ---------------------------------------------------------------------------

class TestSparql:
    def test_select(self, client):
        resp = client.post(
            "/kos/sparql",
            json={"query": "PREFIX skos: <http://www.w3.org/2004/02/skos/core#> SELECT ?c WHERE { ?c a skos:Concept }"},
        )
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------

class TestAdmin:
    def test_reload(self, client):
        resp = client.post("/kos/reload")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_rejections_empty(self, client):
        resp = client.get("/kos/rejections")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_rejections_filter(self, client, mock_svc):
        mock_svc.get_rejections.return_value = [{"id": 1, "rejected_term": "widget"}]
        resp = client.get("/kos/rejections", params={"term": "widget"})
        assert resp.status_code == 200
        mock_svc.get_rejections.assert_called_once_with(term="widget", doc_id=None)
