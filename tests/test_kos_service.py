"""Tests for spindle.kos — KOSService, indices, blacklist, validation."""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from spindle.kos.blacklist import RejectionLog, load_blacklist
from spindle.kos.indices import (
    build_label_uri_map,
    build_uri_concept_cache,
    normalize_label,
    search_aho_corasick,
    build_aho_corasick,
)
from spindle.kos.models import ConceptRecord, EntityMention, ResolutionResult
from spindle.kos.service import KOSService, _slugify
from spindle.kos.validation import check_skos_integrity


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_concept(
    uri: str = "http://spindle.dev/ns/concept/pump",
    concept_id: str = "pump-001",
    pref_label: str = "Pump",
    alt_labels: list | None = None,
    definition: str | None = "A device to move fluids.",
) -> ConceptRecord:
    return ConceptRecord(
        uri=uri,
        concept_id=concept_id,
        pref_label=pref_label,
        alt_labels=alt_labels or ["Fluid pump"],
        definition=definition,
        provenance_object_id=concept_id,
    )


# ── normalize_label ────────────────────────────────────────────────────────────

class TestNormalizeLabel:
    def test_lowercase(self):
        assert normalize_label("Pump") == "pump"

    def test_collapses_whitespace(self):
        assert normalize_label("  Centrifugal  Pump  ") == "centrifugal pump"

    def test_strips_punctuation(self):
        assert normalize_label("pump!") == "pump"

    def test_empty_string(self):
        assert normalize_label("") == ""


# ── label_uri_map ─────────────────────────────────────────────────────────────

class TestBuildLabelUriMap:
    def test_basic(self):
        cache = {
            "uri:pump": _make_concept(pref_label="Pump", alt_labels=["Fluid pump"]),
        }
        label_map = build_label_uri_map(cache, blacklist=set())
        assert "pump" in label_map
        assert "fluid pump" in label_map
        assert "uri:pump" in label_map["pump"]

    def test_blacklist_excludes_labels(self):
        cache = {
            "uri:the": _make_concept(uri="uri:the", concept_id="the", pref_label="the", alt_labels=[]),
        }
        label_map = build_label_uri_map(cache, blacklist={"the"})
        assert "the" not in label_map


# ── Aho-Corasick ──────────────────────────────────────────────────────────────

class TestAhoCorasick:
    @pytest.fixture
    def setup(self):
        cache = {
            "uri:pump": _make_concept(pref_label="Pump", alt_labels=["fluid pump"]),
            "uri:valve": _make_concept(
                uri="uri:valve", concept_id="valve-001",
                pref_label="Valve", alt_labels=[]
            ),
        }
        label_map = build_label_uri_map(cache, blacklist=set())
        return cache, label_map

    def test_returns_empty_when_pyahocorasick_missing(self, setup):
        cache, label_map = setup
        with patch.dict("sys.modules", {"ahocorasick": None}):
            aho = build_aho_corasick(label_map)
        assert aho is None
        result = search_aho_corasick(None, "The pump is broken.", cache)
        assert result == []

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("ahocorasick"),
        reason="pyahocorasick not installed",
    )
    def test_finds_mentions(self, setup):
        cache, label_map = setup
        aho = build_aho_corasick(label_map)
        if aho is None:
            pytest.skip("pyahocorasick not installed")
        mentions = search_aho_corasick(aho, "The pump is broken near the valve.", cache)
        texts = {m.text.lower() for m in mentions}
        assert "pump" in texts or "valve" in texts


# ── KOSService default path ───────────────────────────────────────────────────

class TestKOSServiceDefaultPath:
    """KOSService auto-resolves kos_dir from the stores root when none is given."""

    def test_default_kos_dir_uses_stores_root(self, tmp_path):
        mock_stores_root = tmp_path / "stores"
        expected_kos_dir = mock_stores_root / "kos"

        with patch("spindle.configuration.find_stores_root", return_value=mock_stores_root):
            with patch.dict("sys.modules", {"pyoxigraph": None}):
                svc = KOSService()

        assert svc._kos_dir == expected_kos_dir

    def test_explicit_kos_dir_overrides_default(self, tmp_path):
        explicit_dir = tmp_path / "custom_kos"
        # find_stores_root should NOT be called when kos_dir is explicit
        with patch("spindle.configuration.find_stores_root") as mock_find:
            with patch.dict("sys.modules", {"pyoxigraph": None}):
                svc = KOSService(kos_dir=explicit_dir)
        mock_find.assert_not_called()
        assert svc._kos_dir == explicit_dir


# ── KOSService (no pyoxigraph) ────────────────────────────────────────────────

class TestKOSServiceNoOxigraph:
    """KOSService works in no-op mode when pyoxigraph is not installed."""

    @pytest.fixture
    def svc(self, tmp_path):
        with patch.dict("sys.modules", {"pyoxigraph": None}):
            return KOSService(kos_dir=tmp_path)

    def test_get_concept_returns_none(self, svc):
        assert svc.get_concept("nonexistent") is None

    def test_list_concepts_empty(self, svc):
        assert svc.list_concepts() == []

    def test_stats(self, svc):
        stats = svc.stats()
        assert "concepts" in stats
        assert stats["concepts"] == 0

    def test_sparql_returns_empty(self, svc):
        assert svc.sparql("SELECT * WHERE { ?s ?p ?o }") == []

    def test_search_ahocorasick_empty(self, svc):
        assert svc.search_ahocorasick("text") == []

    def test_get_label_set_empty(self, svc):
        assert svc.get_label_set() == []

    def test_resolve_multistep_unresolved(self, svc):
        results = svc.resolve_multistep(["unknown term"])
        assert len(results) == 1
        assert results[0].resolved is False


class TestKOSServiceWithFixtureKOS:
    """KOSService with real KOS fixture files (requires pyoxigraph)."""

    @pytest.fixture
    def kos_dir(self, tmp_path):
        """Create a minimal KOS fixture directory."""
        kos = tmp_path / "kos"
        kos.mkdir()
        (kos / "config").mkdir()
        (kos / "staging").mkdir()

        # Use full angle-bracket IRIs to avoid slash-in-local-name issues in strict parsers
        _NS = "http://spindle.dev/ns/"
        (kos / "blacklist.txt").write_text("the\na\nan\n")
        (kos / "config" / "scheme.ttl").write_text(
            "@prefix skos: <http://www.w3.org/2004/02/skos/core#> .\n"
            f"<{_NS}scheme/main> a skos:ConceptScheme .\n"
        )
        (kos / "shapes.ttl").write_text(
            "@prefix sh: <http://www.w3.org/ns/shacl#> .\n"
        )
        (kos / "ontology.owl").write_text(
            "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n"
            f"<{_NS}ontology> a owl:Ontology .\n"
        )
        (kos / "kos.ttls").write_text(
            "@prefix skos: <http://www.w3.org/2004/02/skos/core#> .\n"
            "@prefix dct: <http://purl.org/dc/terms/> .\n\n"
            f"<{_NS}concept/pump> a skos:Concept ;\n"
            "    skos:prefLabel \"Pump\"@en ;\n"
            "    skos:altLabel \"Fluid pump\"@en ;\n"
            "    skos:definition \"A device to move fluids.\"@en ;\n"
            "    dct:identifier \"pump-001\" ;\n"
            f"    skos:inScheme <{_NS}scheme/main> .\n\n"
            f"<{_NS}concept/valve> a skos:Concept ;\n"
            "    skos:prefLabel \"Valve\"@en ;\n"
            "    skos:definition \"A flow control device.\"@en ;\n"
            "    dct:identifier \"valve-001\" ;\n"
            f"    skos:inScheme <{_NS}scheme/main> .\n\n"
            f"<{_NS}concept/pump> skos:related <{_NS}concept/valve> .\n"
            f"<{_NS}concept/valve> skos:related <{_NS}concept/pump> .\n"
        )
        return kos

    @pytest.fixture
    def svc(self, kos_dir):
        try:
            import pyoxigraph
        except ImportError:
            pytest.skip("pyoxigraph not installed")
        return KOSService(kos_dir=kos_dir)

    def test_concepts_loaded(self, svc):
        assert svc.stats()["concepts"] >= 2

    def test_get_concept_by_id(self, svc):
        record = svc.get_concept("pump-001")
        assert record is not None
        assert record.pref_label == "Pump"

    def test_list_concepts(self, svc):
        records = svc.list_concepts()
        labels = {r.pref_label for r in records}
        assert "Pump" in labels
        assert "Valve" in labels

    def test_list_concepts_search(self, svc):
        results = svc.list_concepts(search="pump")
        assert all("pump" in r.pref_label.lower() for r in results)

    def test_get_label_set(self, svc):
        labels = svc.get_label_set()
        assert "Pump" in labels

    def test_label_set_includes_alt(self, svc):
        labels = svc.get_label_set(include_alt=True)
        assert "Fluid pump" in labels

    def test_resolve_multistep_exact(self, svc):
        results = svc.resolve_multistep(["pump"])
        assert results[0].resolved is True
        assert results[0].method == "exact_label"

    def test_resolve_multistep_unresolved(self, svc):
        results = svc.resolve_multistep(["xyzzy-machine"])
        assert results[0].resolved is False

    def test_validate_skos_clean(self, svc):
        violations = svc.validate_skos()
        assert violations == []

    def test_sparql_select(self, svc):
        rows = svc.sparql(
            "PREFIX skos: <http://www.w3.org/2004/02/skos/core#> "
            "SELECT ?c WHERE { ?c a skos:Concept } LIMIT 5"
        )
        assert isinstance(rows, list)

    def test_reload(self, svc):
        stats = svc.reload()
        assert stats["status"] == "ok"
        assert "concepts_loaded" in stats

    def test_create_concept(self, svc):
        record = svc.create_concept(
            pref_label="Motor",
            definition="An electric motor.",
        )
        assert record is not None
        assert record.pref_label == "Motor"
        assert svc.get_concept("motor") is not None

    def test_delete_concept(self, svc):
        svc.create_concept(pref_label="TempConcept")
        assert svc.get_concept("tempconcept") is not None
        result = svc.delete_concept("tempconcept")
        assert result is True
        assert svc.get_concept("tempconcept") is None


# ── Blacklist ─────────────────────────────────────────────────────────────────

class TestBlacklist:
    def test_load_blacklist(self, tmp_path):
        bl = tmp_path / "blacklist.txt"
        bl.write_text("# comment\nthe\na\nan\n")
        result = load_blacklist(bl)
        assert result == {"the", "a", "an"}

    def test_missing_file_returns_empty(self, tmp_path):
        assert load_blacklist(tmp_path / "missing.txt") == set()


class TestRejectionLog:
    @pytest.fixture
    def log(self):
        with RejectionLog(":memory:") as l:
            yield l

    def test_add_and_query(self, log):
        log.add("widget", "doc-1", chunk_index=3, rejection_reason="Too generic")
        results = log.query(term="widget")
        assert len(results) == 1
        assert results[0]["rejected_term"] == "widget"

    def test_query_by_doc(self, log):
        log.add("widget", "doc-1")
        log.add("gadget", "doc-2")
        results = log.query(doc_id="doc-1")
        assert len(results) == 1
        assert results[0]["source_doc_id"] == "doc-1"

    def test_query_empty(self, log):
        assert log.query() == []


# ── _slugify ─────────────────────────────────────────────────────────────────

class TestSlugify:
    def test_basic(self):
        assert _slugify("Centrifugal Pump") == "centrifugal-pump"

    def test_special_chars(self):
        assert _slugify("pump (v2)") == "pump-v2"

    def test_empty(self):
        assert _slugify("") == "concept"


# ── SKOS integrity ────────────────────────────────────────────────────────────

class TestSkosIntegrity:
    def test_no_violations_empty_store(self):
        try:
            import pyoxigraph
        except ImportError:
            pytest.skip("pyoxigraph not installed")
        store = pyoxigraph.Store()
        violations = check_skos_integrity(store)
        assert violations == []
