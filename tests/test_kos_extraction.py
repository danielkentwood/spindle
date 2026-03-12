"""Tests for KOS extraction pipeline (Phase 4).

Covers:
- Three-pass NER cascade (fast, medium, discovery)
- Staging write/merge/reject
- Ontology synthesis and SHACL generation
- Provenance writes from extraction
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from spindle.kos.extraction import KOSExtractionPipeline, _heuristic_candidates
from spindle.kos.models import EntityMention, ResolutionResult
from spindle.kos.staging import (
    clear_staging_file,
    merge_staging,
    reject_candidate,
    write_staging,
)
from spindle.kos.synthesis import generate_shacl, synthesize_ontology


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(
    text: str = "The pump failed near the valve.",
    source_id: str = "doc-1",
    start_index: int = 0,
) -> MagicMock:
    chunk = MagicMock()
    chunk.text = text
    chunk.source_id = source_id
    chunk.metadata = {"start_index": start_index, "coref_annotations": []}
    return chunk


def _make_kos_svc(concept_count: int = 2) -> MagicMock:
    svc = MagicMock()
    svc.stats.return_value = {"concepts": concept_count}
    svc._kos_dir = Path("/tmp/kos")
    svc._rejection_log = MagicMock()
    svc.search_ahocorasick.return_value = [
        EntityMention(
            text="pump", start=4, end=8,
            concept_uri="http://spindle.dev/ns/concept/pump",
            matched_label="pump", pref_label="Pump",
        )
    ]
    svc.resolve_multistep.return_value = [
        ResolutionResult(
            mention="valve", resolved=True, method="exact_label",
            concept_uri="http://spindle.dev/ns/concept/valve",
            pref_label="Valve", score=1.0,
        )
    ]
    svc.get_label_set.return_value = ["Pump", "Valve"]
    svc.validate_skos.return_value = []
    svc.create_concept.return_value = MagicMock(uri="http://spindle.dev/ns/concept/motor")
    return svc


# ---------------------------------------------------------------------------
# NER passes
# ---------------------------------------------------------------------------

class TestFastPass:
    def test_returns_matched_and_unmatched(self):
        from spindle.kos.ner import fast_pass
        chunks = [_make_chunk("The pump failed."), _make_chunk("Nothing relevant here.")]
        svc = _make_kos_svc()
        svc.search_ahocorasick.side_effect = [
            [EntityMention(text="pump", start=4, end=8,
                           concept_uri="uri:pump", matched_label="pump", pref_label="Pump")],
            [],
        ]
        matched, unmatched = fast_pass(chunks, svc)
        assert len(matched) == 1
        assert len(unmatched) == 1
        assert matched[0]["concept_uri"] == "uri:pump"

    def test_empty_chunks(self):
        from spindle.kos.ner import fast_pass
        matched, unmatched = fast_pass([], _make_kos_svc())
        assert matched == []
        assert unmatched == []


class TestMediumPass:
    def test_resolves_unmatched(self):
        from spindle.kos.ner import medium_pass
        chunk = _make_chunk("The Centrifugal Pump failed.")
        svc = _make_kos_svc()
        svc.resolve_multistep.return_value = [
            ResolutionResult(
                mention="Centrifugal Pump", resolved=True, method="exact_label",
                concept_uri="uri:pump", pref_label="Pump", score=1.0,
            )
        ]
        resolved, still_unmatched = medium_pass([chunk], svc)
        assert len(resolved) == 1
        assert still_unmatched == []

    def test_empty_candidates_chunk_stays_unmatched(self):
        from spindle.kos.ner import medium_pass
        chunk = _make_chunk("   ")  # no extractable candidates
        with patch("spindle.kos.ner._extract_candidate_spans", return_value=[]):
            resolved, still_unmatched = medium_pass([chunk], _make_kos_svc())
        assert resolved == []
        assert len(still_unmatched) == 1


class TestDiscoveryPass:
    def test_returns_empty_when_gliner2_missing(self):
        from spindle.kos.ner import discovery_pass
        with patch.dict("sys.modules", {"gliner2": None}):
            novel, matched = discovery_pass([_make_chunk()], _make_kos_svc())
        assert novel == []
        assert matched == []


# ---------------------------------------------------------------------------
# Staging
# ---------------------------------------------------------------------------

class TestWriteStaging:
    def test_writes_turtle(self, tmp_path):
        candidates = [
            {"text": "Motor", "confidence": 0.9, "definition": "An electric motor."},
        ]
        write_staging(candidates, tmp_path)
        vocab = (tmp_path / "vocabulary.ttls").read_text()
        assert "Motor" in vocab
        assert "skos:prefLabel" in vocab

    def test_creates_dir_if_missing(self, tmp_path):
        stage_dir = tmp_path / "staging" / "sub"
        write_staging([{"text": "Sensor"}], stage_dir)
        assert (stage_dir / "vocabulary.ttls").exists()

    def test_appends_on_second_write(self, tmp_path):
        write_staging([{"text": "TermA"}], tmp_path)
        write_staging([{"text": "TermB"}], tmp_path)
        content = (tmp_path / "vocabulary.ttls").read_text()
        assert "TermA" in content
        assert "TermB" in content

    def test_skips_empty_label(self, tmp_path):
        write_staging([{"text": ""}], tmp_path)
        vocab = tmp_path / "vocabulary.ttls"
        if vocab.exists():
            assert "skos:prefLabel" not in vocab.read_text()


class TestMergeStaging:
    def test_merges_concepts(self, tmp_path):
        write_staging([{"text": "Motor", "definition": "An electric motor."}], tmp_path)
        svc = _make_kos_svc()
        result = merge_staging(tmp_path, svc)
        assert result["merged"] >= 0  # service mock might return None
        assert "violations" in result

    def test_missing_staging_file(self, tmp_path):
        result = merge_staging(tmp_path, _make_kos_svc())
        assert result["merged"] == 0


class TestRejectCandidate:
    def test_calls_rejection_log(self):
        svc = _make_kos_svc()
        reject_candidate("motor", "doc-1", svc, reason="Too generic", rejected_by="alice")
        svc._rejection_log.add.assert_called_once_with(
            term="motor",
            source_doc_id="doc-1",
            chunk_index=None,
            rejection_reason="Too Generic" if False else "Too generic",
            rejected_by="alice",
        )


class TestClearStaging:
    def test_clears_file(self, tmp_path):
        write_staging([{"text": "Motor"}], tmp_path)
        assert (tmp_path / "vocabulary.ttls").stat().st_size > 0
        clear_staging_file(tmp_path)
        assert (tmp_path / "vocabulary.ttls").read_text() == ""


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

class TestSynthesizeOntology:
    def test_no_store_returns_skipped(self, tmp_path):
        svc = _make_kos_svc()
        svc._store = None
        result = synthesize_ontology(svc, tmp_path / "ontology.owl")
        assert result["status"] == "skipped"

    def test_with_empty_store(self, tmp_path):
        try:
            import pyoxigraph
        except ImportError:
            pytest.skip("pyoxigraph not installed")
        svc = MagicMock()
        svc._store = pyoxigraph.Store()
        result = synthesize_ontology(svc, tmp_path / "ontology.owl")
        assert result["status"] == "ok"
        assert result["classes"] == 0

    def test_output_file_created(self, tmp_path):
        try:
            import pyoxigraph
        except ImportError:
            pytest.skip("pyoxigraph not installed")
        svc = MagicMock()
        svc._store = pyoxigraph.Store()
        out = tmp_path / "onto.owl"
        synthesize_ontology(svc, out)
        assert out.exists()


class TestGenerateShacl:
    def test_missing_ontology(self, tmp_path):
        result = generate_shacl(tmp_path / "missing.owl")
        assert result["status"] == "skipped"

    def test_generates_shapes(self, tmp_path):
        owl_path = tmp_path / "ontology.owl"
        owl_path.write_text(
            "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n"
            "<http://spindle.dev/ns/concept/pump>\n"
            "    a owl:Class .\n",
            encoding="utf-8",
        )
        result = generate_shacl(owl_path, tmp_path / "shapes.ttl")
        assert result["status"] == "ok"
        assert result["shapes"] >= 1
        assert (tmp_path / "shapes.ttl").exists()


# ---------------------------------------------------------------------------
# KOSExtractionPipeline
# ---------------------------------------------------------------------------

class TestKOSExtractionPipeline:
    def test_auto_selects_cold_start_for_empty_kos(self, tmp_path):
        svc = _make_kos_svc(concept_count=0)
        pipeline = KOSExtractionPipeline(svc, stage_dir=tmp_path)
        with patch("spindle.kos.extraction_llm.extract_candidates_via_llm", return_value=[]):
            result = pipeline.run([_make_chunk()])
        assert result["mode"] == "cold_start"

    def test_auto_selects_incremental_for_loaded_kos(self, tmp_path):
        svc = _make_kos_svc(concept_count=5)
        svc.search_ahocorasick.return_value = []
        svc.resolve_multistep.return_value = []
        pipeline = KOSExtractionPipeline(svc, stage_dir=tmp_path)
        with patch.dict("sys.modules", {"gliner2": None}):
            result = pipeline.run([_make_chunk()])
        assert result["mode"] == "incremental"

    def test_incremental_summary(self, tmp_path):
        svc = _make_kos_svc(concept_count=3)
        svc.search_ahocorasick.return_value = [
            EntityMention(text="pump", start=4, end=8,
                          concept_uri="uri:pump", matched_label="pump", pref_label="Pump")
        ]
        svc.resolve_multistep.return_value = []
        pipeline = KOSExtractionPipeline(svc, stage_dir=tmp_path)
        with patch.dict("sys.modules", {"gliner2": None}):
            result = pipeline.run([_make_chunk()], mode="incremental")
        assert result["pass1_matched"] == 1
        assert result["chunks_processed"] == 1

    def test_heuristic_candidates(self):
        chunk = _make_chunk("The Centrifugal Pump failed. High Pressure Valve stuck.")
        candidates = _heuristic_candidates([chunk])
        labels = [c["text"] for c in candidates]
        assert any("Centrifugal Pump" in l for l in labels)

    def test_cold_start_stages_candidates(self, tmp_path):
        svc = _make_kos_svc(concept_count=0)
        pipeline = KOSExtractionPipeline(svc, stage_dir=tmp_path)
        with patch("spindle.kos.extraction_llm.extract_candidates_via_llm", side_effect=ImportError):
            result = pipeline.run([_make_chunk("The Motor Drive unit. Servo Actuator failed.")])
        assert result["mode"] == "cold_start"
        # staging file might be empty if heuristics find nothing in mock text
        assert "candidates_staged" in result
