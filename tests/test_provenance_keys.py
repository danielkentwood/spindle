"""Tests for deterministic provenance key generation and triple -> ProvenanceObject conversion."""

import pytest

from spindle.baml_client.types import (
    CharacterSpan,
    Entity,
    AttributeValue,
    SourceMetadata,
    Triple,
)
from spindle.provenance.keys import triple_provenance_id
from spindle.provenance.conversion import triple_to_provenance_object


def _make_triple(
    subject_name: str = "Alice",
    predicate: str = "works_at",
    object_name: str = "TechCorp",
    source_name: str = "doc-1",
    source_url: str | None = None,
    spans: list[CharacterSpan] | None = None,
) -> Triple:
    if spans is None:
        spans = [CharacterSpan(text="Alice works at TechCorp", start=0, end=23)]
    return Triple(
        subject=Entity(name=subject_name, type="Person", description="", custom_atts={}),
        predicate=predicate,
        object=Entity(name=object_name, type="Organization", description="", custom_atts={}),
        source=SourceMetadata(source_name=source_name, source_url=source_url),
        supporting_spans=spans,
        extraction_datetime="2024-01-15T10:30:00Z",
    )


class TestTripleProvenanceId:
    """Tests for triple_provenance_id deterministic key generation."""

    def test_returns_non_empty_string(self):
        triple = _make_triple()
        result = triple_provenance_id(triple)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_same_triple_same_key(self):
        t1 = _make_triple()
        t2 = _make_triple()
        assert triple_provenance_id(t1) == triple_provenance_id(t2)

    def test_different_subject_different_key(self):
        t1 = _make_triple(subject_name="Alice")
        t2 = _make_triple(subject_name="Bob")
        assert triple_provenance_id(t1) != triple_provenance_id(t2)

    def test_different_predicate_different_key(self):
        t1 = _make_triple(predicate="works_at")
        t2 = _make_triple(predicate="owns")
        assert triple_provenance_id(t1) != triple_provenance_id(t2)

    def test_different_object_different_key(self):
        t1 = _make_triple(object_name="TechCorp")
        t2 = _make_triple(object_name="StartupCo")
        assert triple_provenance_id(t1) != triple_provenance_id(t2)

    def test_different_source_different_key(self):
        t1 = _make_triple(source_name="doc-1")
        t2 = _make_triple(source_name="doc-2")
        assert triple_provenance_id(t1) != triple_provenance_id(t2)

    def test_case_normalization_subject(self):
        """Key should be stable across subject name case differences."""
        t1 = _make_triple(subject_name="Alice Johnson")
        t2 = _make_triple(subject_name="ALICE JOHNSON")
        assert triple_provenance_id(t1) == triple_provenance_id(t2)

    def test_whitespace_normalization(self):
        """Key should be stable across whitespace differences in names."""
        t1 = _make_triple(subject_name="Alice  Johnson")
        t2 = _make_triple(subject_name="Alice Johnson")
        assert triple_provenance_id(t1) == triple_provenance_id(t2)

    def test_none_source_url_stable(self):
        """Key is stable when source_url is None."""
        t1 = _make_triple(source_url=None)
        t2 = _make_triple(source_url=None)
        assert triple_provenance_id(t1) == triple_provenance_id(t2)

    def test_source_url_included_in_key(self):
        """source_url distinguishes keys when present vs absent."""
        t1 = _make_triple(source_url=None)
        t2 = _make_triple(source_url="https://example.com/article1")
        assert triple_provenance_id(t1) != triple_provenance_id(t2)

    def test_key_is_fixed_length(self):
        """SHA256 hex digest should always be 64 chars."""
        t = _make_triple()
        assert len(triple_provenance_id(t)) == 64

    def test_key_independent_of_spans(self):
        """Span content/offsets do not affect the provenance ID (SPO+source only)."""
        t1 = _make_triple(spans=[CharacterSpan(text="Alice works at TechCorp", start=0, end=23)])
        t2 = _make_triple(spans=[CharacterSpan(text="entirely different span text", start=5, end=99)])
        assert triple_provenance_id(t1) == triple_provenance_id(t2)

    def test_key_independent_of_extraction_datetime(self):
        """Extraction datetime does not affect the provenance ID."""
        t1 = _make_triple()
        t1.extraction_datetime = "2024-01-01T00:00:00Z"
        t2 = _make_triple()
        t2.extraction_datetime = "2025-12-31T23:59:59Z"
        assert triple_provenance_id(t1) == triple_provenance_id(t2)


class TestTripleToProvenanceObject:
    """Tests for triple_to_provenance_object conversion."""

    def test_returns_provenance_object(self):
        from spindle.provenance.models import ProvenanceObject
        triple = _make_triple()
        result = triple_to_provenance_object(triple)
        assert isinstance(result, ProvenanceObject)

    def test_object_id_matches_provenance_id(self):
        triple = _make_triple()
        result = triple_to_provenance_object(triple)
        assert result.object_id == triple_provenance_id(triple)

    def test_object_type_is_kg_edge(self):
        triple = _make_triple()
        result = triple_to_provenance_object(triple)
        assert result.object_type == "kg_edge"

    def test_one_doc_per_source(self):
        triple = _make_triple(source_name="article-1")
        result = triple_to_provenance_object(triple)
        assert len(result.docs) == 1
        assert result.docs[0].doc_id == "article-1"

    def test_spans_preserved(self):
        spans = [
            CharacterSpan(text="Alice works at TechCorp", start=0, end=23),
            CharacterSpan(text="TechCorp is a company", start=30, end=51),
        ]
        triple = _make_triple(spans=spans)
        result = triple_to_provenance_object(triple)
        doc = result.docs[0]
        assert len(doc.spans) == 2
        assert doc.spans[0].text == "Alice works at TechCorp"
        assert doc.spans[0].start_offset == 0
        assert doc.spans[0].end_offset == 23
        assert doc.spans[1].text == "TechCorp is a company"

    def test_negative_sentinel_offsets_become_none(self):
        """CharacterSpan start/end=-1 (not found) should be stored as None in EvidenceSpan."""
        spans = [CharacterSpan(text="not found span", start=-1, end=-1)]
        triple = _make_triple(spans=spans)
        result = triple_to_provenance_object(triple)
        evidence_span = result.docs[0].spans[0]
        assert evidence_span.start_offset is None
        assert evidence_span.end_offset is None

    def test_none_offsets_remain_none(self):
        """CharacterSpan start/end=None should be stored as None in EvidenceSpan."""
        spans = [CharacterSpan(text="some span", start=None, end=None)]
        triple = _make_triple(spans=spans)
        result = triple_to_provenance_object(triple)
        evidence_span = result.docs[0].spans[0]
        assert evidence_span.start_offset is None
        assert evidence_span.end_offset is None

    def test_no_spans_yields_empty_evidence_list(self):
        triple = _make_triple(spans=[])
        result = triple_to_provenance_object(triple)
        assert result.docs[0].spans == []

    def test_span_text_preserved(self):
        spans = [CharacterSpan(text="Evidence text here.", start=5, end=24)]
        triple = _make_triple(spans=spans)
        result = triple_to_provenance_object(triple)
        assert result.docs[0].spans[0].text == "Evidence text here."
