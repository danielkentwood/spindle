"""Tests for spindle.provenance.ProvenanceStore."""

import pytest

from spindle.provenance.models import EvidenceSpan, ProvenanceDoc, ProvenanceObject
from spindle.provenance.store import ProvenanceStore


@pytest.fixture
def store():
    """In-memory ProvenanceStore."""
    with ProvenanceStore(":memory:") as s:
        yield s


def _make_prov(
    object_id: str = "edge-001",
    object_type: str = "kg_edge",
    docs: list | None = None,
) -> tuple:
    if docs is None:
        docs = [
            ProvenanceDoc(
                doc_id="doc-A",
                spans=[
                    EvidenceSpan(text="Alice works at Acme.", start_offset=0, end_offset=20),
                    EvidenceSpan(
                        text="She is an engineer.",
                        start_offset=21,
                        end_offset=40,
                        section_path=["Introduction"],
                    ),
                ],
            ),
            ProvenanceDoc(
                doc_id="doc-B",
                spans=[
                    EvidenceSpan(text="Acme employs Alice.", start_offset=5, end_offset=24),
                ],
            ),
        ]
    return object_id, object_type, docs


# ── Pattern 1: point lookup ───────────────────────────────────────────────────


class TestPointLookup:
    def test_basic_create_and_get(self, store):
        obj_id, obj_type, docs = _make_prov()
        store.create_provenance(obj_id, obj_type, docs)

        result = store.get_provenance(obj_id)
        assert result is not None
        assert result.object_id == obj_id
        assert result.object_type == obj_type
        assert len(result.docs) == 2

    def test_spans_are_preserved(self, store):
        obj_id, obj_type, docs = _make_prov()
        store.create_provenance(obj_id, obj_type, docs)

        result = store.get_provenance(obj_id)
        doc_a = next(d for d in result.docs if d.doc_id == "doc-A")
        assert len(doc_a.spans) == 2
        assert doc_a.spans[0].text == "Alice works at Acme."
        assert doc_a.spans[0].start_offset == 0
        assert doc_a.spans[0].end_offset == 20

    def test_section_path_roundtrip(self, store):
        obj_id, obj_type, docs = _make_prov()
        store.create_provenance(obj_id, obj_type, docs)

        result = store.get_provenance(obj_id)
        doc_a = next(d for d in result.docs if d.doc_id == "doc-A")
        span_with_path = next(s for s in doc_a.spans if s.section_path)
        assert span_with_path.section_path == ["Introduction"]

    def test_missing_object_returns_none(self, store):
        assert store.get_provenance("nonexistent-id") is None

    def test_replace_on_duplicate_object_id(self, store):
        obj_id, obj_type, docs = _make_prov()
        store.create_provenance(obj_id, obj_type, docs)
        # Re-insert with different object_type — should replace
        store.create_provenance(obj_id, "owl_entity", [])
        result = store.get_provenance(obj_id)
        assert result.object_type == "owl_entity"
        assert result.docs == []


# ── Pattern 2: reverse lookup ─────────────────────────────────────────────────


class TestReverseLookup:
    def test_get_affected_objects_basic(self, store):
        store.create_provenance("edge-001", "kg_edge", [ProvenanceDoc(doc_id="doc-X")])
        store.create_provenance("edge-002", "kg_edge", [ProvenanceDoc(doc_id="doc-X")])
        store.create_provenance("concept-003", "vocab_entry", [ProvenanceDoc(doc_id="doc-Y")])

        affected = store.get_affected_objects("doc-X")
        obj_ids = {r["object_id"] for r in affected}
        assert obj_ids == {"edge-001", "edge-002"}

    def test_get_affected_objects_no_match(self, store):
        store.create_provenance("edge-001", "kg_edge", [ProvenanceDoc(doc_id="doc-A")])
        assert store.get_affected_objects("doc-MISSING") == []

    def test_multiple_docs_per_object(self, store):
        obj_id, obj_type, docs = _make_prov()
        store.create_provenance(obj_id, obj_type, docs)

        # Both doc-A and doc-B should return edge-001
        for doc_id in ("doc-A", "doc-B"):
            affected = store.get_affected_objects(doc_id)
            assert any(r["object_id"] == obj_id for r in affected)


# ── Cascade deletes ───────────────────────────────────────────────────────────


class TestCascadeDelete:
    def test_delete_removes_docs_and_spans(self, store):
        obj_id, obj_type, docs = _make_prov()
        store.create_provenance(obj_id, obj_type, docs)

        store.delete_provenance(obj_id)

        assert store.get_provenance(obj_id) is None
        # Confirm no orphan rows survive
        rows = store._conn.execute("SELECT COUNT(*) FROM provenance_docs").fetchone()[0]
        assert rows == 0
        spans = store._conn.execute("SELECT COUNT(*) FROM evidence_spans").fetchone()[0]
        assert spans == 0

    def test_delete_nonexistent_is_noop(self, store):
        store.delete_provenance("does-not-exist")  # should not raise


# ── Span updates ─────────────────────────────────────────────────────────────


class TestUpdateProvenance:
    def test_update_spans_for_doc(self, store):
        obj_id, obj_type, docs = _make_prov()
        store.create_provenance(obj_id, obj_type, docs)

        new_spans = [EvidenceSpan(text="Updated span.", start_offset=100, end_offset=113)]
        store.update_provenance_for_doc(obj_id, "doc-A", new_spans)

        result = store.get_provenance(obj_id)
        doc_a = next(d for d in result.docs if d.doc_id == "doc-A")
        assert len(doc_a.spans) == 1
        assert doc_a.spans[0].text == "Updated span."

    def test_update_nonexistent_doc_is_noop(self, store):
        obj_id, obj_type, docs = _make_prov()
        store.create_provenance(obj_id, obj_type, docs)
        # Should not raise
        store.update_provenance_for_doc(obj_id, "doc-MISSING", [])


# ── Batch operations ─────────────────────────────────────────────────────────


class TestBatchOperations:
    def test_multiple_objects(self, store):
        for i in range(10):
            store.create_provenance(
                f"edge-{i:03d}",
                "kg_edge",
                [ProvenanceDoc(doc_id=f"doc-{i % 3}")],
            )

        for i in range(10):
            result = store.get_provenance(f"edge-{i:03d}")
            assert result is not None
            assert result.object_type == "kg_edge"

    def test_batch_object_types(self, store):
        for obj_type in ("kg_edge", "owl_entity", "vocab_entry"):
            store.create_provenance(f"{obj_type}-1", obj_type, [])

        for obj_type in ("kg_edge", "owl_entity", "vocab_entry"):
            result = store.get_provenance(f"{obj_type}-1")
            assert result.object_type == obj_type
