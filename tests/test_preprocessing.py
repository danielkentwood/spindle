"""Tests for spindle.preprocessing package."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from spindle.preprocessing.models import Chunk, DocumentRecord
from spindle.preprocessing.offsets import (
    build_offset_map,
    reconstruct_coref_resolved_text,
    to_document_offset,
)
from spindle.preprocessing.ingestion import (
    DocumentCatalog,
    DocumentIngestionStage,
    compute_content_hash,
)
from spindle.preprocessing.chunking import _chunk_fallback
from spindle.preprocessing.coref import resolve_coreferences_for_document
from spindle.preprocessing.preprocessor import SpindlePreprocessor, _get


# ── Model tests ───────────────────────────────────────────────────────────────


class TestChunk:
    def test_basic_construction(self):
        chunk = Chunk(text="Hello world", source_id="doc-1")
        assert chunk.text == "Hello world"
        assert chunk.source_id == "doc-1"
        assert chunk.metadata == {}

    def test_metadata_accessors(self):
        chunk = Chunk(
            text="Test",
            source_id="doc-1",
            metadata={
                "start_index": 10,
                "end_index": 14,
                "chunk_index": 0,
                "section_path": ["Introduction"],
                "coref_annotations": [{"mention": "he"}],
            },
        )
        assert chunk.start_index == 10
        assert chunk.end_index == 14
        assert chunk.chunk_index == 0
        assert chunk.section_path == ["Introduction"]
        assert len(chunk.coref_annotations) == 1

    def test_missing_metadata_returns_none(self):
        chunk = Chunk(text="Hello", source_id="doc-1")
        assert chunk.start_index is None
        assert chunk.coref_annotations == []


# ── Offset utilities ──────────────────────────────────────────────────────────


class TestOffsets:
    def test_to_document_offset_basic(self):
        chunk = Chunk(
            text="Alice works at Acme.",
            source_id="doc-1",
            metadata={"start_index": 100},
        )
        assert to_document_offset(chunk, 0) == 100
        assert to_document_offset(chunk, 5) == 105

    def test_to_document_offset_no_start_index(self):
        chunk = Chunk(text="Alice", source_id="doc-1")
        assert to_document_offset(chunk, 0) is None

    def test_build_offset_map(self):
        chunks = [
            Chunk(text="Hello", source_id="doc-1", metadata={"start_index": 0, "end_index": 5}),
            Chunk(text="World", source_id="doc-1", metadata={"start_index": 6, "end_index": 11}),
        ]
        offset_map = build_offset_map(chunks)
        assert offset_map[0] == (0, 0)
        assert offset_map[4] == (0, 4)
        assert offset_map[6] == (1, 0)

    def test_build_offset_map_missing_start_index(self):
        chunks = [Chunk(text="Hello", source_id="doc-1")]
        offset_map = build_offset_map(chunks)
        assert offset_map == {}

    def test_reconstruct_coref_resolved_no_annotations(self):
        chunk = Chunk(text="Alice works at Acme.", source_id="doc-1")
        assert reconstruct_coref_resolved_text(chunk) == "Alice works at Acme."

    def test_reconstruct_coref_resolved_with_annotations(self):
        chunk = Chunk(
            text="She works at Acme.",
            source_id="doc-1",
            metadata={
                "coref_annotations": [
                    {
                        "mention": "She",
                        "span_start": 0,
                        "span_end": 3,
                        "resolved_to": "Alice",
                        "chain_id": "chain_0",
                    }
                ]
            },
        )
        resolved = reconstruct_coref_resolved_text(chunk)
        assert resolved == "Alice works at Acme."

    def test_reconstruct_coref_multiple_mentions(self):
        chunk = Chunk(
            text="She and he went home.",
            source_id="doc-1",
            metadata={
                "coref_annotations": [
                    {
                        "mention": "he",
                        "span_start": 8,
                        "span_end": 10,
                        "resolved_to": "Bob",
                        "chain_id": "chain_1",
                    },
                    {
                        "mention": "She",
                        "span_start": 0,
                        "span_end": 3,
                        "resolved_to": "Alice",
                        "chain_id": "chain_0",
                    },
                ]
            },
        )
        resolved = reconstruct_coref_resolved_text(chunk)
        assert "Alice" in resolved
        assert "Bob" in resolved


# ── Document catalog ──────────────────────────────────────────────────────────


class TestDocumentCatalog:
    @pytest.fixture
    def catalog(self):
        with DocumentCatalog(":memory:") as c:
            yield c

    def _make_record(self, doc_id: str = "doc-1") -> DocumentRecord:
        from datetime import datetime

        return DocumentRecord(
            doc_id=doc_id,
            source_path=f"/path/{doc_id}.txt",
            content_hash="abc123",
            docling_json_path=f"/cache/{doc_id}.json",
            last_ingested=datetime(2024, 1, 1),
            version=1,
            metadata={"title": "Test"},
        )

    def test_upsert_and_get(self, catalog):
        record = self._make_record()
        catalog.upsert(record)
        retrieved = catalog.get("doc-1")
        assert retrieved is not None
        assert retrieved.doc_id == "doc-1"
        assert retrieved.content_hash == "abc123"
        assert retrieved.metadata == {"title": "Test"}

    def test_upsert_updates_existing(self, catalog):
        record = self._make_record()
        catalog.upsert(record)
        record.content_hash = "def456"
        record.version = 2
        catalog.upsert(record)
        retrieved = catalog.get("doc-1")
        assert retrieved.content_hash == "def456"
        assert retrieved.version == 2

    def test_get_missing_returns_none(self, catalog):
        assert catalog.get("nonexistent") is None

    def test_all_records(self, catalog):
        for i in range(5):
            catalog.upsert(self._make_record(f"doc-{i}"))
        assert len(catalog.all_records()) == 5


# ── Chunking ──────────────────────────────────────────────────────────────────


class TestChunkingFallback:
    def _make_record(self) -> DocumentRecord:
        from datetime import datetime

        return DocumentRecord(
            doc_id="doc-1",
            source_path="/fake.txt",
            content_hash="hash",
            docling_json_path="/fake.json",
            last_ingested=datetime(2024, 1, 1),
        )

    def test_basic_chunking(self):
        record = self._make_record()
        text = "A" * 1500
        chunks = _chunk_fallback(record, text, chunk_size=500, overlap=0.0)
        assert len(chunks) == 3
        for chunk in chunks:
            assert chunk.source_id == "doc-1"
            assert chunk.start_index is not None

    def test_overlap_produces_more_chunks(self):
        record = self._make_record()
        text = "A" * 1000
        no_overlap = _chunk_fallback(record, text, chunk_size=500, overlap=0.0)
        with_overlap = _chunk_fallback(record, text, chunk_size=500, overlap=0.50)
        assert len(with_overlap) > len(no_overlap)

    def test_start_end_indices_correct(self):
        record = self._make_record()
        text = "Hello World Test"
        chunks = _chunk_fallback(record, text, chunk_size=5, overlap=0.0)
        for chunk in chunks:
            start = chunk.start_index
            end = chunk.end_index
            assert text[start:end] == chunk.text


# ── Coref resolution ──────────────────────────────────────────────────────────


class TestCorefResolution:
    def _make_chunks(self) -> list[Chunk]:
        return [
            Chunk(
                text="Alice works at Acme.",
                source_id="doc-1",
                metadata={"start_index": 0, "end_index": 20},
            ),
            Chunk(
                text="She is an engineer.",
                source_id="doc-1",
                metadata={"start_index": 21, "end_index": 40},
            ),
        ]

    def test_fastcoref_not_installed_returns_unchanged(self):
        chunks = self._make_chunks()
        with patch("spindle.preprocessing.coref._run_fastcoref", side_effect=ImportError):
            result = resolve_coreferences_for_document("Alice works at Acme. She is an engineer.", chunks)
        assert all(chunk.coref_annotations == [] for chunk in result)

    def test_coref_annotations_projected_correctly(self):
        # Simulate fastcoref returning a single chain
        mock_chains = [
            [
                {"text": "Alice", "start": 0, "end": 5},  # representative (longest)
                {"text": "She", "start": 21, "end": 24},  # non-representative
            ]
        ]
        chunks = self._make_chunks()
        doc_text = "Alice works at Acme. She is an engineer."

        with patch("spindle.preprocessing.coref._run_fastcoref", return_value=mock_chains):
            result = resolve_coreferences_for_document(doc_text, chunks)

        # "She" is in the second chunk (start=21)
        second_chunk = result[1]
        assert len(second_chunk.coref_annotations) == 1
        ann = second_chunk.coref_annotations[0]
        assert ann["mention"] == "She"
        assert ann["resolved_to"] == "Alice"

    def test_blacklist_filters_mentions(self):
        mock_chains = [
            [
                {"text": "Alice", "start": 0, "end": 5},
                {"text": "She", "start": 21, "end": 24},
            ]
        ]
        chunks = self._make_chunks()
        doc_text = "Alice works at Acme. She is an engineer."

        with patch("spindle.preprocessing.coref._run_fastcoref", return_value=mock_chains):
            result = resolve_coreferences_for_document(
                doc_text, chunks, blacklist={"alice"}
            )

        # Alice is blacklisted as representative — whole chain should be skipped
        assert all(chunk.coref_annotations == [] for chunk in result)


# ── SpindlePreprocessor integration ──────────────────────────────────────────


class TestSpindlePreprocessor:
    @pytest.fixture
    def tmp_doc(self, tmp_path):
        """Create a minimal fake Docling JSON file and return its path."""
        doc_dir = tmp_path / "docs"
        doc_dir.mkdir()
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Create a text file (will be read by fallback ingestion)
        doc_file = doc_dir / "test.txt"
        doc_file.write_text("Alice works at Acme. She is an engineer.")

        # Pre-create the docling JSON that the ingestion stage would produce
        json_file = cache_dir / "test.json"
        json_file.write_text(
            json.dumps({"text": "Alice works at Acme. She is an engineer."})
        )

        return doc_file, cache_dir

    def test_preprocessor_returns_chunks(self, tmp_path, tmp_doc):
        doc_file, cache_dir = tmp_doc

        # Patch convert_document to avoid needing Docling installed
        with patch(
            "spindle.preprocessing.ingestion.convert_document",
            return_value={
                "json_path": cache_dir / "test.json",
                "docling_result": None,
            },
        ):
            preprocessor = SpindlePreprocessor(
                documents=[doc_file],
                catalog_path=":memory:",
                docling_output_dir=cache_dir,
            )
            chunks = preprocessor({"chunk_size": 100, "overlap": 0.0, "coref_model": None})

        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_incremental_skips_unchanged_docs(self, tmp_path, tmp_doc):
        doc_file, cache_dir = tmp_doc

        with patch(
            "spindle.preprocessing.ingestion.convert_document",
            return_value={
                "json_path": cache_dir / "test.json",
                "docling_result": None,
            },
        ) as mock_convert:
            preprocessor = SpindlePreprocessor(
                documents=[doc_file],
                catalog_path=str(tmp_path / "catalog.db"),
                docling_output_dir=cache_dir,
            )
            # First run
            preprocessor({"chunk_size": 100, "overlap": 0.0, "coref_model": None})
            first_call_count = mock_convert.call_count

            # Second run — same file, same hash — should NOT call convert again
            preprocessor({"chunk_size": 100, "overlap": 0.0, "coref_model": None})
            second_call_count = mock_convert.call_count

        assert second_call_count == first_call_count  # no additional calls


# ── _get helper ───────────────────────────────────────────────────────────────


class TestGetHelper:
    def test_get_from_dict(self):
        assert _get({"chunk_size": 800}, "chunk_size", 600) == 800

    def test_get_default_on_missing(self):
        assert _get({"other": 1}, "chunk_size", 600) == 600

    def test_get_from_none(self):
        assert _get(None, "chunk_size", 600) == 600

    def test_get_from_object(self):
        cfg = MagicMock()
        cfg.chunk_size = 1200
        assert _get(cfg, "chunk_size", 600) == 1200
