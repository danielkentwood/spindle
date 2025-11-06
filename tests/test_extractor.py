"""Unit tests for SpindleExtractor class."""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from freezegun import freeze_time
from spindle.baml_client.types import (
    ExtractionResult,
    Triple,
    Entity,
    AttributeValue,
    CharacterSpan,
    SourceMetadata
)

from spindle import SpindleExtractor
from tests.fixtures.sample_texts import SIMPLE_TEXT, TEXT_WITH_NEWLINES


def _create_test_entity(name: str, entity_type: str, description: str = "", **attrs):
    """Helper to create a test Entity with custom attributes."""
    custom_atts = {
        attr_name: AttributeValue(value=attr_val, type="string")
        for attr_name, attr_val in attrs.items()
    }
    return Entity(
        name=name,
        type=entity_type,
        description=description,
        custom_atts=custom_atts
    )


class TestSpindleExtractorInit:
    """Tests for SpindleExtractor initialization."""
    
    def test_init_with_ontology(self, simple_ontology):
        """Test initializing with an ontology."""
        extractor = SpindleExtractor(simple_ontology)
        
        assert extractor.ontology == simple_ontology
        assert extractor._ontology_recommender is None
    
    def test_init_without_ontology(self):
        """Test initializing without an ontology."""
        extractor = SpindleExtractor()
        
        assert extractor.ontology is None
        assert extractor._ontology_recommender is not None
        assert extractor.ontology_scope == "balanced"
    
    def test_init_with_custom_scope(self):
        """Test initializing with custom ontology scope."""
        extractor = SpindleExtractor(ontology_scope="minimal")
        
        assert extractor.ontology_scope == "minimal"


class TestSpindleExtractorExtract:
    """Tests for SpindleExtractor.extract method."""
    
    @patch('spindle.extractor.b.ExtractTriples')
    @freeze_time("2024-01-15 10:30:00")
    def test_extract_basic(self, mock_baml_extract, simple_ontology):
        """Test basic extraction with mocked BAML call."""
        # Setup mock with Entity objects
        mock_result = ExtractionResult(
            triples=[
                Triple(
                    subject=_create_test_entity("Alice Johnson", "Person", "A software engineer"),
                    predicate="works_at",
                    object=_create_test_entity("TechCorp", "Organization", "A technology company"),
                    source=SourceMetadata(source_name="Test", source_url=None),
                    supporting_spans=[
                        CharacterSpan(text="Alice Johnson works at TechCorp", start=None, end=None)
                    ],
                    extraction_datetime=None
                )
            ],
            reasoning="Extracted based on ontology."
        )
        mock_baml_extract.return_value = mock_result
        
        # Execute
        extractor = SpindleExtractor(simple_ontology)
        result = extractor.extract(
            text=SIMPLE_TEXT,
            source_name="Test Document",
            source_url="https://example.com/test"
        )
        
        # Verify
        assert len(result.triples) == 1
        assert result.triples[0].subject.name == "Alice Johnson"
        assert result.triples[0].subject.type == "Person"
        assert result.triples[0].object.name == "TechCorp"
        assert result.triples[0].object.type == "Organization"
        
        # Check that extraction_datetime was set
        assert result.triples[0].extraction_datetime == "2024-01-15T10:30:00Z"
        
        # Check that BAML was called correctly
        mock_baml_extract.assert_called_once()
        call_args = mock_baml_extract.call_args
        assert call_args[1]["text"] == SIMPLE_TEXT
        assert call_args[1]["ontology"] == simple_ontology
        assert call_args[1]["source_metadata"].source_name == "Test Document"
    
    @patch('spindle.extractor.b.ExtractTriples')
    def test_extract_with_existing_triples(self, mock_baml_extract, simple_ontology, sample_triples):
        """Test extraction with existing triples for entity consistency."""
        mock_result = ExtractionResult(
            triples=[],
            reasoning="No new triples."
        )
        mock_baml_extract.return_value = mock_result
        
        extractor = SpindleExtractor(simple_ontology)
        result = extractor.extract(
            text="Additional text",
            source_name="Test",
            existing_triples=sample_triples
        )
        
        # Check that existing triples were passed to BAML
        call_args = mock_baml_extract.call_args
        assert call_args[1]["existing_triples"] == sample_triples
    
    @patch('spindle.extractor.b.ExtractTriples')
    def test_extract_computes_span_indices(self, mock_baml_extract, simple_ontology):
        """Test that extraction computes character span indices."""
        # Mock BAML to return triple with None indices
        mock_result = ExtractionResult(
            triples=[
                Triple(
                    subject=_create_test_entity("Alice Johnson", "Person"),
                    predicate="works_at",
                    object=_create_test_entity("TechCorp", "Organization"),
                    source=SourceMetadata(source_name="Test"),
                    supporting_spans=[
                        CharacterSpan(text="Alice Johnson works at TechCorp", start=None, end=None)
                    ],
                    extraction_datetime=None
                )
            ],
            reasoning="Test"
        )
        mock_baml_extract.return_value = mock_result
        
        extractor = SpindleExtractor(simple_ontology)
        result = extractor.extract(
            text=SIMPLE_TEXT,
            source_name="Test"
        )
        
        # Check that indices were computed
        span = result.triples[0].supporting_spans[0]
        assert span.start is not None
        assert span.end is not None
        assert span.start >= 0
        assert span.end > span.start
    
    @patch('spindle.extractor.b.ExtractTriples')
    def test_extract_handles_unfound_spans(self, mock_baml_extract, simple_ontology):
        """Test that extraction handles spans that can't be found."""
        # Mock BAML to return triple with span not in text
        mock_result = ExtractionResult(
            triples=[
                Triple(
                    subject=_create_test_entity("Bob", "Person"),
                    predicate="works_at",
                    object=_create_test_entity("Google", "Organization"),
                    source=SourceMetadata(source_name="Test"),
                    supporting_spans=[
                        CharacterSpan(text="Bob works at Google", start=None, end=None)
                    ],
                    extraction_datetime=None
                )
            ],
            reasoning="Test"
        )
        mock_baml_extract.return_value = mock_result
        
        extractor = SpindleExtractor(simple_ontology)
        result = extractor.extract(
            text=SIMPLE_TEXT,  # Doesn't contain "Bob works at Google"
            source_name="Test"
        )
        
        # Check that indices were set to -1 (not found)
        span = result.triples[0].supporting_spans[0]
        assert span.start == -1
        assert span.end == -1
    
    @patch('spindle.OntologyRecommender.recommend')
    @patch('spindle.extractor.b.ExtractTriples')
    def test_extract_auto_recommends_ontology(self, mock_baml_extract, mock_recommend, mock_ontology_recommendation):
        """Test that extraction auto-recommends ontology when not provided."""
        # Setup mocks
        mock_recommend.return_value = mock_ontology_recommendation
        mock_result = ExtractionResult(triples=[], reasoning="Test")
        mock_baml_extract.return_value = mock_result
        
        # Execute
        extractor = SpindleExtractor()  # No ontology provided
        result = extractor.extract(
            text=SIMPLE_TEXT,
            source_name="Test"
        )
        
        # Verify that recommend was called
        mock_recommend.assert_called_once_with(text=SIMPLE_TEXT, scope="balanced")
        
        # Verify that ontology was set
        assert extractor.ontology == mock_ontology_recommendation.ontology
    
    @patch('spindle.OntologyRecommender.recommend')
    @patch('spindle.extractor.b.ExtractTriples')
    def test_extract_uses_custom_scope_for_auto_ontology(self, mock_baml_extract, mock_recommend, mock_ontology_recommendation):
        """Test that custom scope is used for auto-recommendation."""
        mock_recommend.return_value = mock_ontology_recommendation
        mock_result = ExtractionResult(triples=[], reasoning="Test")
        mock_baml_extract.return_value = mock_result
        
        extractor = SpindleExtractor(ontology_scope="comprehensive")
        result = extractor.extract(
            text=SIMPLE_TEXT,
            source_name="Test"
        )
        
        # Verify that recommend was called with custom scope
        mock_recommend.assert_called_once_with(text=SIMPLE_TEXT, scope="comprehensive")
    
    @patch('spindle.OntologyRecommender.recommend')
    @patch('spindle.extractor.b.ExtractTriples')
    def test_extract_ontology_scope_override(self, mock_baml_extract, mock_recommend, mock_ontology_recommendation):
        """Test that ontology_scope parameter overrides default."""
        mock_recommend.return_value = mock_ontology_recommendation
        mock_result = ExtractionResult(triples=[], reasoning="Test")
        mock_baml_extract.return_value = mock_result
        
        extractor = SpindleExtractor(ontology_scope="balanced")
        result = extractor.extract(
            text=SIMPLE_TEXT,
            source_name="Test",
            ontology_scope="minimal"  # Override
        )
        
        # Verify that recommend was called with overridden scope
        mock_recommend.assert_called_once_with(text=SIMPLE_TEXT, scope="minimal")
    
    @patch('spindle.extractor.b.ExtractTriples')
    def test_extract_preserves_existing_indices(self, mock_baml_extract, simple_ontology):
        """Test that extraction preserves already-set span indices."""
        # Mock BAML to return triple with indices already set
        mock_result = ExtractionResult(
            triples=[
                Triple(
                    subject=_create_test_entity("Alice", "Person"),
                    predicate="works_at",
                    object=_create_test_entity("TechCorp", "Organization"),
                    source=SourceMetadata(source_name="Test"),
                    supporting_spans=[
                        CharacterSpan(text="Alice works at TechCorp", start=0, end=23)
                    ],
                    extraction_datetime=None
                )
            ],
            reasoning="Test"
        )
        mock_baml_extract.return_value = mock_result
        
        extractor = SpindleExtractor(simple_ontology)
        result = extractor.extract(
            text=SIMPLE_TEXT,
            source_name="Test"
        )
        
        # Check that existing indices were preserved
        span = result.triples[0].supporting_spans[0]
        assert span.start == 0
        assert span.end == 23
    
    @patch('spindle.extractor.b.ExtractTriples')
    @freeze_time("2024-01-15 10:30:00")
    def test_extract_sets_datetime_for_all_triples(self, mock_baml_extract, simple_ontology):
        """Test that extraction sets datetime for all returned triples."""
        # Mock BAML to return multiple triples
        mock_result = ExtractionResult(
            triples=[
                Triple(
                    subject=_create_test_entity("Alice", "Person"),
                    predicate="works_at",
                    object=_create_test_entity("TechCorp", "Organization"),
                    source=SourceMetadata(source_name="Test"),
                    supporting_spans=[],
                    extraction_datetime=None
                ),
                Triple(
                    subject=_create_test_entity("Bob", "Person"),
                    predicate="works_at",
                    object=_create_test_entity("Google", "Organization"),
                    source=SourceMetadata(source_name="Test"),
                    supporting_spans=[],
                    extraction_datetime=None
                )
            ],
            reasoning="Test"
        )
        mock_baml_extract.return_value = mock_result
        
        extractor = SpindleExtractor(simple_ontology)
        result = extractor.extract(
            text="Test text",
            source_name="Test"
        )
        
        # Check that all triples have datetime set
        assert all(t.extraction_datetime == "2024-01-15T10:30:00Z" for t in result.triples)
    
    @patch('spindle.extractor.b.ExtractTriples')
    def test_extract_with_no_source_url(self, mock_baml_extract, simple_ontology):
        """Test extraction without providing source URL."""
        mock_result = ExtractionResult(triples=[], reasoning="Test")
        mock_baml_extract.return_value = mock_result
        
        extractor = SpindleExtractor(simple_ontology)
        result = extractor.extract(
            text=SIMPLE_TEXT,
            source_name="Test"
            # No source_url provided
        )
        
        # Check that source_url is None in metadata passed to BAML
        call_args = mock_baml_extract.call_args
        assert call_args[1]["source_metadata"].source_url is None


class TestSpindleExtractorExtractAsync:
    """Tests for SpindleExtractor.extract_async method."""
    
    @patch('spindle.extractor.async_b.ExtractTriples')
    @freeze_time("2024-01-15 10:30:00")
    @pytest.mark.asyncio
    async def test_extract_async_basic(self, mock_baml_extract, simple_ontology):
        """Test basic async extraction with mocked BAML call."""
        # Setup mock with Entity objects
        mock_result = ExtractionResult(
            triples=[
                Triple(
                    subject=_create_test_entity("Alice Johnson", "Person", "A software engineer"),
                    predicate="works_at",
                    object=_create_test_entity("TechCorp", "Organization", "A technology company"),
                    source=SourceMetadata(source_name="Test", source_url=None),
                    supporting_spans=[
                        CharacterSpan(text="Alice Johnson works at TechCorp", start=None, end=None)
                    ],
                    extraction_datetime=None
                )
            ],
            reasoning="Extracted based on ontology."
        )
        mock_baml_extract.return_value = mock_result
        
        # Execute
        extractor = SpindleExtractor(simple_ontology)
        result = await extractor.extract_async(
            text=SIMPLE_TEXT,
            source_name="Test Document",
            source_url="https://example.com/test"
        )
        
        # Verify
        assert len(result.triples) == 1
        assert result.triples[0].subject.name == "Alice Johnson"
        assert result.triples[0].extraction_datetime == "2024-01-15T10:30:00Z"
        
        # Check that async BAML was called correctly
        mock_baml_extract.assert_called_once()
        call_args = mock_baml_extract.call_args
        assert call_args[1]["text"] == SIMPLE_TEXT
        assert call_args[1]["ontology"] == simple_ontology
    
    @patch('spindle.extractor.async_b.ExtractTriples')
    @pytest.mark.asyncio
    async def test_extract_async_computes_span_indices(self, mock_baml_extract, simple_ontology):
        """Test that async extraction computes character span indices."""
        mock_result = ExtractionResult(
            triples=[
                Triple(
                    subject=_create_test_entity("Alice Johnson", "Person"),
                    predicate="works_at",
                    object=_create_test_entity("TechCorp", "Organization"),
                    source=SourceMetadata(source_name="Test"),
                    supporting_spans=[
                        CharacterSpan(text="Alice Johnson works at TechCorp", start=None, end=None)
                    ],
                    extraction_datetime=None
                )
            ],
            reasoning="Test"
        )
        mock_baml_extract.return_value = mock_result
        
        extractor = SpindleExtractor(simple_ontology)
        result = await extractor.extract_async(
            text=SIMPLE_TEXT,
            source_name="Test"
        )
        
        # Check that indices were computed
        span = result.triples[0].supporting_spans[0]
        assert span.start is not None
        assert span.end is not None
        assert span.start >= 0
        assert span.end > span.start


class TestSpindleExtractorExtractBatch:
    """Tests for SpindleExtractor.extract_batch method."""
    
    @patch('spindle.extractor.async_b.ExtractTriples')
    @freeze_time("2024-01-15 10:30:00")
    @pytest.mark.asyncio
    async def test_extract_batch_basic(self, mock_baml_extract, simple_ontology):
        """Test basic batch extraction."""
        # Setup mocks for two texts
        mock_result_1 = ExtractionResult(
            triples=[
                Triple(
                    subject=_create_test_entity("Alice", "Person"),
                    predicate="works_at",
                    object=_create_test_entity("TechCorp", "Organization"),
                    source=SourceMetadata(source_name="doc1"),
                    supporting_spans=[],
                    extraction_datetime=None
                )
            ],
            reasoning="First extraction"
        )
        mock_result_2 = ExtractionResult(
            triples=[
                Triple(
                    subject=_create_test_entity("Bob", "Person"),
                    predicate="works_at",
                    object=_create_test_entity("Google", "Organization"),
                    source=SourceMetadata(source_name="doc2"),
                    supporting_spans=[],
                    extraction_datetime=None
                )
            ],
            reasoning="Second extraction"
        )
        mock_baml_extract.side_effect = [mock_result_1, mock_result_2]
        
        extractor = SpindleExtractor(simple_ontology)
        texts = [
            ("Alice works at TechCorp", "doc1", None),
            ("Bob works at Google", "doc2", None)
        ]
        results = await extractor.extract_batch(texts)
        
        # Verify
        assert len(results) == 2
        assert len(results[0].triples) == 1
        assert len(results[1].triples) == 1
        assert results[0].triples[0].subject.name == "Alice"
        assert results[1].triples[0].subject.name == "Bob"
        
        # Verify BAML was called twice
        assert mock_baml_extract.call_count == 2
    
    @patch('spindle.extractor.async_b.ExtractTriples')
    @pytest.mark.asyncio
    async def test_extract_batch_sequential_consistency(self, mock_baml_extract, simple_ontology):
        """Test that batch extraction maintains sequential consistency."""
        # First extraction returns a triple
        mock_result_1 = ExtractionResult(
            triples=[
                Triple(
                    subject=_create_test_entity("Alice", "Person"),
                    predicate="works_at",
                    object=_create_test_entity("TechCorp", "Organization"),
                    source=SourceMetadata(source_name="doc1"),
                    supporting_spans=[],
                    extraction_datetime=None
                )
            ],
            reasoning="First"
        )
        # Second extraction should receive first triple as existing_triples
        mock_result_2 = ExtractionResult(
            triples=[
                Triple(
                    subject=_create_test_entity("Bob", "Person"),
                    predicate="manages",
                    object=_create_test_entity("Alice", "Person"),  # Same entity name
                    source=SourceMetadata(source_name="doc2"),
                    supporting_spans=[],
                    extraction_datetime=None
                )
            ],
            reasoning="Second"
        )
        mock_baml_extract.side_effect = [mock_result_1, mock_result_2]
        
        extractor = SpindleExtractor(simple_ontology)
        texts = [
            ("Alice works at TechCorp", "doc1", None),
            ("Bob manages Alice", "doc2", None)
        ]
        results = await extractor.extract_batch(texts)
        
        # Verify second call received first triple as existing_triples
        assert mock_baml_extract.call_count == 2
        
        # Get the second call's arguments (index 1)
        second_call = mock_baml_extract.call_args_list[1]
        # call_args is a tuple of (args, kwargs), so [1] is kwargs
        second_call_kwargs = second_call[1]
        existing_triples = second_call_kwargs["existing_triples"]
        
        # The second call should have received only the first triple
        # (before the second extraction's triples were added)
        assert len(existing_triples) == 1
        assert existing_triples[0].subject.name == "Alice"
        assert existing_triples[0].source.source_name == "doc1"
    
    @patch('spindle.extractor.async_b.ExtractTriples')
    @pytest.mark.asyncio
    async def test_extract_batch_with_initial_triples(self, mock_baml_extract, simple_ontology, sample_triples):
        """Test batch extraction with initial existing triples."""
        mock_result = ExtractionResult(
            triples=[],
            reasoning="No new triples"
        )
        mock_baml_extract.return_value = mock_result
        
        extractor = SpindleExtractor(simple_ontology)
        texts = [("Additional text", "doc1", None)]
        results = await extractor.extract_batch(texts, existing_triples=sample_triples)
        
        # Verify initial triples were passed
        call_args = mock_baml_extract.call_args
        assert len(call_args[1]["existing_triples"]) == len(sample_triples)
    
    @patch('spindle.extractor.async_b.ExtractTriples')
    @pytest.mark.asyncio
    async def test_extract_batch_maintains_order(self, mock_baml_extract, simple_ontology):
        """Test that batch extraction maintains input order."""
        # Create results in reverse order to test ordering
        results_list = []
        for i in range(3):
            results_list.append(ExtractionResult(
                triples=[
                    Triple(
                        subject=_create_test_entity(f"Person{i}", "Person"),
                        predicate="works_at",
                        object=_create_test_entity(f"Company{i}", "Organization"),
                        source=SourceMetadata(source_name=f"doc{i}"),
                        supporting_spans=[],
                        extraction_datetime=None
                    )
                ],
                reasoning=f"Extraction {i}"
            ))
        mock_baml_extract.side_effect = results_list
        
        extractor = SpindleExtractor(simple_ontology)
        texts = [
            ("Text 0", "doc0", None),
            ("Text 1", "doc1", None),
            ("Text 2", "doc2", None)
        ]
        results = await extractor.extract_batch(texts)
        
        # Verify order is maintained
        assert len(results) == 3
        assert results[0].triples[0].subject.name == "Person0"
        assert results[1].triples[0].subject.name == "Person1"
        assert results[2].triples[0].subject.name == "Person2"


class TestSpindleExtractorExtractBatchStream:
    """Tests for SpindleExtractor.extract_batch_stream method."""
    
    @patch('spindle.extractor.async_b.ExtractTriples')
    @pytest.mark.asyncio
    async def test_extract_batch_stream_yields_results(self, mock_baml_extract, simple_ontology):
        """Test that streaming batch extraction yields results as they complete."""
        mock_result_1 = ExtractionResult(
            triples=[
                Triple(
                    subject=_create_test_entity("Alice", "Person"),
                    predicate="works_at",
                    object=_create_test_entity("TechCorp", "Organization"),
                    source=SourceMetadata(source_name="doc1"),
                    supporting_spans=[],
                    extraction_datetime=None
                )
            ],
            reasoning="First"
        )
        mock_result_2 = ExtractionResult(
            triples=[
                Triple(
                    subject=_create_test_entity("Bob", "Person"),
                    predicate="works_at",
                    object=_create_test_entity("Google", "Organization"),
                    source=SourceMetadata(source_name="doc2"),
                    supporting_spans=[],
                    extraction_datetime=None
                )
            ],
            reasoning="Second"
        )
        mock_baml_extract.side_effect = [mock_result_1, mock_result_2]
        
        extractor = SpindleExtractor(simple_ontology)
        texts = [
            ("Alice works at TechCorp", "doc1", None),
            ("Bob works at Google", "doc2", None)
        ]
        
        # Collect yielded results
        results = []
        async for result in extractor.extract_batch_stream(texts):
            results.append(result)
        
        # Verify results were yielded
        assert len(results) == 2
        assert results[0].triples[0].subject.name == "Alice"
        assert results[1].triples[0].subject.name == "Bob"
    
    @patch('spindle.extractor.async_b.ExtractTriples')
    @pytest.mark.asyncio
    async def test_extract_batch_stream_sequential_consistency(self, mock_baml_extract, simple_ontology):
        """Test that streaming batch extraction maintains sequential consistency."""
        mock_result_1 = ExtractionResult(
            triples=[
                Triple(
                    subject=_create_test_entity("Alice", "Person"),
                    predicate="works_at",
                    object=_create_test_entity("TechCorp", "Organization"),
                    source=SourceMetadata(source_name="doc1"),
                    supporting_spans=[],
                    extraction_datetime=None
                )
            ],
            reasoning="First"
        )
        mock_result_2 = ExtractionResult(
            triples=[],
            reasoning="Second"
        )
        mock_baml_extract.side_effect = [mock_result_1, mock_result_2]
        
        extractor = SpindleExtractor(simple_ontology)
        texts = [
            ("Alice works at TechCorp", "doc1", None),
            ("Additional text", "doc2", None)
        ]
        
        # Collect yielded results
        results = []
        async for result in extractor.extract_batch_stream(texts):
            results.append(result)
        
        # Verify second call received first triple as existing_triples
        second_call_args = mock_baml_extract.call_args_list[1]
        existing_triples = second_call_args[1]["existing_triples"]
        assert len(existing_triples) == 1
        assert existing_triples[0].subject.name == "Alice"
    
    @patch('spindle.extractor.async_b.ExtractTriples')
    @pytest.mark.asyncio
    async def test_extract_batch_stream_maintains_order(self, mock_baml_extract, simple_ontology):
        """Test that streaming batch extraction yields results in order."""
        results_list = []
        for i in range(3):
            results_list.append(ExtractionResult(
                triples=[
                    Triple(
                        subject=_create_test_entity(f"Person{i}", "Person"),
                        predicate="works_at",
                        object=_create_test_entity(f"Company{i}", "Organization"),
                        source=SourceMetadata(source_name=f"doc{i}"),
                        supporting_spans=[],
                        extraction_datetime=None
                    )
                ],
                reasoning=f"Extraction {i}"
            ))
        mock_baml_extract.side_effect = results_list
        
        extractor = SpindleExtractor(simple_ontology)
        texts = [
            ("Text 0", "doc0", None),
            ("Text 1", "doc1", None),
            ("Text 2", "doc2", None)
        ]
        
        # Collect yielded results
        results = []
        async for result in extractor.extract_batch_stream(texts):
            results.append(result)
        
        # Verify order is maintained
        assert len(results) == 3
        assert results[0].triples[0].subject.name == "Person0"
        assert results[1].triples[0].subject.name == "Person1"
        assert results[2].triples[0].subject.name == "Person2"


class TestSpanComputationOptimization:
    """Tests for optimized span index computation."""
    
    def test_compute_all_span_indices_batch_processing(self):
        """Test that batch span computation processes all spans efficiently."""
        from spindle.extractor import _compute_all_span_indices
        
        source_text = "Alice Johnson works at TechCorp. Bob Smith works at Google."
        spans = [
            CharacterSpan(text="Alice Johnson works at TechCorp", start=None, end=None),
            CharacterSpan(text="Bob Smith works at Google", start=None, end=None),
            CharacterSpan(text="Already set", start=10, end=20)  # Already has indices
        ]
        
        result = _compute_all_span_indices(source_text, spans)
        
        # Verify all spans were processed
        assert len(result) == 3
        
        # First span should have indices computed
        assert result[0].start is not None
        assert result[0].end is not None
        assert result[0].start >= 0
        
        # Second span should have indices computed
        assert result[1].start is not None
        assert result[1].end is not None
        assert result[1].start >= 0
        
        # Third span should preserve existing indices
        assert result[2].start == 10
        assert result[2].end == 20
    
    def test_compute_all_span_indices_handles_unfound_spans(self):
        """Test that batch span computation handles spans that can't be found."""
        from spindle.extractor import _compute_all_span_indices
        
        source_text = "Alice works at TechCorp."
        spans = [
            CharacterSpan(text="This text is not in the source", start=None, end=None)
        ]
        
        result = _compute_all_span_indices(source_text, spans)
        
        # Verify unfound span has -1 indices
        assert len(result) == 1
        assert result[0].start == -1
        assert result[0].end == -1
    
    def test_compute_all_span_indices_empty_list(self):
        """Test that batch span computation handles empty span list."""
        from spindle.extractor import _compute_all_span_indices
        
        source_text = "Some text"
        spans = []
        
        result = _compute_all_span_indices(source_text, spans)
        
        # Verify empty list returns empty list
        assert len(result) == 0
        assert isinstance(result, list)
    
    def test_compute_all_span_indices_preserves_text(self):
        """Test that batch span computation preserves span text."""
        from spindle.extractor import _compute_all_span_indices
        
        source_text = "Alice Johnson works at TechCorp."
        spans = [
            CharacterSpan(text="Alice Johnson works at TechCorp", start=None, end=None)
        ]
        
        result = _compute_all_span_indices(source_text, spans)
        
        # Verify text is preserved
        assert len(result) == 1
        assert result[0].text == "Alice Johnson works at TechCorp"

