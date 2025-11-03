"""Unit tests for SpindleExtractor class."""

import pytest
from unittest.mock import patch, MagicMock
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

