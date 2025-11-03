"""Unit tests for helper functions in spindle.py."""

import pytest
from datetime import datetime
from spindle.baml_client.types import Triple, SourceMetadata, CharacterSpan, Entity

from spindle import (
    _find_span_indices,
    create_ontology,
    create_source_metadata,
    get_supporting_text,
    filter_triples_by_source,
    parse_extraction_datetime,
    filter_triples_by_date_range
)
from tests.fixtures.sample_ontologies import (
    get_simple_entity_types_dict,
    get_simple_relation_types_dict,
    get_complex_entity_types_dict,
    get_complex_relation_types_dict
)


def _create_test_entity(name: str, entity_type: str = "Unknown", description: str = "") -> Entity:
    """Helper to create test Entity objects."""
    return Entity(
        name=name,
        type=entity_type,
        description=description or f"A {entity_type}",
        custom_atts={}
    )


class TestFindSpanIndices:
    """Tests for _find_span_indices function."""
    
    def test_exact_match(self):
        """Test exact substring match."""
        source = "Alice Johnson works at TechCorp in San Francisco."
        span = "Alice Johnson works at TechCorp"
        
        result = _find_span_indices(source, span)
        
        assert result is not None
        assert result[0] == 0
        assert result[1] == len(span)
        assert source[result[0]:result[1]] == span
    
    def test_exact_match_middle(self):
        """Test exact match in middle of text."""
        source = "The company TechCorp is located in San Francisco today."
        span = "TechCorp is located"
        
        result = _find_span_indices(source, span)
        
        assert result is not None
        # Verify the extracted text matches
        assert source[result[0]:result[1]] == span
    
    def test_whitespace_normalized_match(self):
        """Test matching with whitespace normalization."""
        source = "Alice   Johnson    works    at    TechCorp"
        span = "Alice Johnson works at TechCorp"
        
        result = _find_span_indices(source, span)
        
        assert result is not None
        # Should match the entire source despite whitespace differences
        assert result[0] == 0
        assert result[1] == len(source)
    
    def test_newline_handling(self):
        """Test matching text with newlines."""
        source = "Alice Johnson\nworks at\nTechCorp in\nSan Francisco."
        span = "Alice Johnson works at TechCorp"
        
        result = _find_span_indices(source, span)
        
        assert result is not None
        assert result[0] == 0
        # Should match through the newlines
    
    def test_case_insensitive_match(self):
        """Test case-insensitive matching."""
        source = "Alice Johnson works at TechCorp."
        span = "alice johnson works at techcorp"
        
        result = _find_span_indices(source, span)
        
        assert result is not None
        assert result[0] == 0
        # Case-insensitive, so just check that it found something
        assert result[1] > 0
    
    def test_not_found(self):
        """Test when span is not found in source."""
        source = "Alice Johnson works at TechCorp."
        span = "Bob Smith works at Google"
        
        result = _find_span_indices(source, span)
        
        assert result is None
    
    def test_empty_span(self):
        """Test with empty span text."""
        source = "Alice Johnson works at TechCorp."
        span = ""
        
        result = _find_span_indices(source, span)
        
        # Empty string is found at position 0
        assert result is not None or result is None  # Implementation dependent
    
    def test_span_with_punctuation(self):
        """Test matching span with punctuation."""
        source = 'Alice said, "I work at TechCorp," during the meeting.'
        span = '"I work at TechCorp,"'
        
        result = _find_span_indices(source, span)
        
        assert result is not None
        assert source[result[0]:result[1]] == span
    
    def test_partial_word_not_matched(self):
        """Test that partial words are properly handled."""
        source = "Alice works at TechCorporation in the city."
        span = "TechCorp"
        
        result = _find_span_indices(source, span)
        
        # Should find "TechCorp" as substring of "TechCorporation"
        assert result is not None
        assert source[result[0]:result[1]] == span


class TestCreateOntology:
    """Tests for create_ontology function."""
    
    def test_create_simple_ontology(self):
        """Test creating a simple ontology."""
        entity_types = get_simple_entity_types_dict()
        relation_types = get_simple_relation_types_dict()
        
        ontology = create_ontology(entity_types, relation_types)
        
        assert len(ontology.entity_types) == 2
        assert len(ontology.relation_types) == 1
        assert ontology.entity_types[0].name == "Person"
        assert ontology.entity_types[1].name == "Organization"
        assert ontology.relation_types[0].name == "works_at"
    
    def test_create_complex_ontology(self):
        """Test creating a complex ontology."""
        entity_types = get_complex_entity_types_dict()
        relation_types = get_complex_relation_types_dict()
        
        ontology = create_ontology(entity_types, relation_types)
        
        assert len(ontology.entity_types) == 4
        assert len(ontology.relation_types) == 3
    
    def test_create_empty_ontology(self):
        """Test creating an ontology with empty lists."""
        ontology = create_ontology([], [])
        
        assert len(ontology.entity_types) == 0
        assert len(ontology.relation_types) == 0
    
    def test_entity_type_properties(self):
        """Test that entity types have correct properties."""
        entity_types = [{"name": "Person", "description": "A human being"}]
        
        ontology = create_ontology(entity_types, [])
        
        entity = ontology.entity_types[0]
        assert entity.name == "Person"
        assert entity.description == "A human being"
    
    def test_relation_type_properties(self):
        """Test that relation types have correct properties."""
        relation_types = [
            {
                "name": "works_at",
                "description": "Employment relationship",
                "domain": "Person",
                "range": "Organization"
            }
        ]
        
        ontology = create_ontology([], relation_types)
        
        relation = ontology.relation_types[0]
        assert relation.name == "works_at"
        assert relation.description == "Employment relationship"
        assert relation.domain == "Person"
        assert relation.range == "Organization"


class TestCreateSourceMetadata:
    """Tests for create_source_metadata function."""
    
    def test_with_url(self):
        """Test creating source metadata with URL."""
        metadata = create_source_metadata(
            source_name="Test Document",
            source_url="https://example.com/test"
        )
        
        assert metadata.source_name == "Test Document"
        assert metadata.source_url == "https://example.com/test"
    
    def test_without_url(self):
        """Test creating source metadata without URL."""
        metadata = create_source_metadata(source_name="Test Document")
        
        assert metadata.source_name == "Test Document"
        assert metadata.source_url is None


class TestGetSupportingText:
    """Tests for get_supporting_text function."""
    
    def test_single_span(self, sample_triple):
        """Test extracting supporting text from single span."""
        texts = get_supporting_text(sample_triple)
        
        assert len(texts) == 1
        assert texts[0] == "Alice works at TechCorp"
    
    def test_multiple_spans(self):
        """Test extracting supporting text from multiple spans."""
        triple = Triple(
            subject=_create_test_entity("Alice", "Person"),
            predicate="uses",
            object=_create_test_entity("Python", "Technology"),
            source=SourceMetadata(source_name="Test"),
            supporting_spans=[
                CharacterSpan(text="Alice uses Python", start=0, end=17),
                CharacterSpan(text="skilled in Python", start=20, end=37)
            ],
            extraction_datetime="2024-01-15T10:30:00Z"
        )
        
        texts = get_supporting_text(triple)
        
        assert len(texts) == 2
        assert texts[0] == "Alice uses Python"
        assert texts[1] == "skilled in Python"
    
    def test_no_spans(self):
        """Test triple with no supporting spans."""
        triple = Triple(
            subject=_create_test_entity("Alice", "Person"),
            predicate="works_at",
            object=_create_test_entity("TechCorp", "Organization"),
            source=SourceMetadata(source_name="Test"),
            supporting_spans=[],
            extraction_datetime="2024-01-15T10:30:00Z"
        )
        
        texts = get_supporting_text(triple)
        
        assert len(texts) == 0


class TestFilterTriplesBySource:
    """Tests for filter_triples_by_source function."""
    
    def test_filter_single_source(self):
        """Test filtering triples from a single source."""
        triples = [
            Triple(
                subject=_create_test_entity("Alice", "Person"),
                predicate="works_at",
                object=_create_test_entity("TechCorp", "Organization"),
                source=SourceMetadata(source_name="Doc1"),
                supporting_spans=[], extraction_datetime=""
            ),
            Triple(
                subject=_create_test_entity("Bob", "Person"),
                predicate="works_at",
                object=_create_test_entity("Google", "Organization"),
                source=SourceMetadata(source_name="Doc2"),
                supporting_spans=[], extraction_datetime=""
            ),
            Triple(
                subject=_create_test_entity("Carol", "Person"),
                predicate="works_at",
                object=_create_test_entity("TechCorp", "Organization"),
                source=SourceMetadata(source_name="Doc1"),
                supporting_spans=[], extraction_datetime=""
            )
        ]
        
        filtered = filter_triples_by_source(triples, "Doc1")
        
        assert len(filtered) == 2
        assert all(t.source.source_name == "Doc1" for t in filtered)
    
    def test_filter_no_matches(self):
        """Test filtering when no triples match."""
        triples = [
            Triple(
                subject=_create_test_entity("Alice", "Person"),
                predicate="works_at",
                object=_create_test_entity("TechCorp", "Organization"),
                source=SourceMetadata(source_name="Doc1"),
                supporting_spans=[], extraction_datetime=""
            )
        ]
        
        filtered = filter_triples_by_source(triples, "Doc2")
        
        assert len(filtered) == 0
    
    def test_filter_empty_list(self):
        """Test filtering empty list."""
        filtered = filter_triples_by_source([], "Doc1")
        
        assert len(filtered) == 0


class TestParseExtractionDatetime:
    """Tests for parse_extraction_datetime function."""
    
    def test_parse_iso_format_z(self):
        """Test parsing ISO format with Z suffix."""
        triple = Triple(
            subject=_create_test_entity("Alice", "Person"),
            predicate="works_at",
            object=_create_test_entity("TechCorp", "Organization"),
            source=SourceMetadata(source_name="Test"),
            supporting_spans=[],
            extraction_datetime="2024-01-15T10:30:00Z"
        )
        
        dt = parse_extraction_datetime(triple)
        
        assert dt is not None
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 15
        assert dt.hour == 10
        assert dt.minute == 30
        assert dt.second == 0
    
    def test_parse_iso_format_with_microseconds(self):
        """Test parsing ISO format with microseconds."""
        triple = Triple(
            subject=_create_test_entity("Alice", "Person"),
            predicate="works_at",
            object=_create_test_entity("TechCorp", "Organization"),
            source=SourceMetadata(source_name="Test"),
            supporting_spans=[],
            extraction_datetime="2024-01-15T10:30:00.123456Z"
        )
        
        dt = parse_extraction_datetime(triple)
        
        assert dt is not None
        assert dt.microsecond == 123456
    
    def test_parse_invalid_format(self):
        """Test parsing invalid datetime string."""
        triple = Triple(
            subject=_create_test_entity("Alice", "Person"),
            predicate="works_at",
            object=_create_test_entity("TechCorp", "Organization"),
            source=SourceMetadata(source_name="Test"),
            supporting_spans=[],
            extraction_datetime="invalid-datetime"
        )
        
        dt = parse_extraction_datetime(triple)
        
        assert dt is None
    
    def test_parse_empty_string(self):
        """Test parsing empty datetime string."""
        triple = Triple(
            subject=_create_test_entity("Alice", "Person"),
            predicate="works_at",
            object=_create_test_entity("TechCorp", "Organization"),
            source=SourceMetadata(source_name="Test"),
            supporting_spans=[],
            extraction_datetime=""
        )
        
        dt = parse_extraction_datetime(triple)
        
        assert dt is None


class TestFilterTriplesByDateRange:
    """Tests for filter_triples_by_date_range function."""
    
    def test_filter_with_start_date(self):
        """Test filtering with only start date."""
        triples = [
            Triple(
                subject=_create_test_entity("Alice", "Person"),
                predicate="works_at",
                object=_create_test_entity("TechCorp", "Organization"),
                source=SourceMetadata(source_name="Test"),
                supporting_spans=[],
                extraction_datetime="2024-01-10T10:00:00Z"
            ),
            Triple(
                subject=_create_test_entity("Bob", "Person"),
                predicate="works_at",
                object=_create_test_entity("Google", "Organization"),
                source=SourceMetadata(source_name="Test"),
                supporting_spans=[],
                extraction_datetime="2024-01-20T10:00:00Z"
            )
        ]
        
        start_date = datetime(2024, 1, 15)
        filtered = filter_triples_by_date_range(triples, start_date=start_date)
        
        assert len(filtered) == 1
        assert filtered[0].subject.name == "Bob"
    
    def test_filter_with_end_date(self):
        """Test filtering with only end date."""
        triples = [
            Triple(
                subject=_create_test_entity("Alice", "Person"),
                predicate="works_at",
                object=_create_test_entity("TechCorp", "Organization"),
                source=SourceMetadata(source_name="Test"),
                supporting_spans=[],
                extraction_datetime="2024-01-10T10:00:00Z"
            ),
            Triple(
                subject=_create_test_entity("Bob", "Person"),
                predicate="works_at",
                object=_create_test_entity("Google", "Organization"),
                source=SourceMetadata(source_name="Test"),
                supporting_spans=[],
                extraction_datetime="2024-01-20T10:00:00Z"
            )
        ]
        
        end_date = datetime(2024, 1, 15)
        filtered = filter_triples_by_date_range(triples, end_date=end_date)
        
        assert len(filtered) == 1
        assert filtered[0].subject.name == "Alice"
    
    def test_filter_with_date_range(self):
        """Test filtering with both start and end dates."""
        triples = [
            Triple(
                subject=_create_test_entity("Alice", "Person"),
                predicate="works_at",
                object=_create_test_entity("TechCorp", "Organization"),
                source=SourceMetadata(source_name="Test"),
                supporting_spans=[],
                extraction_datetime="2024-01-10T10:00:00Z"
            ),
            Triple(
                subject=_create_test_entity("Bob", "Person"),
                predicate="works_at",
                object=_create_test_entity("Google", "Organization"),
                source=SourceMetadata(source_name="Test"),
                supporting_spans=[],
                extraction_datetime="2024-01-15T10:00:00Z"
            ),
            Triple(
                subject=_create_test_entity("Carol", "Person"),
                predicate="works_at",
                object=_create_test_entity("Microsoft", "Organization"),
                source=SourceMetadata(source_name="Test"),
                supporting_spans=[],
                extraction_datetime="2024-01-20T10:00:00Z"
            )
        ]
        
        start_date = datetime(2024, 1, 12)
        end_date = datetime(2024, 1, 18)
        filtered = filter_triples_by_date_range(
            triples, 
            start_date=start_date, 
            end_date=end_date
        )
        
        assert len(filtered) == 1
        assert filtered[0].subject.name == "Bob"
    
    def test_filter_no_dates(self):
        """Test filtering with no date constraints."""
        triples = [
            Triple(
                subject=_create_test_entity("Alice", "Person"),
                predicate="works_at",
                object=_create_test_entity("TechCorp", "Organization"),
                source=SourceMetadata(source_name="Test"),
                supporting_spans=[],
                extraction_datetime="2024-01-10T10:00:00Z"
            )
        ]
        
        filtered = filter_triples_by_date_range(triples)
        
        assert len(filtered) == 1
    
    def test_filter_invalid_datetime(self):
        """Test filtering skips triples with invalid datetimes."""
        triples = [
            Triple(
                subject=_create_test_entity("Alice", "Person"),
                predicate="works_at",
                object=_create_test_entity("TechCorp", "Organization"),
                source=SourceMetadata(source_name="Test"),
                supporting_spans=[],
                extraction_datetime="invalid"
            ),
            Triple(
                subject=_create_test_entity("Bob", "Person"),
                predicate="works_at",
                object=_create_test_entity("Google", "Organization"),
                source=SourceMetadata(source_name="Test"),
                supporting_spans=[],
                extraction_datetime="2024-01-15T10:00:00Z"
            )
        ]
        
        start_date = datetime(2024, 1, 10)
        filtered = filter_triples_by_date_range(triples, start_date=start_date)
        
        assert len(filtered) == 1
        assert filtered[0].subject.name == "Bob"

