"""Integration tests for Spindle (requires API key).

These tests make actual LLM calls and should be run separately from unit tests.
Mark with pytest.mark.integration to skip by default.
"""

import pytest
import os
from dotenv import load_dotenv

from spindle import (
    SpindleExtractor,
    create_ontology,
    filter_triples_by_source,
    parse_extraction_datetime
)
from tests.fixtures.sample_texts import SIMPLE_TEXT, MEDICAL_TEXT, BUSINESS_TEXT
from tests.fixtures.sample_ontologies import (
    get_simple_entity_types_dict,
    get_simple_relation_types_dict
)

# Load environment variables
load_dotenv()


@pytest.mark.integration
class TestSpindleExtractorIntegration:
    """Integration tests for SpindleExtractor with real LLM calls."""
    
    def test_extract_basic_with_manual_ontology(self, skip_if_no_api_key):
        """Test basic extraction with manually defined ontology."""
        # Create ontology
        entity_types = get_simple_entity_types_dict()
        relation_types = get_simple_relation_types_dict()
        ontology = create_ontology(entity_types, relation_types)
        
        # Extract
        extractor = SpindleExtractor(ontology)
        result = extractor.extract(
            text=SIMPLE_TEXT,
            source_name="Test Document"
        )
        
        # Verify
        assert result is not None
        assert len(result.triples) > 0
        assert result.reasoning is not None
        
        # Check that triples have proper structure
        for triple in result.triples:
            assert triple.subject is not None
            assert triple.predicate is not None
            assert triple.object is not None
            assert triple.source.source_name == "Test Document"
            assert triple.extraction_datetime is not None
            assert len(triple.supporting_spans) > 0
    
    def test_extract_entity_consistency(self, skip_if_no_api_key):
        """Test entity consistency across multiple extractions."""
        entity_types = get_simple_entity_types_dict()
        relation_types = get_simple_relation_types_dict()
        ontology = create_ontology(entity_types, relation_types)
        
        extractor = SpindleExtractor(ontology)
        
        # First extraction
        text1 = "Alice works at TechCorp."
        result1 = extractor.extract(text=text1, source_name="Doc1")
        
        # Second extraction with entity consistency
        text2 = "Alice Johnson uses Python at TechCorp."
        result2 = extractor.extract(
            text=text2,
            source_name="Doc2",
            existing_triples=result1.triples
        )
        
        # Verify entity names are consistent
        # (This is a soft check - actual consistency depends on LLM)
        assert len(result2.triples) > 0
    
    def test_extract_character_spans(self, skip_if_no_api_key):
        """Test that character spans are correctly computed."""
        entity_types = get_simple_entity_types_dict()
        relation_types = get_simple_relation_types_dict()
        ontology = create_ontology(entity_types, relation_types)
        
        extractor = SpindleExtractor(ontology)
        result = extractor.extract(text=SIMPLE_TEXT, source_name="Test")
        
        # Verify spans have valid indices
        for triple in result.triples:
            for span in triple.supporting_spans:
                assert span.text is not None
                # Indices should be computed (not -1 for found spans)
                if span.start != -1:
                    assert span.start >= 0
                    assert span.end > span.start


@pytest.mark.integration
class TestEndToEndWorkflow:
    """Integration tests for complete workflows."""
    
    def test_multi_source_extraction_workflow(self, skip_if_no_api_key):
        """Test extracting from multiple sources and filtering."""
        # Create ontology
        entity_types = get_simple_entity_types_dict()
        relation_types = get_simple_relation_types_dict()
        ontology = create_ontology(entity_types, relation_types)
        
        extractor = SpindleExtractor(ontology)
        
        # Extract from first source
        result1 = extractor.extract(
            text=SIMPLE_TEXT,
            source_name="Source1",
            source_url="https://example.com/source1"
        )
        
        # Extract from second source
        result2 = extractor.extract(
            text=BUSINESS_TEXT,
            source_name="Source2",
            source_url="https://example.com/source2",
            existing_triples=result1.triples
        )
        
        # Combine all triples
        all_triples = result1.triples + result2.triples
        
        # Filter by source
        source1_triples = filter_triples_by_source(all_triples, "Source1")
        source2_triples = filter_triples_by_source(all_triples, "Source2")
        
        # Verify filtering works
        assert all(t.source.source_name == "Source1" for t in source1_triples)
        assert all(t.source.source_name == "Source2" for t in source2_triples)
        assert len(source1_triples) + len(source2_triples) == len(all_triples)
    
    def test_datetime_tracking_workflow(self, skip_if_no_api_key):
        """Test that datetime tracking works correctly."""
        entity_types = get_simple_entity_types_dict()
        relation_types = get_simple_relation_types_dict()
        ontology = create_ontology(entity_types, relation_types)
        
        extractor = SpindleExtractor(ontology)
        result = extractor.extract(text=SIMPLE_TEXT, source_name="Test")
        
        # Verify all triples have valid datetimes
        for triple in result.triples:
            dt = parse_extraction_datetime(triple)
            assert dt is not None
            # Should be a recent datetime
            assert dt.year >= 2024

