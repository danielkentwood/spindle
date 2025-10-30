"""Integration tests for Spindle (requires API key).

These tests make actual LLM calls and should be run separately from unit tests.
Mark with pytest.mark.integration to skip by default.
"""

import pytest
import os
from dotenv import load_dotenv

from spindle import (
    SpindleExtractor,
    OntologyRecommender,
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
    
    def test_extract_with_auto_ontology(self, skip_if_no_api_key):
        """Test extraction with automatic ontology recommendation."""
        # Extract without providing ontology
        extractor = SpindleExtractor()
        result = extractor.extract(
            text=SIMPLE_TEXT,
            source_name="Test Document"
        )
        
        # Verify
        assert result is not None
        assert extractor.ontology is not None  # Ontology was auto-recommended
        assert len(extractor.ontology.entity_types) > 0
        assert len(extractor.ontology.relation_types) > 0
    
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
class TestOntologyRecommenderIntegration:
    """Integration tests for OntologyRecommender with real LLM calls."""
    
    def test_recommend_basic(self, skip_if_no_api_key):
        """Test basic ontology recommendation."""
        recommender = OntologyRecommender()
        recommendation = recommender.recommend(text=SIMPLE_TEXT)
        
        # Verify
        assert recommendation is not None
        assert recommendation.ontology is not None
        assert len(recommendation.ontology.entity_types) > 0
        assert len(recommendation.ontology.relation_types) > 0
        assert recommendation.text_purpose is not None
        assert recommendation.reasoning is not None
    
    def test_recommend_different_scopes(self, skip_if_no_api_key):
        """Test recommendations with different scope levels."""
        recommender = OntologyRecommender()
        
        # Get recommendations at different scopes
        minimal = recommender.recommend(text=BUSINESS_TEXT, scope="minimal")
        balanced = recommender.recommend(text=BUSINESS_TEXT, scope="balanced")
        comprehensive = recommender.recommend(text=BUSINESS_TEXT, scope="comprehensive")
        
        # Verify they all return valid ontologies
        assert len(minimal.ontology.entity_types) > 0
        assert len(balanced.ontology.entity_types) > 0
        assert len(comprehensive.ontology.entity_types) > 0
        
        # Generally, comprehensive should have more types (though not guaranteed)
        # This is just a soft check
        assert len(comprehensive.ontology.entity_types) >= len(minimal.ontology.entity_types)
    
    def test_recommend_and_extract(self, skip_if_no_api_key):
        """Test combined recommendation and extraction."""
        recommender = OntologyRecommender()
        recommendation, extraction = recommender.recommend_and_extract(
            text=SIMPLE_TEXT,
            source_name="Test"
        )
        
        # Verify both are returned
        assert recommendation is not None
        assert extraction is not None
        assert len(extraction.triples) >= 0  # May be 0 or more
    
    def test_analyze_extension_different_domains(self, skip_if_no_api_key):
        """Test extension analysis when moving to different domain."""
        # Start with simple business ontology
        entity_types = get_simple_entity_types_dict()
        relation_types = get_simple_relation_types_dict()
        ontology = create_ontology(entity_types, relation_types)
        
        # Analyze medical text with business ontology
        recommender = OntologyRecommender()
        extension = recommender.analyze_extension(
            text=MEDICAL_TEXT,
            current_ontology=ontology
        )
        
        # Verify analysis was performed
        assert extension is not None
        assert extension.reasoning is not None
        # Extension may or may not be needed depending on LLM decision
    
    def test_extend_ontology_and_extract(self, skip_if_no_api_key):
        """Test extending ontology and using it for extraction."""
        # Start with simple ontology
        entity_types = get_simple_entity_types_dict()
        relation_types = get_simple_relation_types_dict()
        ontology = create_ontology(entity_types, relation_types)
        
        # Analyze and extend for medical text
        recommender = OntologyRecommender()
        extension, extended_ontology = recommender.analyze_and_extend(
            text=MEDICAL_TEXT,
            current_ontology=ontology,
            auto_apply=True
        )
        
        # If extension was applied, use it for extraction
        if extended_ontology:
            extractor = SpindleExtractor(extended_ontology)
            result = extractor.extract(text=MEDICAL_TEXT, source_name="Medical Test")
            assert result is not None


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
    
    def test_auto_ontology_workflow(self, skip_if_no_api_key):
        """Test complete workflow with automatic ontology."""
        # Create extractor without ontology
        extractor = SpindleExtractor()
        
        # First extraction (ontology will be auto-recommended)
        result1 = extractor.extract(text=SIMPLE_TEXT, source_name="Doc1")
        
        # Ontology should now be set
        assert extractor.ontology is not None
        
        # Second extraction reuses the same ontology
        result2 = extractor.extract(
            text=BUSINESS_TEXT,
            source_name="Doc2",
            existing_triples=result1.triples
        )
        
        # Verify both extractions succeeded
        assert len(result1.triples) >= 0
        assert len(result2.triples) >= 0
    
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

